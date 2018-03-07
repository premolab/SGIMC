"""The ADMM step function."""
from math import sqrt

import numpy as np

from scipy.optimize import fmin_l_bfgs_b

from ..ops import shrink

from .misc import f_fused


def simple_cg(Ax, b, x, tr_delta=0, rtol=1e-5, atol=1e-8):
    """Simple Conjugate gradient sovler."""
    # Simple Conjugate Gradient Method: the result overwrites `x`!
    # b - Ax(x) here is - \nabla f
    r, rtr = b - Ax(x), 1.0
    p = np.zeros_like(r)

    cg_tol_sq = np.dot(r, r) * rtol + atol
    tr_delta_sq = tr_delta ** 2

    for n_iter in range(len(x)):
        # ddot(&n, r, &inc, r, &inc);
        rtr, rtr_old = np.dot(r, r), rtr
        if rtr < cg_tol_sq:
            break

        # dscal(&n, &(rtr / rtr_old), p, &inc);
        p *= rtr / rtr_old
        # daxpy(&n, &one, r, &1, p, &1);
        p += r

        Ap = Ax(p)

        # ddot(&n, p, &inc, Ap, &inc);
        alpha = rtr / np.dot(p, Ap)
        # daxpy(&n, &alpha, p, &inc, x, &inc);
        x += alpha * p
        # daxpy(&n, &( -alpha ), Ap, &inc, r, &inc);
        r -= alpha * Ap

        # check trust region
        if tr_delta_sq > 0:
            xTx = np.dot(x, x)
            if xTx > tr_delta_sq:
                xTp = np.dot(x, p)
                if(xTp > 0):
                    # backtrack into the trust region
                    p_nrm = np.linalg.norm(p)

                    q = xTp / p_nrm
                    eta = (q - sqrt(q * q + tr_delta_sq - xTx)) / p_nrm

                    # reproject onto the boundary of the region
                    r += eta * Ap
                    x -= eta * p
                else:
                    # this never happens maybe due to CG iteration properties
                    pass
            break
    # end for

    return n_iter


def sub_0_cg(D, Obj, x0=None, eta=1e0, tol=1e-3):
    """Solve the Sub_0 subproblem using CG."""
    if x0 is None:
        x0 = D

    # set up the CG arguments
    x = x0.reshape(-1).copy()
    b = x / eta - Obj.grad().reshape(-1)

    def hess_v(p):
        return Obj.hess_v(p.reshape(D.shape)).reshape(-1) + p / eta

    simple_cg(hess_v, b, x, rtol=tol, tr_delta=0.)
    # assert np.allclose(Ax(x), b)

    return x.reshape(D.shape)


def sub_0_lbfgs(D, Obj, x0=None, eta=1e0, tol=1e-8):
    """Solve the Sub_0 subproblem using L-BFGS."""
    if x0 is None:
        x0 = D

    # Run the L-BFGS
    W_star, F_star, info = fmin_l_bfgs_b(
        f_fused, x0.reshape(-1), fprime=None,
        # f_value, x0.reshape(-1), f_prime,
        args=(Obj, D, eta), approx_grad=False,
        pgtol=tol, iprint=0)

    return W_star.reshape(D.shape)


def sub_m(D, C_lasso, C_group, C_ridge, eta=1e0):
    """Solve the Sub_m subproblem."""
    return shrink(D, C_lasso * eta, C_group * eta, C_ridge * eta)


def step(Obj, W0, C, eta, method="l-bfgs", sparse=True,
         n_iterations=50, rtol=1e-5, atol=1e-8):
    """Perform ADMM on the IMC objective starting at W0."""
    if method not in ("cg", "l-bfgs"):
        raise ValueError("""Unrecognized method `%s`""" % method)

    C_lasso, C_group, C_ridge = C
    iteration, n_sq_dim = 0, sqrt(np.prod(W0.shape))
    LL, WW, ZZ = np.zeros_like(W0), W0.copy(), W0.copy()

    while iteration < n_iterations:
        # no need to copy WW or ZZ, their updates are not inplace
        WW_old, ZZ_old = WW, ZZ

        # Get the tolerances
        tol_p = n_sq_dim * atol + rtol * max(
            np.linalg.norm(WW_old.reshape(-1), 2),
            np.linalg.norm(ZZ_old.reshape(-1), 2))

        tol_d = n_sq_dim * atol + rtol * \
            np.linalg.norm(LL.reshape(-1), 2)  # used to be ZZ_old

        # ADMM steps
        ZZ = sub_m(WW + LL, C_lasso, C_group, C_ridge, eta=eta)
        if method == "cg":
            WW = sub_0_cg((ZZ - W0) - LL, Obj, eta=eta, tol=rtol) + W0

        elif method == "l-bfgs":
            WW = sub_0_lbfgs(ZZ - LL, Obj, x0=WW_old, eta=eta, tol=1e-8)

        # end if

        LL += WW - ZZ

        iteration += 1

        # residuals: primal and dual feasibility.
        resid_p = np.linalg.norm((WW - ZZ).reshape(-1), 2)
        resid_d = np.linalg.norm((ZZ_old - ZZ).reshape(-1), 2) / eta
        if resid_p <= tol_p and resid_d <= tol_d:
            break

        # end if
    # end while

    return ZZ if sparse else WW
