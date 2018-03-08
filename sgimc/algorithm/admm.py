"""The ADMM step function."""
from math import sqrt

import numpy as np

from scipy.optimize import fmin_l_bfgs_b

from ..ops import shrink

from .misc import f_fused


def simple_cg(Ax, b, x, rtol=1e-5, atol=1e-8, args=()):
    """Simple Conjugate gradient sovler."""
    # Simple Conjugate Gradient Method: the result overwrites `x`!
    # b - Ax(x) here is - \nabla f
    return trcg(Ax, b - Ax(x), x, tr_delta=0, rtol=rtol, atol=atol, args=args)


def trcg(Ax, r, x, tr_delta=0, rtol=1e-5, atol=1e-8, args=()):
    """Simple Conjugate gradient sovler with trust region control.

    For the given `x` and `r` solves `r = A(z - x)` and on termination
    updates `r` and `x` inplace with the final residual and solution `z`,
    respectively.
    """

    p, rtr = np.zeros_like(r), 1.0
    cg_tol = np.linalg.norm(r) * rtol + atol
    tr_delta_sq = tr_delta ** 2

    for n_iter in range(len(x)):
        # ddot(&n, r, &inc, r, &inc);
        rtr, rtr_old = np.dot(r, r), rtr
        if sqrt(rtr) < cg_tol:
            break

        # dscal(&n, &(rtr / rtr_old), p, &inc);
        p *= rtr / rtr_old
        # daxpy(&n, &one, r, &1, p, &1);
        p += r

        Ap = Ax(p, *args)

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
                    eta = (q - sqrt(max(q * q + tr_delta_sq - xTx, 0))) / p_nrm

                    # reproject onto the boundary of the region
                    r += eta * Ap
                    x -= eta * p
                else:
                    # this never happens maybe due to CG iteration properties
                    pass
                break
            # end if
        # end if
    # end for

    return n_iter


def sub_0_cg(D, Obj, W0, eta=1e0, rtol=1e-3, atol=1e-5, update=False):
    if update:
        Obj.update(W0)

    grad = (Obj.grad() + (W0 - D) / eta).reshape(-1)
    def f_hessp(s, Obj, W, eta):
        return Obj.hess_v(s.reshape(W.shape)).reshape(-1) + s / eta

    r, delta = -grad, np.zeros_like(W0.reshape(-1))
    trcg(f_hessp, r, delta, rtol=rtol, atol=atol,
         tr_delta=0, args=(Obj, W0, eta))

    return W0 + delta.reshape(W0.shape)


def sub_0_lbfgs(D, Obj, W0=None, eta=1e0, tol=1e-8):
    """Solve the Sub_0 subproblem using L-BFGS."""
    if W0 is None:
        W0 = D

    # Run the L-BFGS
    W_star, F_star, info = fmin_l_bfgs_b(
        f_fused, W0.reshape(-1), fprime=None,
        # f_value, W0.reshape(-1), f_prime,
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
            WW = sub_0_cg(ZZ - LL, Obj, W0=W0, eta=eta)

        elif method == "l-bfgs":
            WW = sub_0_lbfgs(ZZ - LL, Obj, W0=WW_old, eta=eta, tol=1e-8)

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
