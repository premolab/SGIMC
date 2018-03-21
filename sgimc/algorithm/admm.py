"""The ADMM step function."""
from math import sqrt

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from .tron import tron, trcg
from .misc import f_valp, f_grad, f_hess, f_fused

from ..ops import shrink


def sub_0_lbfgs(D, Obj, W0, eta=1e0, C=1.0, tol=1e-3):
    """Solve the Sub_0 subproblem using L-BFGS."""
    W, F_star, info = fmin_l_bfgs_b(
        f_fused, W0.reshape(-1), fprime=None,
        args=(Obj, D, eta, C), approx_grad=False,
        pgtol=tol, iprint=0)

    return W.reshape(D.shape)


def sub_0_cg(D, Obj, W0, eta=1e0, C=1.0, rtol=1e-3, atol=1e-5):
    """Sub_0 problem using CG."""
    args = (Obj, D, eta, C)

    Obj.forward(W0)
    r, delta = -f_grad(W0, *args), np.zeros_like(W0)
    trcg(f_hess, r, delta.reshape(-1), args=args, tr_delta=0,
         n_iterations=20, rtol=rtol, atol=atol)

    return W0 + delta


def sub_0_tron(D, Obj, W0, eta=1e0, C=1.0, rtol=5e-2, atol=1e-4,
               verbose=False):
    """Solve the Sub_0 problem with tron+cg is in lelm-imf."""
    W, f_call = W0.copy(), (f_valp, f_grad, f_hess)
    tron(f_call, W.reshape(-1), n_iterations=5, rtol=rtol, atol=atol,
         args=(Obj, D, eta, C), verbose=verbose)
    return W


def sub_m(D, C_lasso, C_group, C_ridge, eta=1e0):
    """Solve the Sub_m subproblem."""
    return shrink(D, C_lasso * eta, C_group * eta, C_ridge * eta)


def step(Obj, W0, C, eta, method="l-bfgs", sparse=True,
         n_iterations=50, rtol=1e-5, atol=1e-8):
    """Perform ADMM on the IMC objective starting at W0."""
    if method not in ("cg", "l-bfgs", "tron"):
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

        # ADMM prox step for the regularizer
        ZZ = sub_m(WW + LL, C_lasso, C_group, 0., eta=eta)

        # ADMM prox step for the loss
        if method == "cg":
            WW = sub_0_cg(ZZ - LL, Obj, W0=WW_old, C=C_ridge, eta=eta)

        elif method == "tron":
            WW = sub_0_tron(ZZ - LL, Obj, W0=WW_old, C=C_ridge, eta=eta)

        elif method == "l-bfgs":
            WW = sub_0_lbfgs(ZZ - LL, Obj, W0=WW_old,
                             C=C_ridge, eta=eta, tol=1e-8)

        # end if

        # ADMM backward gradinet step for the dual
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
