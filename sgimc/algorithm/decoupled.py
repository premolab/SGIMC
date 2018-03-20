"""Gradient descent step function.

A step function for gradient descent on block-diagonal quadratic
approximaition.
"""
import numpy as np

from scipy.optimize import line_search

from ..ops import shrink_row

from .misc import f_valp, f_grad


def step(Obj, W0, C, eta, rtol=1e-5, atol=1e-8):
    if not isinstance(eta, list):
        eta = W0.shape[0] * [eta]

    assert W0.shape[0] == len(eta)

    C_lasso, C_group, C_ridge = C

    # The gradient of the loss at the current "solution" W0
    Gt = Obj.grad()

    # Get a descent direction
    W_new = np.empty_like(W0)
    for l, eta_l in zip(range(W0.shape[0]), eta):
        # compute the descent direction
        W_new[l] = shrink_row(W0[l] - Gt[l] * eta_l,
                              C_lasso * eta_l,
                              C_group * eta_l,
                              C_ridge * eta_l)

    return W_new

    # pk = W_new - W0

    # alpha = line_search(f_valp, f_grad, args=(Obj, W0, eta),
    #                     xk=W0.reshape(-1), pk=pk.reshape(-1),
    #                     gfk=Gt.reshape(-1))[0] or 1e-2

    # return W0 + alpha * pk
