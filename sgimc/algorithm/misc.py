"""Various evalutaions."""

import numpy as np


def f_value(x, Obj, D, eta, update=True):
    """Compute the objective value."""
    # update the objective with the current argument
    if update:
        Obj.update(x.reshape(D.shape))

    # Get the L2 term
    ridge_value = np.linalg.norm(x - D.reshape(-1), 2) ** 2
    return Obj.value() + 0.5 * ridge_value / eta


def f_prime(x, Obj, D, eta, update=True):
    """Compute the gradient."""
    # update the objective with the current argument
    if update:
        Obj.update(x.reshape(D.shape))

    # Get the gradient of the L2 term
    ridge_prime = x - D.reshape(-1)
    return Obj.grad().reshape(-1) + ridge_prime / eta


def f_hessp(x, p, Obj, D, eta, update=True):
    """Compute the hessian-vector product."""
    # update the objective with the current argument
    if update:
        Obj.update(x.reshape(D.shape))

    # Get the gradient of the L2 term
    return Obj.hess_v(p.reshape(D.shape)).reshape(-1) + p / eta


def f_fused(x, Obj, D, eta, update=True):
    """Compute the objective value and gradient."""
    # update the objective with the current argument
    if update:
        Obj.update(x.reshape(D.shape))

    # Get the L2 term
    ridge_prime = x - D.reshape(-1)
    ridge_value = np.linalg.norm(ridge_prime, 2) ** 2

    return \
        Obj.value() + 0.5 * ridge_value / eta, \
        Obj.grad().reshape(-1) + ridge_prime / eta
