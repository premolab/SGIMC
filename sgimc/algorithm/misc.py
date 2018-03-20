"""Various evalutaions."""
import numpy as np


def f_valp(x, Obj, D, eta, C_ridge=1.0):
    """Compute the forward pass, predictions and the objective."""
    # Compute the forward pass.
    W = x.reshape(D.shape)
    Obj.forward(W)

    valp = Obj.value()
    if eta > 0:
        valp += 0.5 * (np.linalg.norm(W - D, ord="fro")**2) / eta

    if C_ridge > 0:
        valp += 0.5 * C_ridge * np.linalg.norm(W, ord="fro")**2

    return valp


def f_grad(x, Obj, D, eta, C_ridge=1.0):
    """Get the gradient after the recent forward pass."""
    # Update grad and hess statistics.
    W = x.reshape(D.shape)
    Obj.backward()

    grad = Obj.grad()
    if eta > 0:
        grad += (W - D) / eta

    if C_ridge > 0:
        grad += C_ridge * W

    return grad.reshape(-1)


def f_hess(v, Obj, D, eta, C_ridge=1.0):
    """Compute the hessian-vector product for Obj."""
    W = v.reshape(D.shape)
    hess_v = Obj.hess_v(W)
    if eta > 0:
        hess_v += W / eta

    if C_ridge > 0:
        hess_v += C_ridge * W

    return hess_v.reshape(-1)


def f_fused(x, Obj, D, eta, C_ridge=1.0):
    """Compute the objective value and gradient."""
    W = x.reshape(D.shape)

    # Compute the forward pass.
    Obj.forward(W)
    # Update grad and hess statistics.
    Obj.backward()

    valp, grad = Obj.value(), Obj.grad()
    if eta > 0:
        delta = W - D
        valp += 0.5 * (np.linalg.norm(delta, ord="fro")**2) / eta
        grad += delta / eta

    if C_ridge > 0:
        valp += 0.5 * C_ridge * (np.linalg.norm(W, ord="fro")**2)
        grad += C_ridge * W

    return valp, grad.reshape(-1)
