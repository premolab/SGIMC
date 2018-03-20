"""Various evalutaions."""
import numpy as np


def f_valp(x, Obj, D, eta, C_ridge=1.0):
    """Compute the forward pass, predictions and the objective."""
    # Compute the forward pass.
    W = x.reshape(D.shape)
    Obj.forward(W)
    return Obj.value() + 0.5 * (np.linalg.norm(W - D, ord="fro")**2) / eta \
                       + 0.5 * C_ridge * np.linalg.norm(W, ord="fro")**2


def f_grad(x, Obj, D, eta, C_ridge=1.0):
    """Get the gradient after the recent forward pass."""
    # Update grad and hess statistics.
    W = x.reshape(D.shape)
    Obj.backward()
    return (Obj.grad() + (W - D) / eta + C_ridge * W).reshape(-1)


def f_hess(v, Obj, D, eta, C_ridge=1.0):
    """Compute the hessian-vector product for Obj."""
    mult = (C_ridge * eta + 1) / eta
    return Obj.hess_v(v.reshape(D.shape)).reshape(-1) + v * mult


def f_fused(x, Obj, D, eta, C_ridge=1.0):
    """Compute the objective value and gradient."""
    W = x.reshape(D.shape)

    # Compute the forward pass.
    Obj.forward(W)
    # Update grad and hess statistics.
    Obj.backward()

    delta = W - D
    valp = Obj.value() + 0.5 * (np.linalg.norm(delta, ord="fro")**2) / eta
    grad = Obj.grad() + delta / eta

    if C_ridge > 0:
        valp += 0.5 * C_ridge * np.linalg.norm(W, ord="fro")**2
        grad += C_ridge * W

    return valp, grad.reshape(-1)
