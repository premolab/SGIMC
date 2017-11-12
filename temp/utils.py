import numpy as np
from math import sqrt

from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.optimize import fmin_l_bfgs_b, fmin_ncg, line_search

# build lib to import!
from imc_sparse_ops.imc_ops import op_s, op_d, shrink, shrink_row


def sigmoid(x):
    out = 1 / (1 + np.exp(- abs(x)))
    return np.where(x > 0, out, 1 - out)

    
def f_value(x, Obj, D, eta, update=True):
    # update the objective with the current argument
    if update:
        Obj.update(x.reshape(D.shape))

    # Get the L2 term
    ridge_value = np.linalg.norm(x - D.reshape(-1), 2) ** 2
    return Obj.value() + 0.5 * ridge_value / eta


def f_prime(x, Obj, D, eta, update=True):
    # update the objective with the current argument
    if update:
        Obj.update(x.reshape(D.shape))

    # Get the gradient of the L2 term
    ridge_prime = x - D.reshape(-1)
    return Obj.grad().reshape(-1) + ridge_prime / eta


def f_hessp(x, p, Obj, D, eta, update=True):
    # update the objective with the current argument
    if update:
        Obj.update(x.reshape(D.shape))

    # Get the gradient of the L2 term
    return Obj.hess_v(p.reshape(D.shape)).reshape(-1) + p / eta


def f_fused(x, Obj, D, eta, update=True):
    # update the objective with the current argument
    if update:
        Obj.update(x.reshape(D.shape))

    # Get the L2 term
    ridge_prime = x - D.reshape(-1)
    ridge_value = np.linalg.norm(ridge_prime, 2) ** 2

    return Obj.value() + 0.5 * ridge_value / eta, \
           Obj.grad().reshape(-1) + ridge_prime / eta


def simple_cg(Ax, b, x, tol=1e-8):
    # Simple Conjugate Gradient Method: the result overwrites `x`!
    r = b - Ax(x)
    p = r.copy()

    rtr = np.dot(r, r)
    if sqrt(rtr) < tol:
        return 0

    for n_iter in range(len(x)):
        Ap = Ax(p)

        alpha = rtr / np.dot(p, Ap)     # ddot(&n, p, &inc, Ap, &inc);
        x += alpha * p                  # daxpy(&n, &alpha, p, &inc, x, &inc);
        r -= alpha * Ap                 # daxpy(&n, &( -alpha ), Ap, &inc, r, &inc);
        p /= rtr                        # dscal(&n, &(1 / rtr), p, &inc);

        rtr = np.dot(r, r)              # ddot(&n, r, &inc, r, &inc);
        if sqrt(rtr) < tol:
            break

        p *= rtr                        # dscal(&n, &rtr, p, &inc);
        p += r                          # daxpy(&n, &one, r, &1, p, &1);
    # end for

    return n_iter


def sub_0(D, Obj, eta=1e0, tol=1e-8, linearize=False):
    if linearize:
        return D - eta * Obj.grad()

    # set up the CG arguments
    x = D.reshape(-1).copy()
    b = x / eta - Obj.grad().reshape(-1)

    Ax = lambda z: z / eta + Obj.hess_v(z.reshape(D.shape)).reshape(-1)

    n_iter = simple_cg(Ax, b, x, tol=tol)

    return x.reshape(D.shape)


def solver_sub_0(D, Obj, x0=None, eta=1e0,
                 tol=1e-8, method="l-bfgs"):
    if x0 is None:
        x0 = D

    # Run the L-BFGS
    if method == "ncg":
        W_star, F_star, *info = fmin_ncg(
            f_value, x0.reshape(-1),
            fprime=f_prime, fhess_p=f_hessp,
            args=(Obj, D, eta), avextol=tol,
            full_output=True, disp=False, retall=False)

    elif method == "l-bfgs":
        W_star, F_star, info = fmin_l_bfgs_b(
            f_fused, x0.reshape(-1), fprime=None,  # f_value, x0.reshape(-1), f_prime,
            args=(Obj, D, eta), approx_grad=False,
            pgtol=tol, iprint=0)

    # end if

    return W_star.reshape(D.shape)

def sub_m(D, C_lasso, C_group, C_ridge, eta=1e0):
    return shrink(D, C_lasso * eta,
                     C_group * eta,
                     C_ridge * eta)


def QA_argmin(D, Obj, tol=1e-8):

    # set up the CG arguments
    x = D.reshape(-1).copy()
    b = - Obj.grad().reshape(-1)
    Ax = lambda x: Obj.hess_v(x.reshape(D.shape)).reshape(-1)

    n_iter = simple_cg(Ax, b, x, tol=tol)
    return x.reshape(D.shape)