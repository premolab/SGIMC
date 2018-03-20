"""Trust regon optimizer."""
from math import sqrt
import numpy as np


def trcg(Ax, r, x, n_iterations=100, tr_delta=0, rtol=1e-5, atol=1e-8,
         args=(), verbose=False):
    """Simple Conjugate gradient sovler with trust region control.

    For the given `x` and `r` solves `r = A(z - x)` and on termination
    updates `r` and `x` inplace with the final residual and solution `z`,
    respectively.
    """
    p, iteration = r.copy(), 0
    tr_delta_sq = tr_delta ** 2

    rtr, rtr_old = np.dot(r, r), 1.0
    cg_tol = sqrt(rtr) * rtol + atol
    n_iterations = min(n_iterations, len(x))
    while iteration < n_iterations and sqrt(rtr) > cg_tol:
        Ap = Ax(p, *args)
        iteration += 1

        if verbose:
            print("""iter %2d |Ap| %5.3e |p| %5.3e |r| %5.3e |x| %5.3e """
                  """beta %5.3e""" %
                  (iteration, np.linalg.norm(Ap), np.linalg.norm(p),
                   np.linalg.norm(r), np.linalg.norm(x), rtr / rtr_old))

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

        # ddot(&n, r, &inc, r, &inc);
        rtr, rtr_old = np.dot(r, r), rtr
        # dscal(&n, &(rtr / rtr_old), p, &inc);
        p *= rtr / rtr_old
        # daxpy(&n, &one, r, &1, p, &1);
        p += r

    # end for

    return iteration


def tron(func, x, n_iterations=100, rtol=1e-3, atol=1e-5, args=(),
         verbose=False):
    """Trust region Newton optimization."""
    eta0, eta1, eta2 = 1e-4, 0.25, 0.75
    sigma1, sigma2, sigma3 = 0.25, 0.5, 4.0

    f_valp_, f_grad_, f_hess_ = func

    iteration, cg_iter = 0, 0

    fval = f_valp_(x, *args)
    grad = f_grad_(x, *args)
    grad_norm = np.linalg.norm(grad)

    # make a copy of `-grad` and zeros like `W0`
    # r, z = -grad, np.zeros_like(x)
    delta, grad_norm_tol = grad_norm, grad_norm * rtol + atol
    while iteration < n_iterations and grad_norm > grad_norm_tol:
        r, z = -grad, np.zeros_like(x)
        cg_iter += trcg(f_hess_, r, z, tr_delta=delta, args=args, rtol=1e-2)

        z_norm = np.linalg.norm(z)
        if iteration == 0:
            delta = min(delta, z_norm)

        # trcg finds x and r s.t. r + A z = -g and \|r\|\to \min
        # f(x) - f(x+z) ~ -0.5 * (2 g'z + z'Az) = -0.5 * (g'z + z'(-r))
        linear = np.dot(z, grad)
        approxred = -0.5 * (linear - np.dot(z, r))

        # The value and the actual reduction: compute the forward pass.
        fnew = f_valp_(x + z, *args)
        actualred = fval - fnew

        if linear + actualred < 0:
            alpha = max(sigma1, 0.5 * linear / (linear + actualred))
        else:
            alpha = sigma3
        # end if

        if actualred < eta0 * approxred:
            delta = min(max(alpha, sigma1) * z_norm, sigma2 * delta)
        elif actualred < eta1 * approxred:
            delta = max(sigma1 * delta, min(alpha * z_norm, sigma2 * delta))
        elif actualred < eta2 * approxred:
            delta = max(sigma1 * delta, min(alpha * z_norm, sigma3 * delta))
        else:
            delta = max(delta, min(alpha * z_norm, sigma3 * delta))
        # end if

        if verbose:
            print("""iter %2d act %5.3e pre %5.3e delta %5.3e f """
                  """%5.3e |g| %5.3e CG %3d""" %
                  (iteration, actualred, approxred,
                   delta, fval, grad_norm, cg_iter))

        if actualred > eta0 * approxred:
            x += z
            fval, grad = fnew, f_grad_(x, *args)
            grad_norm = np.linalg.norm(grad)
            iteration += 1

            # r, z = -grad, np.zeros_like(x)
        # end if

        if verbose:
            if fval < -1e32:
                print("WARNING: f < -1.0e+32")

            if abs(actualred) <= 0 and approxred <= 0:
                print("WARNING: actred and prered <= 0")

            if abs(actualred) <= 1e-12 * abs(fval) and \
               abs(approxred) <= 1e-12 * abs(fval):
                print("WARNING: actred and prered too small")

    return cg_iter
