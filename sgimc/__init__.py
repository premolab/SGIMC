"""Sparse group IMC."""
from math import sqrt

import numpy as np

import tqdm


def imc_descent(X, W, Y, H, R, C, step_fn, eta=1e-5,
                n_iterations=500, rtol=1e-5, atol=1e-8,
                verbose=False, return_history=False):

    iteration, history_W, history_H = 0, [W], [H]
    n_sq_dim_W, n_sq_dim_H = sqrt(np.prod(W.shape)), sqrt(np.prod(H.shape))
    with tqdm.tqdm(initial=iteration, total=n_iterations,
                   disable=not verbose) as pbar:
        while iteration < n_iterations:
            # Gauss-Siedel iteration
            W_old, W = W, step_fn(X, W, Y, H, R, C, eta,
                                  rtol=rtol, atol=atol)

            H_old, H = H, step_fn(Y, H, X, W, R.T, C, eta,
                                  rtol=rtol, atol=atol)

            pbar.update(1)
            iteration += 1

            if return_history:
                history_W.append(W)
                history_H.append(H)

            # stopping criterion: proximity to the previous value
            tol_W = n_sq_dim_W * atol + rtol * \
                np.linalg.norm(W_old.reshape(-1), 2)
            div_H = np.linalg.norm((H_old - H).reshape(-1), 2)

            tol_H = n_sq_dim_H * atol + rtol * \
                np.linalg.norm(H_old.reshape(-1), 2)
            div_W = np.linalg.norm((W_old - W).reshape(-1), 2)
            if div_W <= tol_W and div_H <= tol_H:
                break
            # end if
        # end while
    # end with

    if not return_history:
        return W, H

    # Stack along the 3rd dimension
    return np.stack(history_W, axis=-1), np.stack(history_H, axis=-1)
