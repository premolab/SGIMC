"""Sparse group IMC."""
from math import sqrt

import numpy as np

import tqdm

from sklearn.utils.extmath import safe_sparse_dot


class IMCProblem(object):
    """A container for the IMC problem."""

    def __init__(self, objective, X, Y, R, n_threads=4):
        """A container for the IMC problem."""
        self._X, self._Y, self._R = X, Y, R.tocsr()
        self._objective = objective

        self.n_threads = n_threads

    def objective(self, W, H, approx_type="quadratic"):
        """Get the approximation of the objective for the current problem."""
        return self._objective(self._X, W, self._Y, H, self._R,
                               approx_type=approx_type,
                               n_threads=self.n_threads)

    def value(self, W, H):
        """Return the value of the current problem."""
        return self.objective(W, H, approx_type="const").value()

    def prediction(self, W, H):
        """Return the prediction for the problem."""
        return safe_sparse_dot(safe_sparse_dot(self._X, W),
                               safe_sparse_dot(self._Y, H).T,
                               dense_output=True)

    def score(self, predict, target):
        """Compute the score for this type of problem."""
        return self._objective.score(predict, target)

    def loss(self, predict, target):
        """Compute the loss assicoated with this problem."""
        return self._objective.v_func(predict, target)

    @property
    def T(self):
        """Create an instance of the transposed problem."""
        if not hasattr(self, "_transpose"):
            # create an instance of the transposed problem
            self._transpose = IMCProblem(self._objective,
                                         self._Y, self._X, self._R.T,
                                         self.n_threads)

            # back reference to self
            self._transpose._transpose = self

        return self._transpose


def imc_descent(problem, W, H, step_fn, step_kwargs={},
                n_iterations=500, return_history=False,
                rtol=1e-5, atol=1e-8, verbose=False,
                check_product=True):

    assert isinstance(problem, IMCProblem), \
        """`problem` must be an IMC problem."""

    b_stop, iteration, history_W, history_H = False, 0, [W], [H]
    n_sq_dim_W, n_sq_dim_H = sqrt(np.prod(W.shape)), sqrt(np.prod(H.shape))

    # patch 2017-12-13 -- stopping criterion on the product of W H
    M, n_sq_dim_M = None, 0
    if check_product:
        M = np.dot(W, H.T)
        n_sq_dim_M = sqrt(np.prod(M.shape))

    with tqdm.tqdm(initial=iteration, total=n_iterations,
                   disable=not verbose) as pbar:
        while (iteration < n_iterations) and (not b_stop):
            # Gauss-Siedel iteration
            W_old, W = W, step_fn(problem, W, H, **step_kwargs)

            H_old, H = H, step_fn(problem.T, H, W, **step_kwargs)

            pbar.update(1)
            iteration += 1

            if return_history:
                history_W.append(W)
                history_H.append(H)

            # stopping criterion: proximity to the previous value
            if not check_product:
                tol_H = n_sq_dim_H * atol + rtol * \
                    np.linalg.norm(H_old.reshape(-1), 2)
                div_H = np.linalg.norm((H_old - H).reshape(-1), 2)

                tol_W = n_sq_dim_W * atol + rtol * \
                    np.linalg.norm(W_old.reshape(-1), 2)
                div_W = np.linalg.norm((W_old - W).reshape(-1), 2)

                b_stop = (div_W <= tol_W) and (div_H <= tol_H)

            else:
                M_old, M = M, np.dot(W, H.T)

                tol_M = n_sq_dim_M * atol + rtol * \
                    np.linalg.norm(M_old.reshape(-1), 2)
                div_M = np.linalg.norm((M_old - M).reshape(-1), 2)

                b_stop = (div_M <= tol_M)
            # end if
        # end while
    # end with

    if not return_history:
        return W, H

    # Stack along the 3rd dimension
    return np.stack(history_W, axis=-1), np.stack(history_H, axis=-1)
