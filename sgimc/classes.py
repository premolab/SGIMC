from math import sqrt

import tqdm
import warnings

import numpy as np

from scipy.special import expit
from scipy.sparse import csr_matrix, isspmatrix

from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.exceptions import NotFittedError, DataConversionWarning
from sklearn.utils import check_array

from .qa_objective import QAObjectiveLogLoss, QAObjectiveL2Loss

from .algorithm import admm_step
from .algorithm.admm import sub_0_tron, sub_m


class IMCProblem(object):
    """A container for the IMC problem."""

    def __init__(self, objective, X, Y, R, sample_weight=None, n_threads=4):
        """A container for the IMC problem."""
        self._X, self._Y, self._R = X, Y, R.tocsr()
        self._objective = objective

        self.sample_weight = sample_weight

        self.n_threads = n_threads

    def objective(self, W, H, approx_type="quadratic"):
        """Get the approximation of the objective for the current problem."""
        return self._objective(
            X=self._X, W=W, Y=self._Y, H=H, R=self._R,
            approx_type=approx_type, sample_weight=self.sample_weight,
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
            sample_weight_T = None
            if self.sample_weight is not None:
                # sample weights are a matrix with the sparsity and shape of R
                C = csr_matrix((self.sample_weight, self._R.nonzero()),
                               shape=self._R.shape, dtype=self._R.dtype)

                # transpose, convert to physical CSR, and then copy
                sample_weight_T = C.T.tocsr().data.copy()

            # create an instance of the transposed problem
            self._transpose = IMCProblem(
                objective=self._objective, X=self._Y, Y=self._X, R=self._R.T,
                sample_weight=sample_weight_T, n_threads=self.n_threads)

            # back reference to self
            self._transpose._transpose = self

        return self._transpose


def step_qa_imc(problem, W, H, C, rtol=1e-5, atol=1e-8, verbose=False):
    Obj = problem.objective(W, H, approx_type="quadratic")
    return sub_0_tron(W, Obj, W0=W, C=C, eta=0., rtol=rtol, atol=atol,
                      verbose=False)


def mf_descent(problem, W, H, C_ridge, n_iterations=500, return_history=False,
               rtol=1e-5, atol=1e-8, verbose=False):

    assert isinstance(problem, IMCProblem), \
        """`problem` must be an IMC problem."""

    b_stop, iteration, history_W, history_H = False, 0, [W], [H]
    with tqdm.tqdm(initial=iteration, total=n_iterations,
                   disable=not verbose) as pbar:
        while (iteration < n_iterations):
            # Gauss-Siedel iteration
            W = step_qa_imc(problem, W, H, C_ridge, rtol=rtol, atol=atol,
                            verbose=False)
            H = step_qa_imc(problem.T, H, W, C_ridge, rtol=rtol, atol=atol,
                            verbose=False)

            if return_history:
                history_W.append(W)
                history_H.append(H)

            pbar.update(1)
            iteration += 1
        # end while
    # end with

    if not return_history:
        return W, H

    # Stack along the last (3rd) dimension
    return np.stack(history_W, axis=-1), np.stack(history_H, axis=-1)


def imc_descent(problem, W, H, step_fn, step_kwargs={},
                n_iterations=500, return_history=False,
                rtol=1e-5, atol=1e-8, verbose=False,
                n_init_iterations=25, check_product=True):
    warnings.warn("""`imc_descent` is deprecated. Please use """
                  """`SparseGroupIMCClassifier` or """
                  """`SparseGroupIMCRegressor` instead.""",
                  DeprecationWarning)

    assert isinstance(problem, IMCProblem), \
        """`problem` must be an IMC problem."""

    C_lasso, C_group, C_ridge = step_kwargs.get("C", (0., 0., 1.))
    W, H = mf_descent(problem, W, H, max(1e-1, C_ridge), rtol=5e-2, atol=1e-4,
                      n_iterations=n_init_iterations, return_history=False,
                      verbose=verbose)

    b_stop, iteration, history_W, history_H = False, 0, [W], [H]

    # stopping criterion on the product of W H
    iteration, M, n_sq_dim_M = 0, None, 0
    if check_product:
        M = np.dot(W, H.T)
        n_sq_dim_M = sqrt(np.prod(M.shape))

    n_sq_dim_W, n_sq_dim_H = sqrt(np.prod(W.shape)), sqrt(np.prod(H.shape))
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

    # Stack along the last (3rd) dimension
    return np.stack(history_W, axis=-1), np.stack(history_H, axis=-1)


class BaseSparseGroupIMC(object):
    def __init__(self,
                 C_lasso=0.1,
                 C_group=0.01,
                 C_ridge=1.0,
                 eta=1e0,
                 max_gs_iter=50,
                 max_admm_iter=2,
                 n_threads=1,
                 check_convergence=False,
                 random_state=None,
                 verbose=False):
        self.C_lasso = C_lasso
        self.C_group = C_group
        self.C_ridge = C_ridge
        self.eta = eta
        self.max_gs_iter = max_gs_iter
        self.max_admm_iter = max_admm_iter
        self.n_threads = n_threads
        self.check_convergence = check_convergence
        self.random_state = random_state
        self.verbose = verbose

    def _step(self, problem, W, H, C, rtol=1e-2, atol=1e-4):
        obj = problem.objective(W, H, approx_type="quadratic")
        if self.max_admm_iter > 2:
            return admm_step(obj, W, C, self.eta, sparse=True,
                             n_iterations=self.max_admm_iter,
                             method="tron", rtol=rtol, atol=atol)

        C_lasso, C_group, C_ridge = C
        WW, LL = W.copy(), np.zeros_like(W)

        # Since the default setting is 2 inner iterations,
        #  the admm_step loop was unrolled and simplified.
        ZZ = sub_m(WW + LL, C_lasso, C_group, 0., eta=self.eta)
        WW = sub_0_tron(ZZ - LL, obj, W0=WW, C=C_ridge, eta=self.eta,
                        rtol=5e-2, atol=1e-4, verbose=False)
        LL += WW - ZZ

        ZZ = sub_m(WW + LL, C_lasso, C_group, 0., eta=self.eta)
        WW = sub_0_tron(ZZ - LL, obj, W0=WW, C=C_ridge, eta=self.eta,
                        rtol=5e-2, atol=1e-4, verbose=False)
        LL += WW - ZZ

        return sub_m(WW + LL, C_lasso, C_group, 0., eta=self.eta)

    def _gauss_siedel(self, problem, W, H, rtol=1e-3, atol=1e-5):
        C_lasso = self.C_lasso
        if isinstance(self.C_lasso, float):
            C_lasso = self.C_lasso, self.C_lasso

        C_group = self.C_group
        if isinstance(self.C_group, float):
            C_group = self.C_group, self.C_group

        C_ridge = self.C_ridge
        if isinstance(self.C_ridge, float):
            C_ridge = self.C_ridge, self.C_ridge

        if not all(isinstance(C, tuple) and len(C) == 2
                   for C in [C_lasso, C_group, C_ridge]):
            raise TypeError("""Regularization coefficient must be either a """
                            """float or a pair of floats.""")

        if self.check_convergence:
            WHt = np.dot(W, H.T)

        C_const_W = C_lasso[0], C_group[0], C_ridge[0]
        C_const_H = C_lasso[1], C_group[1], C_ridge[1]
        for iteration in tqdm.trange(self.max_gs_iter,
                                     disable=not self.verbose):
            # Gauss-Siedel iteration
            W = self._step(problem, W, H, C_const_W)
            H = self._step(problem.T, H, W, C_const_H)

            if self.check_convergence:
                # stopping criterion: proximity to the previous value
                WHt_old, WHt = WHt, np.dot(W, H.T)

                tol_M = rtol * np.linalg.norm(WHt_old, ord="fro")
                div_M = np.linalg.norm(WHt_old - WHt, ord="fro")
                if div_M <= sqrt(np.prod(WHt_old.shape)) * atol + tol_M:
                    break
            # end if
        # end for

        self.coef_W_, self.coef_H_ = W, H

        return self

    def _check_fit_inputs(self, X, Y, R, sample_weight):
        R = check_array(R, accept_sparse="csr", dtype=np.float64,
                        warn_on_dtype=True, ensure_2d=True)

        if not isinstance(R, csr_matrix):
            warnings.warn("""`R` matrix of type `%s` was converted to CSR."""
                          % type(R), DataConversionWarning)
            R = csr_matrix(R)

        X = check_array(X, accept_sparse="csr", dtype=np.float64,
                        warn_on_dtype=True, ensure_2d=True)

        Y = check_array(Y, accept_sparse="csr", dtype=np.float64,
                        warn_on_dtype=True, ensure_2d=True)

        if R.shape != (X.shape[0], Y.shape[0]):
            raise TypeError("""Number of rows in `X` and `Y` does not """
                            """match the demsions of `R`.""")

        if sample_weight is not None:
            if isspmatrix(sample_weight):
                sample_weight = sample_weight.tocsr().data

            sample_weight = np.ravel(sample_weight).copy()
            if len(sample_weight) != R.nnz:
                raise TypeError("""`sample_weight` must have as many """
                                """ elements as are nonzero in R.""")
        # end if

        return X, Y, R, sample_weight

    def _check_coef(self, X, W, name):
        W = W if W is not None else getattr(self, "coef_%s_" % name, None)

        if W is None:
            raise NotFittedError(
                """IMC not fitted: either run `fit()` or provide `%s`."""
                % name)

        if not isinstance(W, np.ndarray):
            raise TypeError(
                """`%s` must be an array.""" % name)

        if W.ndim != 2:
            raise TypeError(
                """`%s` must be a matrix.""" % name)

        if W.shape[0] != X.shape[1]:
            raise TypeError(
                """`%s` must conform to it features.""" % name)

        return W

    def _check_predict_inputs(self, X, Y, W, H):
        X = check_array(X, accept_sparse="csr", dtype=np.float64,
                        warn_on_dtype=True, ensure_2d=True)
        W = self._check_coef(X, W, "W")

        Y = check_array(Y, accept_sparse="csr", dtype=np.float64,
                        warn_on_dtype=True, ensure_2d=True)
        H = self._check_coef(Y, H, "H")

        return X, Y, W, H

    def fit(self, X, Y, R, sample_weight=None, W=None, H=None):
        """The `fit` method."""
        raise NotImplementedError("""Derived classes must implement this.""")

    def predict(self, X, Y, W=None, H=None):
        """The `predict` method."""
        raise NotImplementedError("""Derived classes must implement this.""")


class SparseGroupIMCClassifier(BaseSparseGroupIMC):
    """Sparse-Group Inductive Matrix Completion for binary classification.

    Parameters
    ----------
    rank : int
        Positive value determining the rank of the coefficient matrix in the
        bilinear form.

    C_lasso : float, tuple (default=0.1)
        Nonnegative regularization coefficient for elementwise sparsity. If
        a pair of floats is passed, then the values correspond to different
        regularization coefficients for the factors W and H respectively.

    C_group : float, tuple (default=0.01)
        Nonnegative regularization coefficient for row-wise group sparsity. If
        a pair of floats is passed, then the values correspond to different
        regularization coefficients for the factors W and H respectively.

    C_ridge : float, tuple (default=1.0)
        Nonnegative ridge-like regularization coefficient. If
        a pair of floats is passed, then the values correspond to different
        regularization coefficients for the factors W and H respectively.

    eta : float (default=1.0)
        The ADMM regularization coefficient. Recommended setting is `1.0`.

    max_gs_iter : int (default=100)
        The number of outer iterations.

    max_admm_iter : int (default=2)
        The number of inner iterations.

    n_threads : int (default=-1)
        The number of threads to utilize for compute-intensive operations.
        Positive numbers indicate the number of threads, negative numbers
        indicate determine the number of free cores, not burdened by
        computation.

    check_convergence : bool (default=False)
        Whether to check convergence of the coefficient in hte outer loop, or
        run unitl the number of iterations reaches `max_gs_iter`.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : bool, optional (default=False)
        Controls the verbosity of the outer loop.

    Attributes
    ----------
    W : array-like, shape (d_1, k)
        Estimated coefficient matrix of the row side-features.

    H : array-like, shape (d_2, k)
        Estimated coefficient matrix of the column side-features.
    """
    def __init__(self,
                 rank,
                 C_lasso=0.1,
                 C_group=0.01,
                 C_ridge=1.0,
                 eta=1e-1,
                 max_gs_iter=100,
                 max_admm_iter=2,
                 n_threads=-1,
                 check_convergence=False,
                 random_state=None,
                 verbose=False):
        super(SparseGroupIMCClassifier, self).__init__(
            C_lasso=C_lasso, C_group=C_group, C_ridge=C_ridge,
            eta=eta, max_gs_iter=max_gs_iter, max_admm_iter=max_admm_iter,
            n_threads=n_threads, check_convergence=check_convergence,
            random_state=random_state, verbose=verbose)

        self.rank = rank

    def fit(self, X, Y, R, sample_weight=None, W=None, H=None):
        """Fit the sparse group IMC classifier to the training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_1, d_1)
            Training row side-features, where n_1 is the number of rows and
            d_1 is the number of features per row.

        Y : {array-like, sparse matrix}, shape (n_2, d_2)
            Training column side-features, where n_2 is the number of columns
            and d_2 is the number of side-features per column.

        R : CSR sparse matrix, shape (n_1, n_2)
            Training `-1 / +1` binary labels, where n_1 is the number of
            rows and n_2 is the number of columns.

        sample_weight : array-like, shape (R.nnz,)
            Weight per each non-missing value in R. Higher weights force the
            classifier to put more emphasis on these entries.

        Returns
        -------
        self : object
        """
        # create an instance of an IMC problem
        X, Y, R, sample_weight = self._check_fit_inputs(X, Y, R, sample_weight)
        problem = IMCProblem(QAObjectiveLogLoss, X, Y, R,
                             sample_weight=sample_weight,
                             n_threads=self.n_threads)

        # draw random matrices
        random_state = check_random_state(self.random_state)
        if W is None:
            W = random_state.normal(size=(X.shape[1], self.rank))

        if H is None:
            H = random_state.normal(size=(Y.shape[1], self.rank))

        return self._gauss_siedel(problem, W, H)

    def predict_proba(self, X, Y, W=None, H=None):
        """Compute probability of the `+1` class.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_1, d_1)
            Row side-features, where n_1 is the number of rows and d_1 is
            the number of features per row.

        Y : {array-like, sparse matrix}, shape (n_2, d_2)
            Column side-features, where n_2 is the number of columns and d_2
            is the number of side-features per column.

        W : optional array-like, shape (d_1, k)
            Coefficient matrix of the row side-features to use for prediction.
            The estimated matrix `coef_W_` is used if `W` is not provided.

        H : optional array-like, shape (d_2, k)
            Coefficient matrix of the column side-features to use for the
            prediction. The estimated matrix `coef_H_` is used if `H` is
            not provided.

        Returns
        -------
        proba : array-like, shape (n_1, n_2)
            Returns the probability of `+1` class for each entry.
        """

        X, Y, W, H = self._check_predict_inputs(X, Y, W, H)
        logit = safe_sparse_dot(safe_sparse_dot(X, W),
                                safe_sparse_dot(Y, H).T)
        return expit(logit)

    def predict(self, X, Y, W=None, H=None):
        """Predict the final `-/+ 1` class label.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_1, d_1)
            Row side-features, where n_1 is the number of rows and d_1 is
            the number of features per row.

        Y : {array-like, sparse matrix}, shape (n_2, d_2)
            Column side-features, where n_2 is the number of columns and d_2
            is the number of side-features per column.

        W : optional array-like, shape (d_1, k)
            Coefficient matrix of the row side-features to use for prediction.
            The estimated matrix `coef_W_` is used if `W` is not provided.

        H : optional array-like, shape (d_2, k)
            Coefficient matrix of the column side-features to use for the
            prediction. The estimated matrix `coef_H_` is used if `H` is
            not provided.

        Returns
        -------
        label : array-like, shape (n_1, n_2)
            Returns the `-1/+1` class label for each entry.
        """

        X, Y, W, H = self._check_predict_inputs(X, Y, W, H)
        return np.sign(safe_sparse_dot(safe_sparse_dot(X, W),
                                       safe_sparse_dot(Y, H).T))


class SparseGroupIMCRegressor(BaseSparseGroupIMC):
    """Sparse-Group Inductive Matrix Completion for Regression.

    Parameters
    ----------
    rank : int
        Positive value determining the rank of the coefficient matrix in the
        bilinear form.

    C_lasso : float, tuple (default=0.1)
        Nonnegative regularization coefficient for elementwise sparsity. If
        a pair of floats is passed, then the values correspond to different
        regularization coefficients for the factors W and H respectively.

    C_group : float, tuple (default=0.01)
        Nonnegative regularization coefficient for row-wise group sparsity. If
        a pair of floats is passed, then the values correspond to different
        regularization coefficients for the factors W and H respectively.

    C_ridge : float, tuple (default=1.0)
        Nonnegative ridge-like regularization coefficient. If
        a pair of floats is passed, then the values correspond to different
        regularization coefficients for the factors W and H respectively.

    eta : float (default=1.0)
        The ADMM regularization coefficient. Recommended setting is `1.0`.

    max_gs_iter : int (default=100)
        The number of outer iterations.

    max_admm_iter : int (default=2)
        The number of inner iterations.

    n_threads : int (default=-1)
        The number of threads to utilize for compute-intensive operations.
        Positive numbers indicate the number of threads, negative numbers
        indicate determine the number of free cores, not burdened by
        computation.

    check_convergence : bool (default=False)
        Whether to check convergence of the coefficient in hte outer loop, or
        run unitl the number of iterations reaches `max_gs_iter`.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : bool, optional (default=False)
        Controls the verbosity of the outer loop.

    Attributes
    ----------
    W : array-like, shape (d_1, k)
        Estimated coefficient matrix of the row side-features.

    H : array-like, shape (d_2, k)
        Estimated coefficient matrix of the column side-features.
    """
    def __init__(self,
                 rank,
                 C_lasso=0.1,
                 C_group=0.01,
                 C_ridge=1.0,
                 eta=1e1,
                 max_gs_iter=100,
                 max_admm_iter=2,
                 n_threads=1,
                 check_convergence=False,
                 random_state=None,
                 verbose=False):
        super(SparseGroupIMCRegressor, self).__init__(
            C_lasso=C_lasso, C_group=C_group, C_ridge=C_ridge,
            eta=eta, max_gs_iter=max_gs_iter, max_admm_iter=max_admm_iter,
            n_threads=n_threads, check_convergence=check_convergence,
            random_state=random_state, verbose=verbose)

        self.rank = rank

    def fit(self, X, Y, R, sample_weight=None, W=None, H=None):
        """Fit the sparse group IMC regressor to the training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_1, d_1)
            Training row side-features, where n_1 is the number of rows and
            d_1 is the number of features per row.

        Y : {array-like, sparse matrix}, shape (n_2, d_2)
            Training column side-features, where n_2 is the number of columns
            and d_2 is the number of side-features per column.

        R : CSR sparse matrix, shape (n_1, n_2)
            Training `-1 / +1` binary labels, where n_1 is the number of
            rows and n_2 is the number of columns.

        sample_weight : array-like, shape (R.nnz,)
            Weight per each non-missing value in R. Higher weights force the
            classifier to put more emphasis on these entries.

        Returns
        -------
        self : object
        """
        # create an instance of an IMC problem
        X, Y, R, sample_weight = self._check_fit_inputs(X, Y, R, sample_weight)
        problem = IMCProblem(QAObjectiveL2Loss, X, Y, R,
                             sample_weight=sample_weight,
                             n_threads=self.n_threads)

        # draw random matrices
        random_state = check_random_state(self.random_state)
        if W is None:
            W = random_state.normal(size=(X.shape[1], self.rank))

        if H is None:
            H = random_state.normal(size=(Y.shape[1], self.rank))

        return self._gauss_siedel(problem, W, H)

    def predict(self, X, Y, W=None, H=None):
        """Compute the IMC regression prediction.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_1, d_1)
            Row side-features, where n_1 is the number of rows and d_1 is
            the number of features per row.

        Y : {array-like, sparse matrix}, shape (n_2, d_2)
            Column side-features, where n_2 is the number of columns and d_2
            is the number of side-features per column.

        W : optional array-like, shape (d_1, k)
            Coefficient matrix of the row side-features to use for prediction.
            The estimated matrix `coef_W_` is used if `W` is not provided.

        H : optional array-like, shape (d_2, k)
            Coefficient matrix of the column side-features to use for the
            prediction. The estimated matrix `coef_H_` is used if `H` is
            not provided.

        Returns
        -------
        out : array-like, shape (n_1, n_2)
            Returns the regression prediction.
        """

        X, Y, W, H = self._check_predict_inputs(X, Y, W, H)
        return safe_sparse_dot(safe_sparse_dot(X, W),
                               safe_sparse_dot(Y, H).T)
