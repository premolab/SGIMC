import numpy as np

from ..ops import op_s, op_d

from scipy.sparse import csr_matrix, isspmatrix

from sklearn.utils.extmath import safe_sparse_dot


def op_s_ref(R, X, Z, S, n_threads=None):
    U = csr_matrix(R, copy=False)
    U.data = S
    return safe_sparse_dot(safe_sparse_dot(X.T, U), Z)


def op_d_ref(R, X, Z, D, n_threads=None):
    # ii, jj = R.nonzero()
    # return np.multiply(safe_sparse_dot(X[ii, :], D), Z[jj, :]).sum(axis=1)
    return safe_sparse_dot(safe_sparse_dot(X, D), Z.T)[R.nonzero()]


class QuadraticApproximation(object):
    r"""The base class for quadratic approximation for the IMC objective.

    The IMC objective is
    .. math ::
        F(W, H) = \sum_{i,j\in \Omega}
    """

    def __init__(self, X, W, Y, H, R, sample_weight=None,
                 n_threads=1, approx_type="quadratic"):
        """Initialize the approximation of an IMC objective w.r.t. W."""

        assert isspmatrix(R), """`R` must be a sparse matrix."""

        self.n_threads, self.approx_type = n_threads, approx_type.lower()
        assert self.approx_type in ("const", "linear", "quadratic")

        if sample_weight is not None:
            if isspmatrix(sample_weight):
                sample_weight = sample_weight.tocsr().data

            sample_weight = np.ravel(sample_weight).copy()
            assert len(sample_weight) == R.nnz, \
                """`sample_weight` must have as many elements """ \
                """as are nonzero in R."""

        self.sample_weight = sample_weight

        self.X = csr_matrix(X) if isspmatrix(X) else np.ascontiguousarray(X)
        self.YH = np.ascontiguousarray(
            safe_sparse_dot(Y, H, dense_output=True))

        self.R = R.tocsr()

        self.update(W)

    def update(self, W):
        """Update the approximation w.r.t. W."""
        return self.forward(W).backward()

    def forward(self, W):
        """Forward-update the approximation w.r.t. W."""
        self.p_val = op_d(self.R, self.X, self.YH,
                          np.ascontiguousarray(W), self.n_threads)

        self.v_val = self.v_func(self.p_val, self.R.data)
        if self.sample_weight is not None:
            self.v_val *= self.sample_weight

        return self

    def backward(self):
        """Precompute gradient and hessian statistics."""
        if self.approx_type in ("linear", "quadratic"):
            self.g_val = self.g_func(self.p_val, self.R.data)
            if self.sample_weight is not None:
                self.g_val *= self.sample_weight

            if self.approx_type == "quadratic":
                self.h_val = self.h_func(self.p_val, self.R.data)
                if self.sample_weight is not None:
                    self.h_val *= self.sample_weight

        return self

    def value(self):
        """The current value of the objective."""
        return self.v_val.sum()

    def grad(self, out=None):
        """Return the gradient."""
        if self.approx_type not in ("linear", "quadratic"):
            raise RuntimeError(
                """Gradient requested with `approx_type = "%s"`.""" % (
                    self.approx_type,))

        return op_s(self.R, self.X, self.YH, self.g_val, self.n_threads)

    def hess_v(self, D):
        """Get the hessian-vector product."""
        if self.approx_type != "quadratic":
            raise RuntimeError(
                """Hessian-vector requested with `approx_type = "%s"`.""" % (
                    self.approx_type,))

        c_val = op_d(self.R, self.X, self.YH,
                     np.ascontiguousarray(D), self.n_threads)
        c_val *= self.h_val

        return op_s(self.R, self.X, self.YH, c_val, self.n_threads)

    @staticmethod
    def v_func(predict, target):
        raise NotImplementedError(
            """Derived classes must implement this.""")

    @staticmethod
    def g_func(predict, target):
        raise NotImplementedError(
            """Derived classes must implement this.""")

    @staticmethod
    def h_func(predict, target):
        raise NotImplementedError(
            """Derived classes must implement this.""")
