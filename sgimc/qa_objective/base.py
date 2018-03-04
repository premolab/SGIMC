import numpy as np

from ..ops import op_s, op_d

from scipy.sparse import csr_matrix, isspmatrix


class QuadraticApproximation(object):
    r"""The base class for quadratic approximation for the IMC objective.

    The IMC objective is
    .. math ::
        F(W, H) = \sum_{i,j\in \Omega}
    """

    def __init__(self, X, W, Y, H, R, n_threads=1, approx_type="quadratic"):
        """Initialize the approximation of an IMC objective w.r.t. W."""

        assert isinstance(R, csr_matrix), """`R` must be a CSR matrix."""

        self.n_threads, self.approx_type = n_threads, approx_type.lower()
        assert self.approx_type in ("const", "linear", "quadratic")

        self.X = csr_matrix(X) if isspmatrix(X) else np.asfortranarray(X)
        self.R, self.YH = R.tocsr(), np.asfortranarray(np.dot(Y, H))

        self.update(W)

    def update(self, W):
        """Update the approximation w.r.t. W."""
        self.p_val = op_d(self.R, self.X, self.YH,
                          np.asfortranarray(W), self.n_threads)

        self.v_val = self.v_func(self.p_val, self.R.data)

        if self.approx_type in ("linear", "quadratic"):
            self.g_val = self.g_func(self.p_val, self.R.data)

            if self.approx_type == "quadratic":
                self.h_val = self.h_func(self.p_val, self.R.data)

        return self

    def value(self):
        """The value of the objective at the origin of the approximation."""
        return self.v_val.sum()

    def grad(self):
        """Return the gradient."""
        if self.approx_type not in ("linear", "quadratic"):
            raise RuntimeError(
                """Gradient requested with `approx_type = "%s"`.""" % (
                    self.approx_type,))

        return op_s(self.R, self.X, self.YH,
                    self.g_val, self.n_threads).copy("C")

    def hess_v(self, D):
        """Get the hessian-vector product."""
        if self.approx_type != "quadratic":
            raise RuntimeError(
                """Hessian-vector requested with `approx_type = "%s"`.""" % (
                    self.approx_type,))

        c_val = op_d(self.R, self.X, self.YH,
                     np.asfortranarray(D), self.n_threads)
        c_val *= self.h_val

        return op_s(self.R, self.X, self.YH,
                    c_val, self.n_threads).copy("C")

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
