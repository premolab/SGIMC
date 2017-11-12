import numpy as np

from ..ops import op_s, op_d

class QuadraticApproximation(object):
    r"""The base class for quadratic approximation for the IMC objective.

    The IMC objective is
    .. math ::
        F(W, H) = \sum_{i,j\in \Omega}
    """
    def __init__(self, X, W, Y, H, R, hessian=True, n_threads=1):
        """Initialize the quadratic approximation w.r.t. W."""
        self.n_threads = n_threads
        self.hessian = hessian

        self.R, self.X = R.tocsr(), np.asfortranarray(X)
        self.YH = np.asfortranarray(np.dot(Y, H))

        self.update(W)

    def update(self, W):
        """Initialize the quadratic approximation."""
        self.p_val = op_d(self.R, self.X, self.YH,
                          np.asfortranarray(W), self.n_threads)

        self.v_val = self.v_func(self.p_val, self.R.data)  # / self.R.nnz

        self.g_val = self.g_func(self.p_val, self.R.data)  # / self.R.nnz

        if self.hessian:
            self.h_val = self.h_func(self.p_val, self.R.data)  # / self.R.nnz

        return self

    def value(self):
        """The value of the objective at the origin of the approximation."""
        return self.v_val.sum()

    def grad(self):
        """Return the gradient."""
        return op_s(self.R, self.X, self.YH,
                    self.g_val, self.n_threads).copy("C")

    def hess_v(self, D):
        """Get the hessian-vector product."""
        if not self.hessian:
            raise RuntimeError(
                """Hessian-vector requested with `hessian = False`.""")

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
