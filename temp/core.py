import sys
import signal
import numpy as np

from utils import sigmoid

# build lib to import!
from imc_sparse_ops.imc_ops import op_s, op_d

class DelayedKeyboardInterrupt(object):
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)
            
class QAObjective(object):
    def __init__(self, X, W, Y, H, R, hessian=True, n_threads=1):
        self.n_threads = n_threads
        self.hessian = hessian

        self.R, self.X = R.tocsr(), np.asfortranarray(X)
        self.YH = np.asfortranarray(np.dot(Y, H))

        self.update(W)

    def update(self, W):
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

    @staticmethod
    def score(predict, target):
        raise NotImplementedError(
            """Derived classes must implement this.""")
        
        
class QAObjectiveL2Loss(QAObjective):
    def __init__(self, X, W, Y, H, R, hessian=True, n_threads=8):
        super(QAObjectiveL2Loss, self).__init__(
            X, W, Y, H, R, hessian=hessian, n_threads=n_threads)

    @staticmethod
    def v_func(predict, target):
        return 0.5 * (predict - target)**2

    @staticmethod
    def g_func(predict, target):
        return predict - target

    @staticmethod
    def h_func(predict, target):
        return np.ones(len(target), dtype="double")

    @staticmethod
    def score(predict, target):
        return np.sqrt(((predict - target)**2).mean())
    
    
class QAObjectiveLogLoss(QAObjective):
    def __init__(self, X, W, Y, H, R, hessian=True, n_threads=8):
        super(QAObjectiveLogLoss, self).__init__(
            X, W, Y, H, R, hessian=hessian, n_threads=n_threads)

    @staticmethod
    def v_func(logit, target):
        out = np.log1p(np.exp(- abs(logit)))
        out -= np.minimum(target * logit, 0)
        return out

    @staticmethod
    def g_func(logit, target):
        return sigmoid(logit * target) * target - target

    @staticmethod
    def h_func(logit, target):
        prob = sigmoid(logit)
        return prob * (1 - prob)

    @staticmethod
    def score(logit, target):
        predict = (2. * (logit > 0)) - 1
        return (predict != target).mean()


class QAObjectiveHuberLoss(QAObjectiveL2Loss):
    # The Huber-loss threshold is a global constant for now.
    EPSILON = 1.0e0
    
    def __init__(self, X, W, Y, H, R, hessian=True, epsilon=EPSILON, n_threads=4):
        super(QAObjectiveHuberLoss, self).__init__(
            X, W, Y, H, R, hessian=hessian, n_threads=n_threads)

        self.epsilon = epsilon

    @staticmethod
    def v_func(predict, target):
#     def v_func(self, predict, target):
#         eps = self.epsilon
        eps = EPSILON

        resid = abs(predict - target)
        return np.where(resid > eps,
                        eps * (resid - eps / 2),
                        0.5 * resid**2)

    @staticmethod
    def g_func(predict, target):
#     def g_func(self, predict, target):
#         eps = self.epsilon
        eps = EPSILON

        resid = predict - target
        return np.where(abs(resid) > eps,
                        eps * np.sign(resid),
                        resid)

    @staticmethod
    def h_func(predict, target):
#     def h_func(self, predict, target):
#         eps = self.epsilon
        eps = EPSILON

        return (abs(predict - target) <= eps) * 1.