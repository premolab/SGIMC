"""Various utility functions and classes to help with experimentation."""
import signal

import numpy as np
from math import sqrt
from scipy.sparse import coo_matrix
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics import mean_squared_error

from sgimc.utils import sparsify_with_mask


class DelayedKeyboardInterrupt(object):
    """An uninterruptible critical section.

    This critical section postpones the firing on the keyboard interrupt
    unitl after the its `with`-scope.
    """

    def __enter__(self):
        """Enter the critical section and hook the keyboard interrupts."""
        self.signal_received = False
        self.old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        """Handle the fired interrupt."""
        self.signal_received = (sig, frame)

    def __exit__(self, type, value, traceback):
        """Leave the scope of the critical section and service interruprts."""
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


def get_prediction(X, W_stack, H_stack, Y, binarize=False):
    """Calculates the resulting matrix given by the model R_hat = X W H' Y'."""
    
    w, h = W_stack[...,-1], H_stack[...,-1]
    r_hat = safe_sparse_dot(safe_sparse_dot(X, w), safe_sparse_dot(Y, h).T, dense_output=True)
    
    if binarize:
        r_hat = np.sign(r_hat)
    
    return r_hat


def combine_with_identity(X, return_sparse=True):
    """Concatenates X with the identity matrix on the right."""
    
    assert X.ndim == 2, 'Input matrix should have ndim = 2'
    
    X_add = np.eye(X.shape[0])
    X_comb = np.concatenate((X, X_add), axis=1)
    
    if return_sparse:
        X_comb = sparsify_with_mask(X_comb, X_comb != 0)
    
    return X_comb


def add_noise_features(X, n_features, scale, random_state, return_sparse=True):
    
    if n_features == 0:
        return X
    
    n = X.shape[0]
    X_noise = random_state.normal(scale=scale, size=(n, n_features))
    X_comb = np.concatenate((X.toarray(), X_noise), axis=1)
    
    if return_sparse:
        X_comb = sparsify_with_mask(X_comb, X_comb != 0)
        
    return X_comb


# =============================== Functions to calculate loss ===============================

def norm(r, mask=None):
    if mask is None: return np.sum(r**2)
    else: return np.sum(r[mask]**2)
    

def relative_loss(r, r_hat, mask=None):
    a = norm((r - r_hat), mask=mask)
    b = norm(r, mask=mask)
    return a/b


def calculate_loss(R, X, W_stack, H_stack, Y, mask=None):
    """Calculates loss specified in 'norm' fucntion between R and R_hat,
    given by the model R_hat = X W H' Y'. Mask provides element wise loss."""
    
    r_hat = get_prediction(X, W_stack, H_stack, Y)
    l = relative_loss(R, r_hat, mask=mask)
    
    return l


def rmse(r, r_hat, mask=None):
    """Calculates rmse between R and R_hat, elements can be specified by mask."""
    if mask is None:
        mse = mean_squared_error(r.ravel(), r_hat.ravel())
    else:
        mse = mean_squared_error(r[mask], r_hat[mask])
    return sqrt(mse)


def accuracy(r, r_hat, mask=None):
    """Calculates accuracy between R and R_hat, elements can be specified by mask."""
    if mask is None:
        a = r.ravel()
        b = r_hat.ravel()
        return len(a[a == b]) * 1. / len(a)
    else:
        a = r[mask]
        b = r_hat[mask]
        return len(a[a == b]) * 1. / len(a)
