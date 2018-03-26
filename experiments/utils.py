"""Various utility functions and classes to help with experimentation."""
import os
import time
import gzip
import pickle
import signal

import numpy as np
from sklearn.utils.extmath import safe_sparse_dot


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


def save(obj, path, filename=None, gz=None):
    """Pickle a pythonic `obj` into a file given by `path`.

    Parameters
    ----------
    obj: any python object
        An object to pickle.

    path: string
        A file in which to pickle the object.

    filename: string, optinal
        Specify filename for re-building experiments results. If None - will
        be saved as time.

    gz: integer, or None, optinal
        If None, then does not apply compression while pickling. Otherwise
        must be an integer 0-9 which determines the level of GZip compression:
        the lower the level the less thorough but the more faster the
        compression is. Value `0` produces a GZip archive with no compression
        whatsoever, whereas the value of `9` produces the most compressed
        archive.

    Returns
    -------
    filename: string
        The name of the resulting archive.
    """
    if not(gz is None or (isinstance(gz, int) and 0 <= gz <= 9)):
        raise TypeError("""`gz` parameter must be either `None` """
                        """or an integer 0-9.""")

    if not os.path.isdir(path):
        os.makedirs(path)

    open_ = open if gz is None else lambda f, m: gzip.open(f, m, gz)
    if filename is None:
        filename_ = "%s-%s.%s" % (path, time.strftime("%Y%m%d_%H%M%S"),
                                  "pic" if gz is None else "gz")
    else:
        filename_ = "%s%s%s" % (path, filename,
                                '.pic' if gz is None else '.gz')

    with open_(filename_, "wb+") as f:
        pickle.dump(obj, f)
    if filename is None:
        return filename_


def load(filename):
    """Recover an object from the file identified by `filename`.

    Parameters
    ----------
    filename: string
        A `file` in which an object is pickled.

    Returns
    -------
    object: a python object
        The recovered pythonic object.
    """
    open_ = open if not filename.endswith(".gz") else gzip.open

    with open_(filename, "rb") as f:
        obj = pickle.load(f)

    return obj


def get_prediction(X, W_stack, H_stack, Y, binarize=False):
    """Calculates the resulting matrix given by the model R_hat = X W H' Y'."""
    
    w, h = W_stack[...,-1], H_stack[...,-1]
    r_hat = safe_sparse_dot(safe_sparse_dot(X, w), safe_sparse_dot(Y, h).T, dense_output=True)
    
    if binarize:
        r_hat = np.sign(r_hat)
    
    return r_hat

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


def accuracy(r, r_hat, mask=None):
    if mask is None:
        a = r.ravel()
        b = r_hat.ravel()
        return len(a[a == b]) * 1. / len(a)
    else:
        a = r[mask]
        b = r_hat[mask]
        return len(a[a == b]) * 1. / len(a)


def invert(mask1):
    """Inverts the given boolean mask."""
    assert not mask1 is None
    mask1 = np.asarray(mask1, dtype='int8')
    mask2 = np.array(mask1 - 1, dtype='bool')
    return mask2