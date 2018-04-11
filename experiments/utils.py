"""Various utility functions and classes to help with experimentation."""
import os
import time
import gzip
import pickle
import signal

import numpy as np
from math import sqrt

from scipy.sparse import coo_matrix

from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as TTS

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


def combine_with_identity(X, return_sparse=True):
    """Concatenates X with the identity matrix on the right."""
    
    assert X.ndim == 2, 'Input matrix should have ndim = 2'
    
    X_add = np.eye(X.shape[0])
    X_comb = np.concatenate((X, X_add), axis=1)
    
    if return_sparse:
        X_comb = sparsify_with_mask(X_comb, X_comb > 0)
    
    return X_comb


def from_interactions_to_coo(interactions, user_ids_begins_from_1=True, item_ids_begins_from_1=True,
                             shape=None):
    # task-hardcoded function
    if user_ids_begins_from_1:
        user_ids = interactions[0] - 1
    else:
        user_ids = interactions[0]
        
    if item_ids_begins_from_1:
        item_ids = interactions[1] - 1
    else:
        item_ids = interactions[1]
        
    ratings = interactions[2]
    
    if shape is None:
        shape = (np.max(user_ids)+1, np.max(item_ids)+1)
    R_coo = coo_matrix((ratings, (user_ids, item_ids)), shape=shape, dtype='float64')
    
    return R_coo


def sample_from_interactions(interactions, train_size, seed=42):
    """Randomly (with specified seed) sample from interactions."""
    train_t, test_t = TTS(interactions.T, train_size=train_size, random_state=seed, shuffle=False)
    train, test = train_t.T, test_t.T
    return train, test
    
    
def divide_train_test(I, shape, train_size, seed=42):
    # TODO: add q1 and q2 instead train_size (q)s.
    # TODO: make it work with arbitrary user and item ids.
    n, m = shape
    user_ids = np.array([i for i in range(n)])
    item_ids = np.array([i for i in range(m)])
    
    train_user_ids, test_user_ids = TTS(user_ids, train_size=train_size, random_state=seed, shuffle=False)
    train_item_ids, test_item_ids = TTS(item_ids, train_size=train_size, random_state=seed, shuffle=False)
    
    oo, on, no, nn = [], [], [], []
    for elem in I.T:
        if elem[0] in train_user_ids and elem[1] in train_item_ids:
            oo.append(elem)
        elif elem[0] in train_user_ids and elem[1] in test_item_ids:
            on.append(elem)
        elif elem[0] in test_user_ids and elem[1] in train_item_ids:
            no.append(elem)
        else:
            nn.append(elem)
    
    return np.array(oo).T, np.array(on).T, np.array(no).T, np.array(nn).T


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


def invert(mask1):
    """Inverts the given boolean mask."""
    assert not mask1 is None
    mask1 = np.asarray(mask1, dtype='int8')
    mask2 = np.array(mask1 - 1, dtype='bool')
    return mask2