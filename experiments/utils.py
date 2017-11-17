import os
import time
import gzip
import pickle


def save(obj, path, filename=None, gz=None):
    """Pickle a pythonic `obj` into a file given by `path`.

    Parameters
    ----------
    obj: any python object
        An object to pickle.
    path: string
        A file in which to pickle the object.
    filename: string
        Specify filename for re-building experiments results. If None - will
        be saved as time.
    gz: integer, or None
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
        filename_ = path + filename \
        + '{}'.format('.pic' if gz is None else '.gz')

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
