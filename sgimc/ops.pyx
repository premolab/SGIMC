# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

# Author: Ivan Nazarov <ivannnnz@gmail.com>
import numpy as np
cimport numpy as np
cimport cython

from cpython cimport bool

from scipy.sparse import csr_matrix

from libc.math cimport fabs, sqrt
from libc.string cimport memcpy

from scipy.linalg.cython_blas cimport dnrm2

ctypedef fused real:
    cython.floating


np.import_array()


cdef extern from "src/threads.c":
    int __has_omp "has_omp" () nogil


def has_omp():
    """Check if compiled with OMP support."""
    return __has_omp() > 0


def op_d(object R, object X, double[:, ::1] Z, double[:, ::1] D, int n_threads=1):
    """Compute the `op_d` operation returning data for a sparse matrix.

    The function maps a dense `D` to a flat array of data for a CSR sparse
    matrix `X D Z'`, with structure identical to `R`.

    Parameters
    ----------
    R : CSR sparse matrix, shape = [n_1, n_2]
        Determines the sprasity structure to compute the results for.

    X : CSR sparse or dense matrix, shape = [n_1, d_1], order="C"
        The left matrix in the result.

    Z : dense matrix, shape = [n_2, k], order="C"
        The right matrix in the result.

    D : dense matrix, shape = [d_1, k], order="C"
        The matrix to compute the result for.

    n_threads : int, default 1
        The number of threads to use for computing the operation.

    Returns
    -------
    S : 1-D array
        The flat data for the resulting sparse matrix with structure given
        in CSR matrix R.
    """
    # take a dense D and return an sparse X D Z'

    if not isinstance(R, csr_matrix):
        raise TypeError("""`R` must be CSR matrix.""")

    cdef double[::1] S = np.zeros(R.nnz, dtype="double")
    cdef int errcode = -1

    if not has_omp():
        n_threads = 1

    if not isinstance(X, csr_matrix):
        errcode = _dense_op_d(R, X, Z, D, S, n_threads)
    else:
        errcode = _sparse_op_d(R, X, Z, D, S, n_threads)
    # endif

    if errcode != 0:
        raise MemoryError

    return S.base


def op_s(object R, object X, double[:, ::1] Z, double[::1] S, int n_threads=1):
    """Compute the `op_s` operation returning dense matrix.

    The function takes a flat array of data for a CSR sparse matrix `S` with
    structure determined by `R` and returns a dense matrix `X' S Z`.

    Parameters
    ----------
    R : CSR sparse matrix, shape = [n_1, n_2]
        Determines the sprasity structure to compute the results for.

    X : CSR sparse or dense matrix, shape = [n_1, d_1], order="C"
        The left matrix in the result.

    Z : dense matrix, shape = [n_2, k], order="C"
        The right matrix in the result.

    S : 1-D array
        The flat data of the sparse matrix with structure given
        in CSR matrix R.

    n_threads : int, default 1
        The number of threads to use for computing the operation.

    Returns
    -------
    D : dense matrix, shape = [d_1, k], order="C"
        The resulting dense matrix.
    """
    if not isinstance(R, csr_matrix):
        raise TypeError("""`R` must be CSR matrix.""")

    cdef int d_1 = X.shape[1], k = Z.shape[1]
    cdef double[:, ::1] D = np.zeros((d_1, k), dtype="double", order="C")
    cdef int errcode = -1

    if not has_omp():
        n_threads = 1

    if not isinstance(X, csr_matrix):
        errcode = _dense_op_s(R, X, Z, S, D, n_threads)
    else:
        errcode = _sparse_op_s(R, X, Z, S, D, n_threads)
    # endif

    if errcode != 0:
        raise MemoryError

    return D.base


def diag_op_d(object R, double[::1, :] X, double[::1, :] Z, double[::1] D):
    """Compute the `op_d` operation for a diagonal matrix `D`.

    Unlike `op_d` this function takes the main diagonal of a square matrix,
    and computes a flat array of data for a CSR sparse matrix `X D Z'` with
    structure determined by `R`.

    Parameters
    ----------
    R : CSR sparse matrix, shape = [n_1, n_2]
        Determines the sprasity structure to compute the results for.

    X : dense matrix, shape = [n_1, k], order="F"
        The left matrix in the result.

    Z : dense matrix, shape = [n_2, k], order="F"
        The right matrix in the result.

    D : 1-D array
        The main diagonal of the suare matrix.

    Returns
    -------
    S : 1-D array
        The flat data for the resulting sparse matrix with structure given
        in CSR matrix R.
    """
    if not isinstance(R, csr_matrix):
        raise TypeError("""`R` must be CSR matrix.""")

    if X.shape[1] != Z.shape[1]:
        raise TypeError("""`X` and `Z` have the same number of columns.""")

    # shapes
    cdef int n_1 = X.shape[0], n_2 = Z.shape[0], k = X.shape[1]

    # sparsity
    cdef int[::1] Sp = R.indptr, Sj = R.indices

    # output
    cdef double[::1] S = np.zeros(R.nnz, dtype="double")

    cdef int errcode = -1

    errcode = __diag_op_d(&Sp[0], &Sj[0], n_1, n_2, k,
                          &X[0, 0], &Z[0, 0], &D[0], &S[0])

    if errcode != 0:
        raise MemoryError

    return S.base


def diag_op_s(object R, double[::1, :] X, double[::1, :] Z, double[::1] S):
    """Compute the `op_s` operation returning the main diagonal of the matrix.

    Like `op_s` this function takes a flat array of data for a CSR sparse
    matrix `S` with structure determined by `R`, but unlike the former
    `diag_op_s` computes only the main digonal of the dense matrix `X' S Z`.
    The result is returned in a flat array, representing the values on 
    the main diagonal.

    Parameters
    ----------
    R : CSR sparse matrix, shape = [n_1, n_2]
        Determines the sprasity structure to compute the results for.

    X : dense matrix, shape = [n_1, k], order="F"
        The left matrix in the result.

    Z : dense matrix, shape = [n_2, k], order="F"
        The right matrix in the result.

    S : 1-D array
        The flat data of the sparse matrix with structure given
        in CSR matrix R.

    Returns
    -------
    D : 1-D array, shape = [k,]
        The computed main diagonal.
    """
    if not isinstance(R, csr_matrix):
        raise TypeError("""`R` must be CSR matrix.""")

    if X.shape[1] != Z.shape[1]:
        raise TypeError("""`X` and `Z` have the same number of columns.""")

    # shapes
    cdef int n_1 = X.shape[0], n_2 = Z.shape[0], k = X.shape[1]

    # sparsity
    cdef int[::1] Sp = R.indptr, Sj = R.indices

    # output
    cdef double[::1] D = np.zeros(k, dtype="double")

    cdef int errcode = -1

    errcode = __diag_op_s(&Sp[0], &Sj[0], n_1, n_2, k,
                          &X[0, 0], &Z[0, 0], &S[0], &D[0])

    if errcode != 0:
        raise MemoryError

    return D.base


cdef inline real soft_threshold(real x, real C) nogil:
    """Compute the soft_thresholding operator."""
    cdef real out = 0.
    if fabs(x) <= C:
        out = 0.
    elif x > 0:
        out = x - C
    else:
        out = x + C
    # end if
    return out


cdef inline real norm2(int n, real *x, int inc) nogil:
    """Compute the L2-norm of a vector."""
    cdef double scale, absxi, norm, ssq

    cdef int i
    
    if n < 1 or inc < 1:
        norm = 0.0
    elif n == 1:
        norm = fabs(x[0])
    else:
        scale, ssq, i = 1.0, 0.0, 0
        while i < n * inc:
            if x[i] != 0:
                absxi = fabs(x[i])
                if scale > absxi:
                    ssq = 1.0 + ssq * (scale / absxi)**2
                    scale = absxi
                else:
                    ssq += (absxi / scale)**2
                # end if
            # end if
            i += inc
        # end while
        norm = scale * sqrt(ssq)
    # end if
    return norm


def shrink(double[:, ::1] D, double C_lasso, double C_group, double C_ridge):
    """Perform L2 group-lass shrinkage on the matrix `D`.

    Soft thresholds each element of `D` and forces group sparsity on the rows.
    Finally applies ridge shrinkage to the result.

    Parameters
    ----------
    D : 2d array, shape = [m, n]
        The dense matrix to apply L2 group-sparse operator to.

    C_lasso : float
        Nonnegative value for Lasso regularization.

    C_group : float
        Nonnegative value for row-group regularization.

    C_ridge : float
        Nonnegative value for Ridge regularization.

    Returns
    -------
    out : 2d array, shape = [m, n]
        The matrix after applying L2 group-sparse shrinkage.
    """
    # shapes
    cdef int d_1 = D.shape[0], k = D.shape[1], one = 1

    # counters and temporary variables
    cdef int i, j
    cdef double shrinkage, nrm, tmp

    # output
    cdef double[:, ::1] out = np.empty_like(D)

    with nogil:
        # Lasso shrinkage
        if C_lasso > 0:
            for i in range(d_1):
                for j in range(k):
                    out[i, j] = soft_threshold(D[i, j], C_lasso)
                # end for
            # end for
        else:
            memcpy(&out[0, 0], &D[0, 0], sizeof(double) * d_1 * k)
        # end if

        # Group shrinkage
        if C_group > 0:
            for i in range(d_1):
                # group shrinkage
                # nrm = dnrm2(&k, &out[i, 0], &d_1)  # norm2(k, &out[i, 0], d_1)
                nrm = dnrm2(&k, &out[i, 0], &one)  # norm2(k, &out[i, 0], d_1)
                if nrm <= C_group:
                    for j in range(k):
                        out[i, j] = 0.
                    # end for
                else:
                    for j in range(k):
                        out[i, j] -= out[i, j] * C_group / nrm
                    # end for
                # end if
            # end for
        # end if

        # Ridge regularization
        if C_ridge > 0:
            for i in range(d_1):
                for j in range(k):
                    out[i, j] /= 1 + C_ridge
                # end for
            # end for
        # end if
    # end with

    return out.base


def shrink_row(double[::1] D, double C_lasso, double C_group, double C_ridge):
    """Perform L2 group-lass shrinkage on a vector `D`.

    Soft thresholds each element of `D` and forces group sparsity on the
    entire array. Finally applies ridge shrinkage to the result.

    Parameters
    ----------
    D : 1d array, shape = [n,]
        The array to apply L2 group-sparse operator to.

    C_lasso : float
        Nonnegative value for Lasso regularization.

    C_group : float
        Nonnegative value for row-group regularization.

    C_ridge : float
        Nonnegative value for Ridge regularization.

    Returns
    -------
    out : 1d array, shape = [n,]
        The array after applying L2 group-sparse shrinkage.
    """
    # shapes
    cdef int k = D.shape[0], one = 1

    # counters and temporary variables
    cdef int i
    cdef double shrinkage, nrm, tmp

    # output
    cdef double[::1] out = np.empty_like(D)

    with nogil:
        # Lasso shrinkage
        if C_lasso > 0:
            for i in range(k):
                out[i] = soft_threshold(D[i], C_lasso)
            # end for
        else:
            memcpy(&out[0], &D[0], sizeof(double) * k)
        # end if

        # Group shrinkage
        if C_group > 0:
            nrm = dnrm2(&k, &out[0], &one)  # norm2(k, &out[i, 0], d_1)
            if nrm <= C_group:
                for i in range(k):
                    out[i] = 0.
                # end for
            else:
                for i in range(k):
                    out[i] -= out[i] * C_group / nrm
                # end for
            # end if
        # end if

        # Ridge regularization
        if C_ridge > 0:
            for i in range(k):
                out[i] /= 1 + C_ridge
            # end for
        # end if
    # end with

    return out.base


def _dense_op_d(object R,
                double[:, ::1] X,
                double[:, ::1] Z,
                double[:, ::1] D,
                double[::1] S,
                int n_threads=1):
    cdef int errcode = -1
    cdef int n_1 = X.shape[0], n_2 = Z.shape[0]
    cdef int d_1 = X.shape[1], k = Z.shape[1]
    cdef int[::1] Sp = R.indptr, Sj = R.indices

    if n_threads != 1:
        errcode = __omp_dense_op_d(
            n_1, d_1, &X[0, 0],
            n_2, k, &Z[0, 0],
            &D[0, 0],
            &Sp[0], &Sj[0], &S[0],
            n_threads)
    else:
        errcode = __op_dense_d(
            n_1, d_1, &X[0, 0],
            n_2, k, &Z[0, 0],
            &D[0, 0],
            &Sp[0], &Sj[0], &S[0])
    # end if

    return errcode


def _sparse_op_d(object R,
                 object X,
                 double[:, ::1] Z,
                 double[:, ::1] D,
                 double[::1] S,
                 int n_threads=1):
    cdef int errcode = -1
    cdef int n_1 = X.shape[0], n_2 = Z.shape[0]
    cdef int d_1 = X.shape[1], k = Z.shape[1]
    cdef int[::1] Sp = R.indptr, Sj = R.indices
    cdef int[::1] Xp = X.indptr, Xj = X.indices
    cdef double[::1] Xx = X.data

    if n_threads != 1:
        errcode = __omp_sparse_op_d(
            n_1, d_1, &Xp[0], &Xj[0], &Xx[0],
            n_2, k, &Z[0, 0],
            &D[0, 0],
            &Sp[0], &Sj[0], &S[0], n_threads)
    else:
        errcode = __sparse_op_d(
            n_1, d_1, &Xp[0], &Xj[0], &Xx[0],
            n_2, k, &Z[0, 0],
            &D[0, 0],
            &Sp[0], &Sj[0], &S[0])
    # end if

    return errcode


def _dense_op_s(object R,
                double[:, ::1] X,
                double[:, ::1] Z,
                double[::1] S,
                double[:, ::1] D,
                int n_threads=1):
    cdef int errcode = -1
    cdef int n_1 = X.shape[0], n_2 = Z.shape[0]
    cdef int d_1 = X.shape[1], k = Z.shape[1]
    cdef int[::1] Sp = R.indptr, Sj = R.indices

    if n_threads != 1:
        errcode = __omp_dense_op_s(
            n_1, d_1, &X[0, 0],
            n_2, k, &Z[0, 0],
            &Sp[0], &Sj[0], &S[0],
            &D[0, 0], n_threads)
    else:
        errcode = __dense_op_s(
            n_1, d_1, &X[0, 0],
            n_2, k, &Z[0, 0],
            &Sp[0], &Sj[0], &S[0],
            &D[0, 0])
    # end if

    return errcode


def _sparse_op_s(object R,
                 object X,
                 double[:, ::1] Z,
                 double[::1] S,
                 double[:, ::1] D,
                 int n_threads=1):
    cdef int errcode = -1
    cdef int n_1 = X.shape[0], n_2 = Z.shape[0]
    cdef int d_1 = X.shape[1], k = Z.shape[1]
    cdef int[::1] Sp = R.indptr, Sj = R.indices
    cdef int[::1] Xp = X.indptr, Xj = X.indices
    cdef double[::1] Xx = X.data

    if n_threads != 1:
        errcode = __omp_sparse_op_s(
            n_1, d_1, &Xp[0], &Xj[0], &Xx[0],
            n_2, k, &Z[0, 0],
            &Sp[0], &Sj[0], &S[0],
            &D[0, 0], n_threads)
    else:
        errcode = __sparse_op_s(
            n_1, d_1, &Xp[0], &Xj[0], &Xx[0],
            n_2, k, &Z[0, 0],
            &Sp[0], &Sj[0], &S[0],
            &D[0, 0])
    # end if

    return errcode
