cimport numpy as np


cdef extern from "src/dense_ops.c":
    int __dense_op_s "dense_op_s"(
        const int n_1,
        const int d_1,
        const double * const X,
        const int n_2,
        const int k,
        const double * const Z,
        const int * const Sp,
        const int * const Sj,
        const double * const S,
              double *out) nogil
    int __omp_dense_op_s "omp_dense_op_s"(
        const int n_1,
        const int d_1,
        const double * const X,
        const int n_2,
        const int k,
        const double * const Z,
        const int * const Sp,
        const int * const Sj,
        const double * const S,
              double * out,
        const int n_threads) nogil

cdef extern from "src/dense_ops.c":
    int __op_dense_d "dense_op_d"(
        const int n_1,
        const int d_1,
        const double * const X,
        const int n_2,
        const int k,
        const double * const Z,
        const double * const D,
        const int * const Sp,
        const int * const Sj,
              double * out) nogil
    int __omp_dense_op_d "omp_dense_op_d"(
        const int n_1,
        const int d_1,
        const double * const X,
        const int n_2,
        const int k,
        const double * const Z,
        const double * const D,
        const int * const Sp,
        const int * const Sj,
              double * out,
        const int n_threads) nogil


cdef extern from "src/sparse_ops.c":
    int __sparse_op_s "sparse_op_s"(
        const int n_1,
        const int d_1,
        const int *Xp,
        const int *Xj,
        const double *X,
        const int n_2,
        const int k,
        const double *Z,
        const int *Sp,
        const int *Sj,
        const double *S,
              double *out) nogil
    int __omp_sparse_op_s "omp_sparse_op_s"(
        const int n_1,
        const int d_1,
        const int *Xp,
        const int *Xj,
        const double *X,
        const int n_2,
        const int k,
        const double *Z,
        const int *Sp,
        const int *Sj,
        const double *S,
              double *out,
        const int n_threads) nogil
    int __sparse_op_d "sparse_op_d"(
        const int n_1,
        const int d_1,
        const int *Xp,
        const int *Xj,
        const double *X,
        const int n_2,
        const int k,
        const double *Z,
        const double *D,
        const int *Sp,
        const int *Sj,
              double *out) nogil
    int __omp_sparse_op_d "omp_sparse_op_d"(
        const int n_1,
        const int d_1,
        const int *Xp,
        const int *Xj,
        const double *X,
        const int n_2,
        const int k,
        const double *Z,
        const double *D,
        const int *Sp,
        const int *Sj,
              double *out,
        const int n_threads) nogil


cdef extern from "src/diag_ops.c":
    int __diag_op_d "diag_op_d"(
        const int *Sp,
        const int *Sj,
        const int n_1,
        const int n_2,
        const int k,
        const double *X,
        const double *Z,
        const double *S,
              double *out) nogil
    int __diag_op_s "diag_op_s"(
        const int *Sp,
        const int *Sj,
        const int n_1,
        const int n_2,
        const int k,
        const double *X,
        const double *Z,
        const double *D,
              double *out) nogil
