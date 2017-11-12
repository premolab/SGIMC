cdef extern from "src/threads.c":
    pass


cdef extern from "src/op_d/full.c":
    int __op_d "full_op_d" (
        const int *indptr, const int *indices,
        const int n_1, const int d_1, const double *X,
        const int n_2, const int k, const double *Z,
        const double *D, double *out) nogil

cdef extern from "src/op_d/omp_full.c":
    int __omp_op_d "omp_op_d"(
        const int *indptr, const int *indices,
        const int n_1, const int d_1, const double *X,
        const int n_2, const int k, const double *Z,
        const double *D, double *out, const int n_threads) nogil

cdef extern from "src/op_s/full.c":
    int __op_s "full_op_s"(
        const int *indptr, const int *indices,
        const int n_1, const int d_1, const double *X,
        const int n_2, const int k, const double *Z,
        const double *S, double *out) nogil

cdef extern from "src/op_s/omp_full.c":
    int __omp_op_s "omp_op_s"(
        const int *indptr, const int *indices,
        const int n_1, const int d_1, const double *X,
        const int n_2, const int k, const double *Z,
        const double *S, double *out, const int n_threads) nogil

cdef extern from "src/op_d/diag.c":
    int __diag_op_d "diag_op_d"(
        const int *indptr, const int *indices,
        const int n_1, const int n_2, const int k,
        const double *X, const double *Z,
        const double *D, double *out) nogil

cdef extern from "src/op_s/diag.c":
    int __diag_op_s "diag_op_s"(
        const int *indptr, const int *indices,
        const int n_1, const int n_2, const int k,
        const double *X, const double *Z,
        const double *S, double *out) nogil
