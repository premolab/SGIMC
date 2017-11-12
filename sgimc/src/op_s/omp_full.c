#ifndef DISABLE_OPENMP

#include "common.h"
#include "threads.h"

#include "cblas.h"


int omp_op_s(const int *indptr, const int *indices,
             const int n_1, const int d_1, const double *X,
             const int n_2, const int k, const double *Z,
             const double *S, double *out,
             const int n_threads)
{
    /* shared thread variabels */
    const int n_effective_threads = get_max_threads(n_threads);

    // compute entries of the CSR sparse X' S Z
    int errcode = -1;

    double * const *local = (double * const *) \
        alloc_local(n_effective_threads, d_1 * k * sizeof(double));
    if(local == NULL) goto lbl_exit;

    // #pragma omp parallel for schedule(static) reduction(+:f)
    // int i, j;
    #pragma omp parallel \
                shared(indptr, indices, S, X, Z, local) \
                num_threads(n_effective_threads)
    {
        double * const buf = local[omp_get_thread_num()];

        #pragma omp for schedule(static) nowait
        for(int i = 0; i < n_1; ++i) {
            for(int j = indptr[i]; j < indptr[i + 1]; ++j)
                cblas_dger(CblasColMajor, d_1, k,
                           S[j], &X[i], n_1, &Z[indices[j]], n_2,
                           buf, d_1);
        }

    }

    for(int i=0; i < n_effective_threads; ++i)
        cblas_daxpy(d_1 * k, 1.0, local[i], 1, out, 1);

    errcode = 0;


lbl_exit: ;
    free_local((void**)local, n_effective_threads);

    return errcode;
}


int omp_op_s_experimental(const int *indptr, const int *indices,
             const int n_1, const int d_1, const double *X,
             const int n_2, const int k, const double *Z,
             const double *S, double *out,
             const int n_threads)
{
    /* shared thread variabels */
    const int n_effective_threads = get_max_threads(n_threads);

    // compute entries of the CSR sparse X' S Z
    int errcode = -1;

    const int nnz = indptr[n_1] - indptr[0];
    double **ptr_local = NULL;
    int *ptr_r_indices = NULL;

    ptr_local = (double **) \
        alloc_local(n_effective_threads, d_1 * k * sizeof(double));
    if(ptr_local == NULL) goto lbl_exit;

    ptr_r_indices = (int *) malloc(nnz * sizeof(int));
    if(ptr_r_indices == NULL) goto lbl_exit;

    for(int i = 0; i < n_1; ++i)
        for(int j=indptr[i]; j < indptr[i + 1]; ++j)
            ptr_r_indices[j] = i;

    double * const * local = ptr_local;
    int const *r_indices = ptr_r_indices;

    #pragma omp parallel \
                shared(indptr, indices, S, X, Z, local, r_indices) \
                num_threads(n_effective_threads)
    {
        double * const buf = local[omp_get_thread_num()];

        #pragma omp for schedule(static)
        for(int j=0; j < nnz; ++j)
            cblas_dger(CblasColMajor, d_1, k,
                       S[j], &X[r_indices[j]], n_1, &Z[indices[j]], n_2,
                       buf, d_1);

    }

    for(int i=0; i < n_effective_threads; ++i)
        cblas_daxpy(d_1 * k, 1.0, local[i], 1, out, 1);

    errcode = 0;


lbl_exit: ;
    free_local((void**)ptr_local, n_effective_threads);
    free(ptr_r_indices);

    return errcode;
}

#endif
