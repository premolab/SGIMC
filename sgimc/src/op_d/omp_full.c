#ifndef DISABLE_OPENMP
#include "common.h"
#include "threads.h"

#include "cblas.h"


/*
 * OMP multi threaded versions
 */
int omp_op_d(const int *indptr, const int *indices,
             const int n_1, const int d_1, const double *X,
             const int n_2, const int k, const double *Z,
             const double *D, double *out,
             const int n_threads)
{
    const int n_effective_threads = get_max_threads(n_threads);

    const int BLOCK_SIZE = 256;
    int errcode = -1;

    double *XD = NULL, **local_tmp = NULL;

    XD = (double *) malloc(BLOCK_SIZE * k * sizeof(double));
    if(XD == NULL) goto lbl_exit;

    local_tmp = (double **) alloc_local(n_effective_threads,
                                        BLOCK_SIZE * k * sizeof(double));
    if(local_tmp == NULL) goto lbl_exit;

    // compute entries of the CSR-sparse XDZ'
    int i = 0, i_base = 0, i_stop = 0;
    while(i < n_1) {
        if(indptr[i + 1] == indptr[i]) {
            ++i;
            continue;
        }

        // Compute the next block of X D
        if(i >= i_stop) {
            i_base = i;
            i_stop = min(n_1, BLOCK_SIZE + i_base);

            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        i_stop - i_base, k, d_1,
                        1.0, &X[i_base], n_1, D, d_1,
                        0.0, XD, BLOCK_SIZE);
        }

        #pragma omp parallel \
                    shared(indptr, indices, XD, Z, out) \
                    num_threads(n_effective_threads)
        {
            int j, l, size;  // thrad-private variables
            double * const tmp = local_tmp[omp_get_thread_num()];

            #pragma omp for schedule(static) nowait
            for(int inx=i_base; inx < i_stop; ++inx) {

                j = indptr[inx];
                while(j < indptr[inx + 1]) {
                    size = min(indptr[inx + 1] - j, BLOCK_SIZE);

                    // Collect sparse columns from Z into a dense matrix
                    for(l = 0; l < size; ++l)
                        cblas_dcopy(k,
                                    &Z[indices[j + l]], n_2,
                                    &tmp[l], BLOCK_SIZE);

                    // mat-vec multiply
                    cblas_dgemv(CblasColMajor, CblasNoTrans, size, k,
                                1.0, tmp, BLOCK_SIZE, &XD[inx - i_base], BLOCK_SIZE,
                                1.0, &out[j], 1);

                    j += size;
                }
            }
        }

        i = i_stop;
    }
    errcode = 0;


lbl_exit: ;
    free_local((void**)local_tmp, n_effective_threads);
    free(XD);

    return errcode;
}


int omp_op_d_experimental(
    const int *indptr, const int *indices,
    const int n_1, const int d_1, const double *X,
    const int n_2, const int k, const double *Z,
    const double *D, double *out,
    const int n_threads)
{
    // const int n_effective_threads = get_max_threads(n_threads);

    // return op_d(indptr, indices, n_1, d_1, X, n_2, k, Z, D, out);

    const int n_effective_threads = get_max_threads(n_threads);

    const int BLOCK_SIZE = 256;
    int errcode = -1;

    const int nnz = indptr[n_1] - indptr[0];
    double *XD = NULL;
    int *r_indices = NULL;

    r_indices = (int *) malloc(nnz * sizeof(int));
    if(r_indices == NULL) goto lbl_exit;

    XD = (double *) malloc(BLOCK_SIZE * k * sizeof(double));
    if(XD == NULL) goto lbl_exit;

    // compute entries of the CSR-sparse XDZ'
    int i, j;
    for(i = 0; i < n_1; ++i)
        for(j = indptr[i]; j < indptr[i + 1]; ++j)
            r_indices[j] = i;

    int i_base = 0, i_stop = 0;
    i = 0;
    while(i < n_1) {
        if(indptr[i + 1] == indptr[i]) {
            ++i;
            continue;
        }

        // Compute the next block of X D
        if(i >= i_stop) {
            i_base = i;
            i_stop = min(n_1, BLOCK_SIZE + i_base);

            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        i_stop - i_base, k, d_1,
                        1.0, &X[i_base], n_1, D, d_1,
                        0.0, XD, BLOCK_SIZE);
        }

        #pragma omp parallel \
                    shared(indptr, indices, XD, Z, out) \
                    num_threads(n_effective_threads)
        {
            #pragma omp for schedule(static) nowait
            for(j = indptr[i_base]; j < indptr[i_stop]; ++j)
                out[j] = cblas_ddot(k, &Z[indices[j]], n_2,
                                    &XD[r_indices[j] - i_base], BLOCK_SIZE);
        }

        i = i_stop;
    }
    errcode = 0;


lbl_exit: ;
    free(XD);
    free(r_indices);

    return errcode;
}

#endif
