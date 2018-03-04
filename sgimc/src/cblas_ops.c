#include "common.h"
#include "cblas.h"


#define BLOCK_XD


int cblas_op_s(const int n_1,
               const int d_1,
               const double *X,
               const int n_2,
               const int k,
               const double *Z,
               const int *Sp,
               const int *Sj,
               const double *S,
                     double *out)
{
    // compute entries of the CSR sparse X' S Z

    // #pragma omp parallel for schedule(static) reduction(+:f)

    int i, j;
    for(i = 0; i < n_1; ++i) {
        for(j = Sp[i]; j < Sp[i + 1]; ++j)
            cblas_dger(CblasColMajor, d_1, k,
                       S[j], &X[i], n_1, &Z[Sj[j]], n_2,
                       out, d_1);
    }

    return 0;
}


int cblas_op_d(const int n_1,
               const int d_1,
               const double *X,
               const int n_2,
               const int k,
               const double *Z,
               const double *D,
               const int *Sp,
               const int *Sj,
                     double *out)
{
    const int BLOCK_SIZE = 256;
    int errcode = -1;

    double *XD = NULL, *tmp = NULL;

    tmp = (double *) malloc(BLOCK_SIZE * k * sizeof(double));
    if(tmp == NULL) goto lbl_exit;

    // compute X D
    #ifdef BLOCK_XD
    XD = (double *) malloc(BLOCK_SIZE * k * sizeof(double));
    if(XD == NULL) goto lbl_exit;

    #else
    XD = (double *) malloc(n_1 * k * sizeof(double));
    if(XD == NULL) goto lbl_exit;

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n_1, k, d_1, 1.0, X, n_1, D, d_1,
                0.0, XD, n_1);

    #endif

    // compute entries of the CSR-sparse XDZ'
    int i, j, l, size;
    int i_base = 0, i_stop = 0;
    for(i = 0; i < n_1; ++i) {
        if(Sp[i + 1] == Sp[i])
            continue;

        #ifdef BLOCK_XD
        // Compute the next block of X D
        if(i >= i_stop) {
            i_base = i;
            i_stop = min(n_1, BLOCK_SIZE + i_base);

            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        i_stop - i_base, k, d_1,
                        1.0, &X[i_base], n_1, D, d_1,
                        0.0, XD, BLOCK_SIZE);
        }

        #endif

        j = Sp[i];
        while(j < Sp[i + 1]) {
            size = min(Sp[i + 1] - j, BLOCK_SIZE);

            // Collect the necessary rows of Z into a dense matrix
            for(l = 0; l < size; ++l)
                cblas_dcopy(k,
                            &Z[Sj[j + l]], n_2,
                            &tmp[l], BLOCK_SIZE);

            // Detect row index runs
            // run = 1;
            // inx_base = Sj[j + l];
            // while(l + run < size) {
            //     if(Sj[j + l + run] != inx_base + run)
            //         break;
            //     ++run;
            // }

            // mat-vec multiply
            #ifdef BLOCK_XD
            cblas_dgemv(CblasColMajor, CblasNoTrans, size, k,
                        1.0, tmp, BLOCK_SIZE, &XD[i - i_base], BLOCK_SIZE,
                        1.0, &out[j], 1);
            #else
            cblas_dgemv(CblasColMajor, CblasNoTrans, size, k,
                        1.0, tmp, BLOCK_SIZE, &XD[i], n_1,
                        1.0, &out[j], 1);

            #endif

            j += size;
        }
    }
    errcode = 0;


lbl_exit: ;
    free(XD);
    free(tmp);

    return errcode;
}


#ifndef DISABLE_OPENMP
#include "threads.h"


int omp_cblas_op_s(const int n_1,
                   const int d_1,
                   const double *X,
                   const int n_2,
                   const int k,
                   const double *Z,
                   const int *Sp,
                   const int *Sj,
                   const double *S,
                         double *out,
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
                shared(Sp, Sj, S, X, Z, local) \
                num_threads(n_effective_threads)
    {
        double * const buf = local[omp_get_thread_num()];

        #pragma omp for schedule(static) nowait
        for(int i = 0; i < n_1; ++i) {
            for(int j = Sp[i]; j < Sp[i + 1]; ++j)
                cblas_dger(CblasColMajor, d_1, k,
                           S[j], &X[i], n_1, &Z[Sj[j]], n_2,
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


int omp_cblas_op_d(const int n_1,
                   const int d_1,
                   const double *X,
                   const int n_2,
                   const int k,
                   const double *Z,
                   const double *D,
                   const int *Sp,
                   const int *Sj,
                         double *out,
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
        if(Sp[i + 1] == Sp[i]) {
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
                    shared(Sp, Sj, XD, Z, out) \
                    num_threads(n_effective_threads)
        {
            int j, l, size;  // thrad-private variables
            double * const tmp = local_tmp[omp_get_thread_num()];

            #pragma omp for schedule(static) nowait
            for(int inx=i_base; inx < i_stop; ++inx) {

                j = Sp[inx];
                while(j < Sp[inx + 1]) {
                    size = min(Sp[inx + 1] - j, BLOCK_SIZE);

                    // Collect sparse columns from Z into a dense matrix
                    for(l = 0; l < size; ++l)
                        cblas_dcopy(k,
                                    &Z[Sj[j + l]], n_2,
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


#endif
