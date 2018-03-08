#include "common.h"
#include "cblas.h"

#ifndef DISABLE_OPENMP
#include "threads.h"
#endif


int dense_op_s(const int n_1,
               const int d_1,
               const double * const X,
               const int n_2,
               const int k,
               const double * const Z,
               const int * const Sp,
               const int * const Sj,
               const double * const S,
                     double * out)
{
    // compute entries of dense X' S Z (single threaded)
    int errcode = -1;

    double *tmp = NULL;
    tmp = (double *) malloc(sizeof(double) * k);
    if(tmp == NULL)
        goto lbl_exit;

    int i, j, l, t;
    for(i = 0; i < n_1; ++i) {
        // get \tau_i = \sum_{j:(i,j)\in \Omega} S_{ij} e_j' Z
        memset(tmp, 0, k * sizeof(double));
        for(j = Sp[i]; j < Sp[i + 1]; ++j) {
            const double Sij = S[j];
            const double * const row = &Z[Sj[j] * k];
            for(t = 0; t < k; ++t) {
                tmp[t] += row[t] * Sij;  // Z is n_2 x k  row-major
            }
        }

        // if(Sp[i] == Sp[i + 1])
        //     continue;

        // multiply X'e_i \tau_i
        const double * const Xi = &X[i * d_1];
        for(l = 0; l < d_1; ++l) {
            const double Xil = Xi[l];  // `X` is n_1 x d_1 row-major
            double *row = &out[l * k];  // `out` is d_1 x k row-major
            for(t = 0; t < k; ++t) {
                row[t] += tmp[t] * Xil;
            }
        }
    }

    errcode = 0;

lbl_exit: ;
    free(tmp);

    return errcode;
}


int dense_op_d(const int n_1,
               const int d_1,
               const double * const X,
               const int n_2,
               const int k,
               const double * const Z,
               const double * const D,
               const int * const Sp,
               const int * const Sj,
                     double * out)
{
    // compute entries of the CSR sparse X D Z' (single threaded)
    int errcode = -1;

    double *tmp = NULL;
    tmp = (double *) malloc(sizeof(double) * k);
    if(tmp == NULL)
        goto lbl_exit;

    int i, j, l, t;
    for(i = 0; i < n_1; ++i) {
        const double * const Xi = &X[i * d_1];  // `X` is n_1 x d_1 row-major
        // if(Sp[i] == Sp[i + 1])
        //     continue;

        // compute e_i' XD
        memset(tmp, 0, sizeof(double) * k);
        for(l = 0; l < d_1; ++l) {
            const double Xil = Xi[l];
            const double * const row = &D[l * k];  // `D` is d_1 x k row-major
            for(t = 0; t < k; ++t) {
                // compute sum_l e_i' X e_l e_l' D e_t
                tmp[t] += Xil * row[t];
            }
        }

        // compute e_i' XD e_t e_t' Z' e_{Sj[j]}
        for(j = Sp[i]; j < Sp[i+1]; ++j) {
            const double * const row = &Z[Sj[j] * k];  // `Z` is n_2 x k row-major
            for(t = 0; t < k; ++t)
                out[j] += tmp[t] * row[t];
        }
    }

    errcode = 0;

lbl_exit: ;
    free(tmp);

    return errcode;
}


#ifndef DISABLE_OPENMP

int omp_dense_op_s(const int n_1,
                   const int d_1,
                   const double * const X,
                   const int n_2,
                   const int k,
                   const double * const Z,
                   const int * const Sp,
                   const int * const Sj,
                   const double * const S,
                         double * out,
                   const int n_threads)
{
    // compute entries of dense X' S Z
    int errcode = -1;

    // shared thread variables
    const int n_effective_threads = get_max_threads(n_threads);

    // #pragma omp parallel for schedule(static) reduction(+:f)
    int i, l, j, t;
    double *tmp;

    double * const *local = (double * const *) \
        alloc_local(n_effective_threads, k * d_1 * sizeof(double));
    if(local == NULL)
        goto lbl_exit;

    #pragma omp parallel shared(local) \
                num_threads(n_effective_threads) \
                private(i, l, j, t, tmp)
    {
        double * const buf = local[omp_get_thread_num()];
        tmp = (double *) malloc(k * sizeof(double));

        #pragma omp for schedule(dynamic,50) nowait
        for(i = 0; i < n_1; ++i) {
            // get \tau_i = \sum_{j:(i,j)\in \Omega} S_{ij} e_j' Z
            memset(tmp, 0, k * sizeof(double));
            for(j = Sp[i]; j < Sp[i + 1]; ++j) {
                const double Sij = S[j];
                const double * const row = &Z[Sj[j] * k];
                for(t = 0; t < k; ++t)
                    tmp[t] += row[t] * Sij;  // Z is n_2 x k row-major
            }

            // multiply X'e_i \tau_i
            const double * const Xi = &X[i * d_1];
            for(l = 0; l < d_1; ++l) {
                const double Xil = Xi[l];
                double *row = &buf[l * k];  // `buf` is d_1 x k row-major
                for(t = 0; t < k; ++t) {
                    row[t] += tmp[t] * Xil;
                }
            }
        }

        free(tmp);

        #pragma omp barrier

        #pragma omp for collapse(2) schedule(static) nowait
        for(i = 0; i < d_1; ++i) {
            for(j = 0; j < k; ++j) {
                for(l = 0; l < n_effective_threads; ++l) {
                    out[j + i * k] += local[l][j + i * k];  // out is d_1 x k row-major
                }
            }
        }
    }
    errcode = 0;

lbl_exit: ;
    free_local((void**)local, n_effective_threads);

    return errcode;
}

int omp_dense_op_d(const int n_1,
                   const int d_1,
                   const double * const X,
                   const int n_2,
                   const int k,
                   const double * const Z,
                   const double * const D,
                   const int * const Sp,
                   const int * const Sj,
                         double * out,
                   const int n_threads)
{
    // compute entries of the CSR sparse X D Z'
    int errcode = -1;

    // shared thread variables
    const int n_effective_threads = get_max_threads(n_threads);

    double * const *local = (double * const *) \
        alloc_local(n_effective_threads, k * sizeof(double));
    if(local == NULL)
        goto lbl_exit;

    int i, j, l, t;
    #pragma omp parallel shared(local) \
                num_threads(n_effective_threads) \
                private(i, j, l, t)
    {
        double * const tmp = local[omp_get_thread_num()];

        #pragma omp for schedule(dynamic,50) nowait
        for(i = 0; i < n_1; ++i) {
            const double * const Xi = &X[i * d_1];
            // if(Sp[i] == Sp[i + 1])
            //     continue;

            // compute e_i' XD
            memset(tmp, 0, sizeof(double) * k);
            for(l = 0; l < d_1; ++l) {
                const double Xil = Xi[l];
                const double * const row = &D[l * k];
                for(t = 0; t < k; ++t) {
                    // compute sum_l e_i' X e_l e_l' D e_t
                    tmp[t] += Xil * row[t];
                }
            }

            // compute e_i' XD e_t e_t' Z' e_{Sj[j]}
            for(j = Sp[i]; j < Sp[i+1]; ++j) {
                const double * const row = &Z[Sj[j] * k];
                for(t = 0; t < k; ++t)
                    out[j] += tmp[t] * row[t];
            }
        }
    }

    errcode = 0;

lbl_exit: ;
    free_local((void**)local, n_effective_threads);

    return errcode;
}

#endif
