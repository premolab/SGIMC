#include "common.h"
#include "cblas.h"


int dense_op_s(const int n_1,
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
    // compute entries of dense X' S Z (single threaded)
    int errcode = -1;

    double *tmp = NULL;
    tmp = (double *) malloc(sizeof(double) * max(d_1, k));
    if(tmp == NULL)
        goto lbl_exit;

    int i, j, l, ll, t, tt;
    for(i = 0; i < n_1; ++i) {
        // get \tau = \sum_{j:i\in \Omega} S_{ij} e_j' Z
        memset(tmp, 0, k * sizeof(double));
        for(j = Sp[i]; j < Sp[i + 1]; ++j) {
            for(t = 0, tt = Sj[j]; t < k; ++t, tt+=n_2)
                tmp[t] += Z[tt] * S[j];  // Z is n_2 x k fortran
            // cblas_daxpy(k, S[j], &Z[Sj[j]], n_2, tmp, 1);
        }

        // multiply X'e_i \tau
        for(l = 0, ll=i; l < d_1; ++l, ll+=n_1) {
            for(t = 0, tt = l; t < k; ++t, tt+=d_1)
                out[tt] += tmp[t] * X[ll];  // out is d_1 x k fortran
        }
    }

    errcode = 0;

lbl_exit: ;
    free(tmp);

    return errcode;
}


int dense_op_d(const int n_1,
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
    // compute entries of the CSR sparse X D Z' (single threaded)
    int errcode = -1;

    double *tmp = NULL;
    tmp = (double *) malloc(sizeof(double) * k);
    if(tmp == NULL)
        goto lbl_exit;

    int i, j, l, ll, t, tt;
    for(i = 0; i < n_1; ++i) {
        // compute e_i' XD
        memset(tmp, 0, sizeof(double) * k);
        for(l = 0, ll=i; l < d_1; ++l, ll+=n_1) {
            for(t = 0, tt = l; t < k; ++t, tt+=d_1)
                // compute sum_l e_i' X e_l e_l' D e_t
                tmp[t] += X[ll] * D[tt];
        }

        // compute e_i' XD e_t e_t' Z' e_{Sj[j]}
        for(j = Sp[i]; j < Sp[i+1]; ++j)
            for(t = 0, tt = Sj[j]; t < k; ++t, tt+=n_2)
                out[j] += tmp[t] * Z[tt];
    }

    errcode = 0;

lbl_exit: ;
    free(tmp);

    return errcode;
}


#ifndef DISABLE_OPENMP
#include "threads.h"


int omp_dense_op_s(const int n_1,
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
    // compute entries of dense X' S Z
    int errcode = -1;

    // shared thread variables
    const int n_effective_threads = get_max_threads(n_threads);

    // #pragma omp parallel for schedule(static) reduction(+:f)
    #pragma omp parallel \
                shared(Sp, Sj, S, X, Z) \
                num_threads(n_effective_threads)
    {
        // Notes: row-col parallel doesn't work (race)
        #pragma omp for schedule(static) nowait
        for(int t = 0; t < k; ++t) {
            const double *row = &Z[t * n_2];  // Z is n_2 x k fortran
            double *col = &out[t * d_1];  // out is d_1 x k fortran

            for(int i = 0; i < n_1; ++i) {
                // get \tau_{it} = \sum_{j:i\in \Omega} S_{ij} e_j' Z e_t
                double tmp = 0;
                for(int j = Sp[i]; j < Sp[i + 1]; ++j)
                    tmp += row[Sj[j]] * S[j];  // Z is n_2 x k fortran

                // compute out_{lt} = e_l' X e_i \tau_{it}
                for(int l = 0, ll=i; l < d_1; ++l, ll+=n_1)
                    col[l] += tmp * X[ll];
            }
        }
    }

    errcode = 0;

    return errcode;
}


int omp_dense_op_d(const int n_1,
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
    // compute entries of the CSR sparse X D Z'
    int errcode = -1;

    // shared thread variables
    const int n_effective_threads = get_max_threads(n_threads);

    double * const *local = (double * const *) \
        alloc_local(n_effective_threads, k * sizeof(double));
    if(local == NULL)
        goto lbl_exit;

    #pragma omp parallel \
                shared(Sp, Sj, out, X, Z, local) \
                num_threads(n_effective_threads)
    {
        double * const tmp = local[omp_get_thread_num()];

        #pragma omp for schedule(static) nowait
        for(int i = 0; i < n_1; ++i) {
            // compute e_i' XD
            memset(tmp, 0, sizeof(double) * k);
            for(int l = 0, ll=i; l < d_1; ++l, ll+=n_1) {
                for(int t = 0, tt = l; t < k; ++t, tt+=d_1)
                    // compute sum_l e_i' X e_l e_l' D e_t
                    tmp[t] += X[ll] * D[tt];
            }

            // compute e_i' XD e_t e_t' Z' e_{Sj[j]}
            for(int j = Sp[i]; j < Sp[i+1]; ++j)
                for(int t = 0, tt = Sj[j]; t < k; ++t, tt+=n_2)
                    out[j] += tmp[t] * Z[tt];
        }
    }

    errcode = 0;

lbl_exit: ;
    free_local((void**)local, n_effective_threads);

    return errcode;
}


#endif
