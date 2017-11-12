#include "common.h"

#include "cblas.h"


int full_op_s(const int *indptr, const int *indices,
              const int n_1, const int d_1, const double *X,
              const int n_2, const int k, const double *Z,
              const double *S,
              double *out)
{
    // compute entries of the CSR sparse X' S Z

    // #pragma omp parallel for schedule(static) reduction(+:f)

    int i, j;
    for(i = 0; i < n_1; ++i) {
        for(j = indptr[i]; j < indptr[i + 1]; ++j)
            cblas_dger(CblasColMajor, d_1, k,
                       S[j], &X[i], n_1, &Z[indices[j]], n_2,
                       out, d_1);
    }

    return 0;
}


int op_s_experimental(
    const int *indptr, const int *indices,
    const int n_1, const int d_1, const double *X,
    const int n_2, const int k, const double *Z,
    const double *S,
    double *out)
{
    const int BLOCK_SIZE = 256;
    int errcode = -1;

    double *tmp = NULL;

    const int flat_dim = d_1 * k;
    tmp = (double *) malloc(flat_dim * BLOCK_SIZE * sizeof(double));
    if(tmp == NULL) goto lbl_exit;

    memset(tmp, 0, sizeof(double) * flat_dim * BLOCK_SIZE);

    int i, j;

    // compute entries of the CSR sparse X' S Z
    const int nnz = indptr[n_1];

    int j_base = indptr[0];
    int j_stop = min(nnz, j_base + BLOCK_SIZE);
    for(i = 0; i < n_1; ++i) {
        j = indptr[i];
        while(j < indptr[i + 1]) {
            // compute X'e_i \odot Z'e_j
            const double *a = &X[i], *b = &Z[indices[j]];
            double *c = &tmp[(j - j_base) * flat_dim];
            cblas_dger(CblasColMajor, d_1, k,
                       1.0, a, n_1, b, n_2,
                       c, d_1);
            j++;

            // check if accumulated enough columns
            if(j >= j_stop) {
                // mat-vec multiply
                cblas_dgemv(CblasColMajor, CblasNoTrans,
                            flat_dim, j_stop - j_base,
                            1.0, tmp, flat_dim, &S[j_base], 1,
                            1.0, out, 1);

                // reset
                j_base = j;
                j_stop = min(nnz, j_base + BLOCK_SIZE);
                memset(tmp, 0, sizeof(double) * flat_dim * BLOCK_SIZE);
            }
        }
    }
    errcode = 0;


lbl_exit: ;
    free(tmp);

    return errcode;
}
