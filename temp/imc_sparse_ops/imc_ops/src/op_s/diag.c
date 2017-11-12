#include "common.h"

#include "cblas.h"


int diag_op_s(const int *indptr, const int *indices,
              const int n_1, const int n_2, const int k,
              const double *X, const double *Z,
              const double *S, double *out)
{

    const int BLOCK_SIZE = 256;
    int errcode = -1;

    double *tmp = NULL;

    tmp = (double *) malloc(k * BLOCK_SIZE * sizeof(double));
    if(tmp == NULL) goto lbl_exit;

    int i, j, p;
    // compute entries of the CSR sparse X' S Z
    const int nnz = indptr[n_1] - indptr[0];

    int j_base = indptr[0];
    int j_stop = min(nnz, j_base + BLOCK_SIZE);
    for(i = 0; i < n_1; ++i) {
        j = indptr[i];
        while(j < indptr[i + 1]) {
            const double *a = &X[i], *b = &Z[indices[j]];

            // compute X'e_i \odot Z'e_j
            double *c = &tmp[(j - j_base) * k];
            for(p = 0; p < k; ++p) {

                *c = (*a) * (*b);

                a += n_1;
                b += n_2;

                c++;
            }

            j++;

            // check if accumulated enough columns
            if(j >= j_stop) {
                // mat-vec multiply
                cblas_dgemv(CblasColMajor, CblasNoTrans,
                            k, j_stop - j_base,
                            1.0, tmp, k, &S[j_base], 1,
                            1.0, out, 1);

                // reset
                j_base = j;
                j_stop = min(nnz, j_base + BLOCK_SIZE);
            }
        }
    }
    errcode = 0;


lbl_exit: ;
    free(tmp);

    return errcode;
}
