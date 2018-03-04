#include "common.h"

#include "cblas.h"


int diag_op_s(const int *Sp, const int *Sj,
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
    const int nnz = Sp[n_1] - Sp[0];

    int j_base = Sp[0];
    int j_stop = min(nnz, j_base + BLOCK_SIZE);
    for(i = 0; i < n_1; ++i) {
        j = Sp[i];
        while(j < Sp[i + 1]) {
            const double *a = &X[i], *b = &Z[Sj[j]];

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


int diag_op_d(const int *Sp, const int *Sj,
         const int n_1, const int n_2, const int k,
         const double *X, const double *Z,
         const double *D, double *out)
{
    const int BLOCK_SIZE = 256;
    int errcode = -1;

    double *tmp = NULL, *XD = NULL;

    tmp = (double *) malloc(BLOCK_SIZE * k * sizeof(double));
    if(tmp == NULL) goto lbl_exit;

    XD = (double *) malloc(BLOCK_SIZE * k * sizeof(double));
    if(XD == NULL) goto lbl_exit;

    // compute entries of the CSR-sparse X diag D Z'
    int i, j, l, size;
    int i_base = 0, i_stop = 0;
    for(i = 0; i < n_1; ++i) {
        if(Sp[i + 1] == Sp[i])
            continue;

        // Compute the next block of X diag D
        if(i >= i_stop) {
            i_base = i;
            i_stop = min(n_1, BLOCK_SIZE + i_base);

            memset(XD, 0, sizeof(double) * k * BLOCK_SIZE);

            // Mutliply columns of X by the assocated value
            int ix = i_base, ixd = 0;
            for(l = 0; l < k; ++l) {
                cblas_daxpy(i_stop - i_base, D[l], &X[ix], 1,
                            &XD[ixd], 1);

                // ld steps
                ix += n_1;
                ixd += BLOCK_SIZE;
            }
        }

        j = Sp[i];
        while(j < Sp[i + 1]) {
            size = min(Sp[i + 1] - j, BLOCK_SIZE);

            // Collect the needed rows of Z into a dense matrix
            for(l = 0; l < size; ++l)
                cblas_dcopy(k,
                            &Z[Sj[j + l]], n_2,
                            &tmp[l], BLOCK_SIZE);

            cblas_dgemv(CblasColMajor, CblasNoTrans, size, k,
                        1.0, tmp, BLOCK_SIZE, &XD[i - i_base], BLOCK_SIZE,
                        1.0, &out[j], 1);

            j += size;
        }
    }
    errcode = 0;


lbl_exit: ;
    free(tmp);
    free(XD);

    return errcode;
}
