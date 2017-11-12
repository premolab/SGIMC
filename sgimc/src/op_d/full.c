#include "common.h"

#include "cblas.h"

#define BLOCK_XD


/*
 * Single threaded versions
 */
int full_op_d(const int *indptr, const int *indices,
              const int n_1, const int d_1, const double *X,
              const int n_2, const int k, const double *Z,
              const double *D,
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
        if(indptr[i + 1] == indptr[i])
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

        j = indptr[i];
        while(j < indptr[i + 1]) {
            size = min(indptr[i + 1] - j, BLOCK_SIZE);

            // Collect the necessary rows of Z into a dense matrix
            for(l = 0; l < size; ++l)
                cblas_dcopy(k,
                            &Z[indices[j + l]], n_2,
                            &tmp[l], BLOCK_SIZE);

            // Detect row index runs
            // run = 1;
            // inx_base = indices[j + l];
            // while(l + run < size) {
            //     if(indices[j + l + run] != inx_base + run)
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
