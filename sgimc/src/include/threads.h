#ifndef DISABLE_OPENMP

#ifndef _OMP_H_
#define _OMP_H_

#include <omp.h>

void **alloc_local(const int n_threads, const size_t n_size);

void free_local(void **buffer, const int n_threads);

int get_max_threads(const int n_threads);


#endif

#endif