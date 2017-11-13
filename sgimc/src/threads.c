#include "common.h"
#include "threads.h"


#ifdef DISABLE_OPENMP

int has_omp(void)
{
    return 0;
}

#else

int has_omp(void)
{
    return 1;
}


int get_max_threads(const int n_threads)
{
    int max_threads = omp_get_max_threads();

    if (n_threads > 0)
        return min(max_threads, n_threads);

    if(n_threads < 0)
        return max(1, max_threads + (n_threads + 1));

    return 1;
}


void *aligned_malloc(size_t size, int align)
{
    void *mem = malloc(size + (align - 1) + sizeof(void *));
    if(mem == NULL)
        return NULL;

    char *amem = ((char *) mem) + sizeof(void *);
    if(align > 1)
        amem += align - ((uintptr_t) amem & (align - 1));

    ((void**)amem)[-1] = mem;

    return amem;
}

void aligned_free(void * mem)
{
    if(mem != NULL)
        free(((void**)mem)[-1]);
}


void **alloc_local(const int n_threads, const size_t n_size)
{
    int i;
    void **buffer = NULL;

    buffer = (void **) malloc(n_threads * sizeof(void *));
    if(buffer == NULL)
        return NULL;

    // memset(buffer, 0, n_threads * sizeof(void *));
    for(i=0; i < n_threads; ++i)
        buffer[i] = NULL;

    for(i=0; i < n_threads; ++i) {
        // buffer[i] = (void *) malloc((n_size + 1048576 - 1) & -1048576);
        buffer[i] = (void *) aligned_malloc((size_t) ((n_size + 256 - 1) & -256), 256);
        if(buffer[i] == NULL) goto lbl_exit;
    }

    for(i=0; i < n_threads; ++i)
        memset(buffer[i], 0, n_size);

    return buffer;

lbl_exit: ;
    free_local(buffer, n_threads);

    return NULL;
}


void free_local(void **buffer, const int n_threads)
{
    if(buffer == NULL)
        return;

    for(int i=0; i < n_threads; ++i)
        aligned_free(buffer[i]);

    free(buffer);
}
#endif
