#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>

#define  max(x,y)       ((x)>(y)?(x):(y))
#define  min(x,y)       ((x)<(y)?(x):(y))


inline int isclose(double a, double b)
{
    const double rtol = 1e-5, atol = 1e-8;

    return(fabs(a - b) <= atol +  fabs(b) * rtol);
}


#endif
