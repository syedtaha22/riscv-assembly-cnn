#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include "config.h"    // for CL_IN_DIM, CL_OUT_DIM, etc.

int get_max_index_in_range(float* start, float* end) {
    float* maxPtr = start;
    int index = 0;
    for (float* ptr = start; ptr < end; ++ptr) {
        if (*ptr > *maxPtr) {
            maxPtr = ptr;
            index = ptr - start;
        }
    }
    return index;
}

#endif // UTILS_H