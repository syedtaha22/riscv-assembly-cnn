#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include "config.h"    // for CL_IN_DIM, CL_OUT_DIM, etc.

void get_max_index_in_range(float* start, float* end) {
    if (start == NULL || end == NULL || start >= end) prediction = -1; // Invalid range

    float max_value = *start;
    prediction = 0; // Default index
    for (float* ptr = start; ptr < end; ++ptr) {
        if (*ptr > max_value) {
            max_value = *ptr;
            prediction = ptr - start; // Calculate index
        }
    }
}

#endif // UTILS_H