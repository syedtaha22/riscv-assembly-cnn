#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>

#include "config.h"    // for CL_IN_DIM, CL_OUT_DIM, etc.


/**
 * @brief Gets the index of the maximum value in a range of a float array.
 *
 * This function iterates through a range of a float array to find the index of the maximum value.
 * If the range is invalid (start is NULL, end is NULL, or start >= end), it sets the prediction to -1.
 *
 * @param start Pointer to the start of the range.
 * @param end Pointer to the end of the range.
 * @note The prediction variable is expected to be defined globally, as it stores the index of the maximum value.
 */
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