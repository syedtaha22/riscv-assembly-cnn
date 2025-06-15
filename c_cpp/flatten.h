#ifndef FLATTEN_H
#define FLATTEN_H

#include <stdint.h>     // for uint32_t
#include "config.h"    // for CL_IN_DIM, CL_OUT_DIM, etc.

void flatten(float* input) {
    uint32_t index = 0;
    uint32_t spatial_size = FL_IN_DIM * FL_IN_DIM;

    for (uint32_t i = 0; i < FL_IN_DIM; ++i) {
        for (uint32_t j = 0; j < FL_IN_DIM; ++j) {
            uint32_t input_patch_index = i * FL_IN_DIM + j;

            for (uint32_t c = 0; c < FL_IN_CHANNELS; ++c) {
                flattened_out[index++] = input[c * spatial_size + input_patch_index];
            }
        }
    }
}


#endif // FLATTEN_H