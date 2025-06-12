#ifndef MAXPOOL_H
#define MAXPOOL_H

#include <stdint.h>    // for uint32_t
#include "config.h"    // for MP_IN_DIM, MP_OUT_DIM, etc.

void maxPool(float* input) {
    uint32_t input_area = MP_IN_DIM * MP_IN_DIM;
    uint32_t output_area = MP_OUT_DIM * MP_OUT_DIM;

    for (uint32_t c = 0; c < MP_OUT_CHANNELS; ++c) {
        uint32_t input_offset = c * input_area;
        uint32_t output_offset = c * output_area;

        for (uint32_t i = 0; i < MP_OUT_DIM; ++i) {
            for (uint32_t j = 0; j < MP_OUT_DIM; ++j) {

                float maxVal = -__builtin_inff(); // Smallest float

                for (uint32_t di = 0; di < MP_KERNEL_DIM; ++di) {
                    uint32_t in_i = i * MP_STRIDE + di;

                    for (uint32_t dj = 0; dj < MP_KERNEL_DIM; ++dj) {
                        uint32_t in_j = j * MP_STRIDE + dj;
                        uint32_t input_idx = input_offset + in_i * MP_IN_DIM + in_j;

                        float val = input[input_idx];
                        if (val > maxVal) maxVal = val;
                    }
                }

                max_pool_out[output_offset + i * MP_OUT_DIM + j] = maxVal;
            }
        }
    }
}


#endif // MAXPOOL_H