#ifndef MAXPOOL_H
#define MAXPOOL_H

#include <stdint.h>    // for uint32_t
#include <stdlib.h>    // for malloc

#include "config.h"    // for MP_IN_DIM, MP_OUT_DIM, etc.

/**
 * @brief Performs 2D max pooling on multi-channel input data.
 *
 * This function applies a 2D max pooling operation over the input feature map.
 * The input is assumed to be a flattened (1D) representation of a multi-channel 2D image.
 * For each channel, the function selects the maximum value within each non-overlapping
 * pooling window and constructs a downsampled output feature map.
 *
 * The output is a dynamically allocated 1D array representing the flattened 3D feature map
 * (channels × output height × output width). It is the caller's responsibility to deallocate
 * the returned memory to avoid memory leaks.
 *
 *
 * @param input Pointer to the flattened input data array (MP_IN_CHANNELS × MP_IN_DIM × MP_IN_DIM).
 * @return Pointer to the dynamically allocated output feature map. (MP_OUT_CHANNELS × MP_OUT_DIM × MP_OUT_DIM).
 *
 * @note The function assumes MP_KERNEL_DIM × MP_KERNEL_DIM pooling windows with stride MP_STRIDE,
 *       applied independently to each channel. All arrays must be stored in row-major order.
 */
float* maxPool(float* input) {
    float* output = (float*)malloc(MP_OUT_DIM * MP_OUT_DIM * MP_OUT_CHANNELS * sizeof(float));

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

                output[output_offset + i * MP_OUT_DIM + j] = maxVal;
            }
        }
    }

    return output;
}


#endif // MAXPOOL_H