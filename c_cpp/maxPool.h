#ifndef MAXPOOL_H
#define MAXPOOL_H

#include <stdint.h>    // for uint32_t
#include <stdlib.h>    // for malloc

#include "config.h"    // for MP_IN_DIM, MP_OUT_DIM, etc.

/**
 * @brief Finds the maximum value in a range of a 1D array.
 *
 * This function takes a pointer to the start and end of a range in a 1D array
 * and returns a pointer to the maximum value in that range.
 *
 * @param start Pointer to the start of the range.
 * @param end Pointer to the end of the range.
 * @return Pointer to the maximum value in the range.
 */
float* get_max_in_range(float* start, float* end) {
    float* maxPtr = start;
    for (float* ptr = start; ptr < end; ++ptr) if (*ptr > *maxPtr) maxPtr = ptr;
    return maxPtr;
}


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
 * Optimized for RISC-V vector extension. Maps directly to riscv-vector instructions.
 *
 * @param input Pointer to the flattened input data array (MP_IN_CHANNELS × MP_IN_DIM × MP_IN_DIM).
 * @return Pointer to the dynamically allocated output feature map. (MP_OUT_CHANNELS × MP_OUT_DIM × MP_OUT_DIM).
 *
 * @note The function assumes MP_KERNEL_DIM × MP_KERNEL_DIM pooling windows with stride MP_STRIDE,
 *       applied independently to each channel. All arrays must be stored in row-major order.
 */
float* maxPool(float* input) {
    // Allocate output array
    float* output = (float*)malloc(MP_OUT_DIM * MP_OUT_DIM * MP_OUT_CHANNELS * sizeof(float));

    // Perform max pooling
    for (uint32_t c = 0; c < MP_OUT_CHANNELS; ++c) {
        uint32_t input_channel_offset = c * (MP_IN_DIM * MP_IN_DIM);
        uint32_t output_channel_offset = c * (MP_OUT_DIM * MP_OUT_DIM);

        for (uint32_t i = 0; i < MP_OUT_DIM; ++i) {
            for (uint32_t j = 0; j < MP_OUT_DIM; ++j) {

                // Loop structured for direct mapping to riscv-vector extension
                // This maps to strided segment load in the vector extension
                // vlsseg2e32.v
                float input_batch[MP_KERNEL_DIM * MP_KERNEL_DIM];

                for (uint32_t di = 0; di < MP_KERNEL_DIM; ++di) {
                    for (uint32_t dj = 0; dj < MP_KERNEL_DIM; ++dj) {
                        input_batch[di * MP_KERNEL_DIM + dj] = input[input_channel_offset + (i * MP_STRIDE + di) * MP_IN_DIM + (j * MP_STRIDE + dj)];
                    }
                }

                // Maps to vfredmax.vs
                float maxVal = *get_max_in_range(input_batch, input_batch + (MP_KERNEL_DIM * MP_KERNEL_DIM));

                uint32_t output_index = output_channel_offset + i * MP_OUT_DIM + j;
                output[output_index] = maxVal;

            }
        }
    }
    return output;
}

#endif // MAXPOOL_H