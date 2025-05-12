#ifndef MAXPOOL_H
#define MAXPOOL_H

#include <stdint.h>    // for uint32_t
#include <stdlib.h>    // for malloc

#include "config.h"    // for MP_IN_DIM, MP_OUT_DIM, etc.


/**
 * @brief Computes the maximum of two floating-point numbers.
 *
 * This function returns the maximum of two floating-point numbers.
 * Defined as a static function to limit its scope to this file.
 *
 * @param a First floating-point number.
 * @param b Second floating-point number.
 * @return The maximum of a and b.
 *
 * @note This function is used internally for the max pooling operation.
 */
static float get_max(float a, float b) { return a > b ? a : b; }

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
        for (uint32_t i = 0; i < MP_OUT_DIM; ++i) {
            for (uint32_t j = 0; j < MP_OUT_DIM; ++j) {


                float maxVal = -1e9;
                for (uint32_t di = 0; di < MP_KERNEL_DIM; ++di) {
                    for (uint32_t dj = 0; dj < MP_KERNEL_DIM; ++dj) {
                        uint32_t input_index = c * (MP_IN_DIM * MP_IN_DIM) + (i * MP_STRIDE + di) * MP_IN_DIM + (j * MP_STRIDE + dj);
                        maxVal = get_max(maxVal, input[input_index]);
                    }
                }

                uint32_t output_index = c * (MP_OUT_DIM * MP_OUT_DIM) + i * MP_OUT_DIM + j;
                output[output_index] = maxVal;


            }
        }
    }
    return output;
}

#endif // MAXPOOL_H