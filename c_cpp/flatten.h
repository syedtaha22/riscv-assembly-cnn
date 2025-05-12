#ifndef FLATTEN_H
#define FLATTEN_H

#include <stdlib.h>     // for malloc
#include <stdint.h>     // for uint32_t

#include "config.h"    // for CL_IN_DIM, CL_OUT_DIM, etc.


/**
 * @brief Flattens a 3D feature map into a 1D array.
 *
 * This function converts a 3D tensor representing a multi-channel 2D feature map into a
 * 1D array in row-major order. The input is assumed to be a flattened (1D) representation
 * of a tensor with dimensions (channels × height × width), and the output is a contiguous
 * 1D array where spatial dimensions (height and width) are traversed before channels.
 *
 * The output is a dynamically allocated 1D array of size FL_OUT_DIM. It is the caller's
 * responsibility to deallocate the returned memory to avoid memory leaks.
 *
 * Optimized and structured for direct mapping to riscv-vector. Although uses an extra loop
 * in c/cpp terms, those loops can be reduced to single instructons
 *
 * @param input Pointer to the flattened input data array (FL_IN_CHANNELS × FL_IN_DIM × FL_IN_DIM).
 * @return Pointer to the dynamically allocated 1D output array (FL_OUT_DIM).
 *
 * @note The function flattens the input by iterating over spatial dimensions first,
 *       followed by channels, producing an output suitable for fully connected layers.
 */
float* flatten(float* input) {
    // Allocate output array
    float* output = (float*)malloc(FL_OUT_DIM * sizeof(float));
    uint32_t index = 0;
    for (uint32_t i = 0; i < FL_IN_DIM; ++i) {
        for (uint32_t j = 0; j < FL_IN_DIM; ++j) {
            // Vectorise the inner loop
            float input_patch[8];
            uint32_t input_patch_index = i * FL_IN_DIM + j;

            // Structured for direct mapping to riscv-v. Can use one vlse32.v (strided vector load)
            for (uint32_t c = 0; c < FL_IN_CHANNELS; ++c) input_patch[c] = input[c * (FL_IN_DIM * FL_IN_DIM) + input_patch_index];

            // Can be saved directly with on single riscv-v instruction. vse32.v
            for (uint32_t c = 0; c < FL_IN_CHANNELS; ++c) output[index++] = input_patch[c];
        }
    }
    return output;
}


#endif // FLATTEN_H