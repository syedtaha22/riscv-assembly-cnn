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
 *
 * @param input Pointer to the flattened input data array (FL_IN_CHANNELS × FL_IN_DIM × FL_IN_DIM).
 * @return Pointer to the dynamically allocated 1D output array (FL_OUT_DIM).
 *
 * @note The function flattens the input by iterating over spatial dimensions first,
 *       followed by channels, producing an output suitable for fully connected layers.
 */
float* flatten(float* input) {
    float* output = (float*)malloc(FL_OUT_DIM * sizeof(float));
    uint32_t index = 0;
    uint32_t spatial_size = FL_IN_DIM * FL_IN_DIM;

    for (uint32_t i = 0; i < FL_IN_DIM; ++i) {
        for (uint32_t j = 0; j < FL_IN_DIM; ++j) {
            uint32_t input_patch_index = i * FL_IN_DIM + j;

            // Unroll loop over channels for efficiency
            for (uint32_t c = 0; c < FL_IN_CHANNELS; ++c) {
                output[index++] = input[c * spatial_size + input_patch_index];
            }
        }
    }

    return output;
}


#endif // FLATTEN_H