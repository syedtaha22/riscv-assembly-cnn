#ifndef CONV2D_H
#define CONV2D_H

#include <stdint.h>    // for uint32_t
#include "config.h"    // for CL_IN_DIM, CL_OUT_DIM, etc.

/**
 * @brief Performs 2D convolution on multi-channel input data using specified filters and biases.
 *
 * This function applies a 2D convolution operation over the input data using the given set of filters
 * and corresponding biases. The input is assumed to be a flattened (1D) representation of a
 * multi-channel 2D image. Each filter is convolved across the spatial dimensions of the input,
 * aggregating across input channels, and a bias is added to each output channel accordingly.
 *
 * The output is a dynamically allocated 1D array representing the flattened 3D feature map
 * (output channels × output height × output width). It is the caller's responsibility to deallocate
 * the returned memory to avoid memory leaks.
 *
 * Optimized to leverage the fact that everything is in row-major order.
 *
 * @param input  Pointer to the flattened input data array (CL_IN_CHANNELS × CL_IN_DIM × CL_IN_DIM).
 * @return Pointer to the dynamically allocated output feature map (CL_NUM_FILTERS × CL_OUT_DIM × CL_OUT_DIM).
 *
 * @note All arrays must be stored in row-major order.
 */
void conv2d(float* input) {
    for (uint32_t k = 0; k < CL_NUM_FILTERS; ++k) {
        for (uint32_t i = 0; i < CL_OUT_DIM; ++i) {
            for (uint32_t j = 0; j < CL_OUT_DIM; ++j) {
                float sum = conv_biases[k];
                for (uint32_t fi = 0; fi < CL_FILTER_DIM; ++fi) {
                    // Compute base index for the input patch row
                    uint32_t input_row_base = (i * CL_STRIDE + fi) * CL_IN_DIM + (j * CL_STRIDE);
                    uint32_t filter_row_base = k * (CL_FILTER_DIM * CL_FILTER_DIM) + fi * CL_FILTER_DIM;
                    for (uint32_t fj = 0; fj < CL_FILTER_DIM; ++fj) {
                        float input_val = input[input_row_base + fj];
                        float filter_val = conv_filters[filter_row_base + fj];
                        sum += input_val * filter_val;
                    }
                }
                // Store result
                uint32_t output_index = k * (CL_OUT_DIM * CL_OUT_DIM) + i * CL_OUT_DIM + j;
                conv_out[output_index] = sum;
            }
        }
    }
}

#endif // CONV2D_H