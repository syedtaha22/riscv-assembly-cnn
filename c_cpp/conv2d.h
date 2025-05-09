#ifndef CONV2D_H
#define CONV2D_H

#include <stdint.h>    // for uint32_t
#include <stdlib.h>    // for malloc

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
 * @param input  Pointer to the flattened input data array (CL_IN_CHANNELS × CL_IN_DIM × CL_IN_DIM).
 * @param filters Pointer to the flattened filter weights
 *                (CL_NUM_FILTERS × CL_FILTER_DIM × CL_FILTER_DIM).
 * @param biases Pointer to the bias values for each output channel (CL_NUM_FILTERS).
 * @return Pointer to the dynamically allocated output feature map (CL_NUM_FILTERS × CL_OUT_DIM × CL_OUT_DIM).
 *
 *
 *
 * @note As of chore/simplify_logic: Since CL_IN_CHANNELS is 1, we can simply remove the loop, and multiplication
 * @note As of chore/simplify_logic: Since CL_STRIDE is 1, we can simply remove the stride logic
 * @note All arrays must be stored in row-major order.
 */
float* conv2d(float* input, float* filters, float* biases) {
    float* output = (float*)malloc(CL_OUT_DIM * CL_OUT_DIM * CL_NUM_FILTERS * sizeof(float));

    for (uint32_t k = 0; k < CL_NUM_FILTERS; ++k) {
        for (uint32_t i = 0; i < CL_OUT_DIM; ++i) {
            for (uint32_t j = 0; j < CL_OUT_DIM; ++j) {
                float sum = 0.0f;
                for (uint32_t fi = 0; fi < CL_FILTER_DIM; ++fi) {
                    for (uint32_t fj = 0; fj < CL_FILTER_DIM; ++fj) {
                        uint32_t input_index = (i + fi) * (CL_IN_DIM)+(j + fj);
                        uint32_t filter_index = k * (CL_FILTER_DIM * CL_FILTER_DIM) + fi * (CL_FILTER_DIM)+fj;
                        sum += input[input_index] * filters[filter_index];
                    }
                }
                uint32_t output_index = k * (CL_OUT_DIM * CL_OUT_DIM) + i * CL_OUT_DIM + j;
                output[output_index] = sum + biases[k];
            }
        }
    }

    return output;
}

#endif // CONV2D_H