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


/**
 * @brief Below is an optimised IM2COL + GEMM implementation for 2D convolution.
 *
 * It is written to minimize number of operations and memory accesses. By
 * making use of pointer arithmetic.
 *
 * This code is commented out to avoid conflicts with the main conv2d function (above),
 * which was used for initial benchmarks.
 *
 */


 // #define PATCH_SIZE (CL_FILTER_DIM * CL_FILTER_DIM)
 // #define NUM_PATCHES (CL_OUT_DIM * CL_OUT_DIM)
 // #define CLS_CLINDM (CL_STRIDE * CL_IN_DIM)
 // #define CLFD_NUMP (CL_FILTER_DIM * NUM_PATCHES)
 //
 // void im2col(const float* input, float* im2col_matrix) {
 //     // im2col_matrix size: PATCH_SIZE * NUM_PATCHES
 //     static const uint32_t odps = CL_OUT_DIM * PATCH_SIZE; // output dimension * patch size
 //     static const uint32_t idcs = CL_IN_DIM * CL_STRIDE; // input dimension *  convolution stride
 //
 //     for (uint32_t i = 0; i < CL_OUT_DIM; ++i) {
 //         uint32_t iodps = i * odps; // i = i, od = output dimension, ps = patch size
 //         uint32_t iidcs = i * idcs;   // i = i, id = input dimension, cs = convolution stride
 //
 //         for (uint32_t j = 0; j < CL_OUT_DIM; ++j) {
 //             uint32_t patch_col = j * PATCH_SIZE + iodps; // patch_col = i * CL_OUT_DIM + j
 //             uint32_t input_row_base = j * CL_STRIDE + iidcs; // input
 //
 //             for (uint32_t fi = 0; fi < CL_FILTER_DIM; ++fi) {
 //                 uint32_t input_row = fi * CL_IN_DIM + input_row_base;
 //                 uint32_t patch_base = fi * CL_FILTER_DIM + patch_col;
 //
 //                 // uint32_t patch_row_base = fi * CL_FILTER_DIM;
 //                 float vals[CL_FILTER_DIM]; // vle16, vse16
 //                 for (uint32_t fj = 0; fj < CL_FILTER_DIM; ++fj) vals[fj] = input[input_row + fj];
 //                 for (uint32_t fj = 0; fj < CL_FILTER_DIM; ++fj) im2col_matrix[patch_base + fj] = vals[fj];
 //             }
 //
 //         }
 //     }
 // }
 //
 // void conv2d(float* input) {
 //     // Allocate im2col matrix as 1D array
 //     float im2col_matrix[PATCH_SIZE * NUM_PATCHES];
 //
 //     // Fill im2col matrix
 //     im2col(input, im2col_matrix);
 //
 //     float* bptr = conv_biases; // Pointer to the bias for the current filter
 //     float* optr = conv_out; // Pointer to the output array
 //     float* fptr = conv_filters; // Pointer to the current filter weights
 //
 //     // GEMM: output = filters (CL_NUM_FILTERS x PATCH_SIZE) * im2col_matrix (PATCH_SIZE x NUM_PATCHES)
 //     for (uint32_t f = 0; f < CL_NUM_FILTERS; ++f) {
 //         float* iptr = im2col_matrix; // Pointer to the im2col matrix
 //
 //         // Loop Over all patches
 //         for (uint32_t p = 0; p < NUM_PATCHES; ++p) {
 //             float sum = *bptr; // Start with the bias for the current filter
 //             float* tfptr = fptr; // Pointer to the current filter weights
 //
 //             uint32_t temp = PATCH_SIZE; // Number of elements in the filter
 //
 //             while (temp) {
 //                 temp--; // Decrement the number of elements left to process
 //                 sum += *tfptr * *iptr; // Element-wise multiplication and accumulation
 //                 tfptr++; // Move to the next filter weight
 //                 iptr++; // Move to the next element in im2col matrix
 //             }
 //
 //             *optr = sum; // Store the result in the output
 //             optr++; // Move to the next output element
 //         }
 //         fptr += PATCH_SIZE; // Move to the next filter weights
 //         bptr++;
 //     }
 // }

#endif // CONV2D_H