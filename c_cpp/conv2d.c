/*************************************************************
 * Conv2D Implementation
**************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>     // for uint32_t

// Conv2d parameters
#define CL_IN_DIM        28
#define CL_OUT_DIM       24
#define CL_IN_CHANNELS   1
#define CL_NUM_FILTERS   8
#define CL_FILTER_DIM    5
#define CL_STRIDE        1

/* Model parameters */
float* conv_filters;   // 4D array for filters: To be initialized as a 1D array, in the .data section
float* conv_biases;    // 1D array for biases:  To be initialized as a 1D array, in the .data section


float* conv2d(float* input, float* filters, float* biases) {
    // Allocate output array
    float* output = (float*)malloc(CL_OUT_DIM * CL_OUT_DIM * CL_NUM_FILTERS * sizeof(float));
    float local_output[24 * 24 * 8] = { 0 };

    // Perform convolution
    for (uint32_t k = 0; k < CL_NUM_FILTERS; ++k) {
        for (uint32_t i = 0; i < CL_OUT_DIM; ++i) {
            for (uint32_t j = 0; j < CL_OUT_DIM; ++j) {
                float sum = 0.0f;
                for (uint32_t fi = 0; fi < CL_FILTER_DIM; ++fi) {
                    for (uint32_t fj = 0; fj < CL_FILTER_DIM; ++fj) {
                        for (uint32_t c = 0; c < CL_IN_CHANNELS; ++c) {
                            uint32_t input_index = (i * CL_STRIDE + fi) * (CL_IN_DIM * CL_IN_CHANNELS) + (j * CL_STRIDE + fj) * CL_IN_CHANNELS + c;

                            uint32_t filter_index = k * (CL_FILTER_DIM * CL_FILTER_DIM * CL_IN_CHANNELS) + fi * (CL_FILTER_DIM * CL_IN_CHANNELS) + fj * CL_IN_CHANNELS + c;
                            sum += input[input_index] * filters[filter_index];
                        }
                    }
                }
                uint32_t output_index = k * (CL_OUT_DIM * CL_OUT_DIM) + i * CL_OUT_DIM + j;
                // uint32_t output_index = i * (cl_out_dim * cl_num_filters) + j * cl_num_filters + k;
                output[output_index] = sum + biases[k];
                local_output[output_index] = sum + biases[k]; // Store the result with the bias added
            }
        }
    }

    return output;
}
