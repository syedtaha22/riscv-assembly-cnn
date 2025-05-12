#ifndef DENSE_H
#define DENSE_H

#include <stdlib.h>     // for malloc
#include <stdint.h>     // for uint32_t

#include "config.h"    // for D_IN_DIM, D_OUT_DIM

/**
 * @brief Performs a fully connected (dense) layer operation.
 *
 * This function applies a dense (fully connected) transformation to the input vector
 * using the provided weight matrix and bias vector. The input is assumed to be a 1D array
 * of size D_IN_DIM. The weights are stored in a flattened 1D array in row-major order
 * (input dimension × output dimension), and the biases are a 1D array of size D_OUT_DIM.
 *
 * The output is a dynamically allocated 1D array of size D_OUT_DIM. It is the caller's
 * responsibility to deallocate the returned memory to avoid memory leaks.
 *
 * Optimized and structured for direct mapping to riscv-vector. Although uses an extra loop
 * in c/cpp terms, those loops can be reduced to single instructons
 *
 * @param input   Pointer to the input vector (D_IN_DIM).
 * @param weights Pointer to the flattened weight matrix
 *                (D_IN_DIM × D_OUT_DIM, stored row-major).
 * @param biases  Pointer to the bias vector (D_OUT_DIM).
 * @return Pointer to the dynamically allocated output vector (D_OUT_DIM).
 *
 * @note Each output unit computes a dot product between the input and its corresponding
 *       weight vector, followed by the addition of a bias term.
 */
float* dense(float* input, float* weights, float* biases) {
    // Allocate output array
    float* output = (float*)malloc(D_OUT_DIM * sizeof(float));

    for (uint32_t i = 0; i < D_OUT_DIM; ++i) {
        float sum = biases[i];
        // Structure the inner loop for direct mapping to riscv-vector

        float input_batch[D_IN_DIM];
        // This maps to a vector load operation. vle32.v
        for (uint32_t j = 0; j < D_IN_DIM; ++j) input_batch[j] = input[j];

        float weights_batch[D_IN_DIM];
        // This maps to a strided load operation. vlse32.v
        for (uint32_t j = 0; j < D_IN_DIM; ++j) weights_batch[j] = weights[j * D_OUT_DIM + i];

        // Compute the dot product
        // This maps to a vector multiply and accumulate operation. vfmacc.vv
        for (uint32_t j = 0; j < D_IN_DIM; ++j) sum += input_batch[j] * weights_batch[j];

        // Store result
        // This maps to a vector store operation. vse32.v
        output[i] = sum;
    }
    return output;
}

#endif // DENSE_H