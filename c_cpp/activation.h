#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <stdint.h>     // for uint32_t
#include <math.h>      // for expf

/**
 * @brief Computes the softmax of a given input array.
 *
 * @param input Pointer to the input array.
 * @param size Size of the input array.
 * @note The input array is modified in place to contain the softmax values.
 */
void softmax(float* input, uint32_t size) {
    // Compute exponentials and simultaneously accumulate sum
    float sum = 0.0f;
    for (uint32_t i = 0; i < size; ++i) {
        input[i] = expf(input[i]);
        sum += input[i];
    }

    // Normalize
    for (uint32_t i = 0; i < size; ++i) input[i] /= sum;
}

/**
 * @brief Applies the ReLU activation function to a given input array.
 * @param input Pointer to the input array.
 * @param size Size of the input array.
 * @return Pointer to the modified input array.
 * @note The ReLU function replaces negative values with zero.
 */
void ReLU(float* input, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        input[i] = (input[i] > 0) ? input[i] : 0;
    }
}


#endif // ACTIVATION_H

