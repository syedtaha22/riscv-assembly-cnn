#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <stdint.h>     // for uint32_t

// Euler's number for softmax calculation
#define EULER_NUMBER     2.718281828459045

/**
 * @brief Computes the exponential of a given integer.
 *
 * This function computes the power of Euler's number (e) raised to the given integer x.
 * It handles both positive and negative integers.
 *
 * Defined as a static function to limit its scope to this file.
 *
 * @param x The integer to compute the exponential of.
 * @return The computed exponential value.
 * @note This function uses a simple loop to compute the power of Euler's number.
 */
static float exp_x(int x) {
    float result = 1.0;
    int abs_x = x;
    if (x < 0) abs_x = -x;
    for (int i = 0; i < abs_x; ++i) result *= EULER_NUMBER;
    if (x < 0) return 1.0 / result;
    else return result;
}

/**
 * @brief Computes the softmax of a given input array.
 *
 * Structured for direct mapping to riscv-vector instructions
 *
 * @param input Pointer to the input array.
 * @param size Size of the input array.
 * @note The input array is modified in place to contain the softmax values.
 */
void softmax(float* input, uint32_t size) {

    // Compute the exponential of each element
    for (uint32_t i = 0; i < size; ++i) input[i] = exp_x(input[i]);

    // Can be reduced to vfredosum.vv
    float sum = 0.0f;
    for (uint32_t i = 0; i < size; ++i) sum += input[i];

    // Can be mapped to vfdiv.vf
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

