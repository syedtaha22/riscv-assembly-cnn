#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <stdint.h>     // for uint32_t

/**
 * @brief Computes the exponential of a given double using Taylor series expansion.
 *
 * This function computes the power of Euler's number (e) raised to the given double x
 * using Taylor series expansion. It is more accurate for small values of x.
 *
 * Provides a better mapping over to the RISC-V vector instructions.
 *
 * @param x The double to compute the exponential of.
 * @param terms The number of terms to use in the Taylor series expansion.
 * @return The computed exponential value.
 *
 */
static float exp_taylor(double x, int terms) {
    double result = 1.0;
    double term = 1.0;

    for (int i = 1; i < terms; ++i) {
        term *= x / i;
        result += term;
    }

    return result;
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

    // Compute the exponential of each element, vfmul.vv, vfadd.vv, vfdiv.vv
    for (uint32_t i = 0; i < size; ++i) input[i] = exp_taylor(input[i], 1000);

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

