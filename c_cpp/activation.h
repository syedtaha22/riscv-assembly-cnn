#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <stdint.h>     // for uint32_t

float exp_taylor(float x, int terms) {
    float result = 1.0f;
    float term = 1.0f;

    for (int i = 1; i < terms; ++i) {
        term *= x / i;
        result += term;
    }

    return result;
}

void softmax(float* input, uint32_t size) {
    // Compute exponentials and simultaneously accumulate sum
    float sum = 0.0f;
    for (uint32_t i = 0; i < size; ++i) {
        input[i] = exp_taylor(input[i], 1000);
        sum += input[i];
    }

    // Normalize
    for (uint32_t i = 0; i < size; ++i) input[i] /= sum;
}

void ReLU(float* input, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        input[i] = (input[i] > 0) ? input[i] : 0;
    }
}


#endif // ACTIVATION_H

