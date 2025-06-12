#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <stdint.h>     // for uint32_t
#include <math.h>      // for expf

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

void ReLU(float* input, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        input[i] = (input[i] > 0) ? input[i] : 0;
    }
}


#endif // ACTIVATION_H

