#ifndef DENSE_H
#define DENSE_H

#include <stdint.h>     // for uint32_t
#include "config.h"    // for D_IN_DIM, D_OUT_DIM

void dense(float* input) {
    for (uint32_t i = 0; i < D_OUT_DIM; ++i) {
        float sum = dense_biases[i];
        for (uint32_t j = 0; j < D_IN_DIM; ++j)
            sum += input[j] * dense_weights[j * D_OUT_DIM + i];
        dense_out[i] = sum;
    }
}

#endif // DENSE_H
