/*************************************************************
 * forward-pass-linear.c but uses float instead of float
**************************************************************/

#include <stdio.h>
#include <stdlib.h>     // for malloc
#include <stdint.h>     // for uint32_t
#include <time.h>       // For clock_t and clock()
#include <string.h>     // For strstr
#include <stdbool.h>    // For bool type

#include "flatten.h"
#include "conv2d.h"
#include "maxPool.h"
#include "activation.h"
#include "dense.h"

#include "utils.h"

float* forward(float* input) {
    if (!params_loaded()) {
        fprintf(stderr, "Error: Model parameters not loaded.\n");
        return NULL;
    }

    float* conv_out = conv2d(input, conv_filters, conv_biases);
    ReLU(conv_out, CL_OUT_DIM * CL_OUT_DIM * CL_NUM_FILTERS); // Apply ReLU activation
    float* x = maxPool(conv_out);
    float* out = flatten(x);
    float* dense_out = dense(out, dense_weights, dense_biases);
    softmax(dense_out, D_OUT_DIM);

    // -------------- CLEANUP --------------
    free(conv_out);
    free(x);
    free(out);
    // -------------------------------------

    // Note: dense_out is returned, so it should not be deleted here
    return dense_out;
}

int main() {
    float* input = (float*)malloc(CL_IN_DIM * CL_IN_DIM * CL_IN_CHANNELS * sizeof(float));
    int correct_predictions = 0;
    load_model_params();

    for (int i = 0; i < 100; i++) {
        int label = read_mnist_sample(fopen("../models/mnist_samples.csv", "r"), input, i);
        if (label < 0) {
            fprintf(stderr, "Error reading sample %d\n", i);
            free(input);
            return -1;
        }

        float* output = forward(input);

        int predicted_label = get_max_index_in_range(output, output + D_OUT_DIM);


        if (predicted_label == label) {
            printf("Correct prediction for image %d: predicted %d, actual %d\n", i, predicted_label, label);
            correct_predictions++;
        }

        free(output);
    }

    // Print accuracy
    printf("Accuracy: %.2f%%\n", (float)correct_predictions / 100 * 100);

    clean_params();
    if (input != NULL) {
        free(input);
        input = NULL;
    }

    return 0;
}
