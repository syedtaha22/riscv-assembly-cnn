#include <stdint.h>     // for uint32_t
#include <time.h>       // for time_t

#include "conv2d.h"
#include "activation.h"
#include "maxPool.h"
#include "flatten.h"
#include "dense.h"
#include "utils.h"

void forward(float* input) {
    conv2d(input);
    ReLU(conv_out, CL_OUT_DIM * CL_OUT_DIM * CL_NUM_FILTERS); // Apply ReLU activation
    maxPool(conv_out);
    flatten(max_pool_out);
    dense(flattened_out);
    softmax(dense_out, D_OUT_DIM);
}

int main() {
    float input[CL_IN_DIM * CL_IN_DIM * CL_IN_CHANNELS];
    int correct_predictions = 0;
    // load_model_params();
    double total_time = 0.0;
    int samples = 100; // Number of samples to process

    for (int i = 0; i < samples; i++) {
        int label = read_mnist_sample(fopen("../models/mnist_samples.csv", "r"), input, i);
        if (label < 0) {
            fprintf(stderr, "Error reading sample %d\n", i);
            return -1;
        }

        clock_t start = clock();
        forward(input);
        clock_t end = clock();
        double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
        total_time += elapsed_time;

        get_max_index_in_range(dense_out, dense_out + D_OUT_DIM);

        if (prediction == label) {
            printf("Correct   prediction for image %d: predicted %d, actual %d\n", i + 1, prediction, label);
            correct_predictions++;
        }
        else {
            printf("Incorrect prediction for image %d: predicted %d, actual %d\n", i + 1, prediction, label);
        }
    }

    // Print accuracy
    printf("Accuracy: %.2f%%\n", (float)correct_predictions / samples * 100);
    // Print average time per sample
    printf("Average time per sample: %.6f seconds\n", total_time / samples);

    return 0;
}