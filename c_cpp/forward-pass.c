
// include for printf
#include <stdio.h>
#include <stdlib.h>

#include <time.h> // For clock_t and clock()


// Conv2d parameters
int cl_in_dim = 28;
int cl_out_dim = 24;
int cl_in_channels = 1;
int cl_num_filters = 8;
int cl_filter_dim = 5;
int cl_stride = 1;

// MaxPool parameters
int mp_in_dim = 24;
int mp_out_dim = 12;
int mp_in_channels = 8;
int mp_out_channels = 8;
int mp_kernel_dim = 2;
int mp_stride = 2;

// Flatten parameters
int fl_in_dim = 12;
int fl_out_dim = 1152; // 12 * 12 * 8
int fl_in_channels = 8;
int fl_out_channels = 1;

// Dense parameters
int d_in_dim = 1152;
int d_out_dim = 10;

// Tunable parameters for each layer i.e weights and biases

// Conv2d
float**** conv_filters;
float* conv_biases;

// MaxPool - No weights or biases
// Flatten - No weights or biases

// Dense
float** dense_weights;
float* dense_biases;


float* get_max_in_range(float* start, float* end) {
    float* maxPtr = start;
    for (float* ptr = start; ptr < end; ++ptr) {
        if (*ptr > *maxPtr) maxPtr = ptr;
    }
    return maxPtr;
}

float get_max(float a, float b) {
    return a > b ? a : b;
}

float exp_taylor(float x, int n_terms) {
    float result = 1.0f;
    float term = 1.0f;

    for (int i = 1; i <= n_terms; ++i) {
        term *= x / i;  // Compute x^i / i!
        result += term; // Add to the sum
    }

    return result;
}


float relu(float x) {
    return x > 0 ? x : 0;
}

void softmax(float* input, int size) {
    float maxVal = *get_max_in_range(input, input + size);
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        input[i] = exp_taylor(input[i] - maxVal, 10);
        sum += input[i];
    }
    for (int i = 0; i < size; ++i) input[i] /= sum;
}


float*** conv2d(float*** input, float**** filters, float* biases) {
    // Allocate output array
    float*** output = (float***)malloc(cl_out_dim * sizeof(float**));
    for (int i = 0; i < cl_out_dim; ++i) {
        output[i] = (float**)malloc(cl_out_dim * sizeof(float*));
        for (int j = 0; j < cl_out_dim; ++j) output[i][j] = (float*)malloc(cl_num_filters * sizeof(float));
    }

    // Perform convolution
    for (int k = 0; k < cl_num_filters; ++k) {
        for (int i = 0; i < cl_out_dim; ++i) {
            for (int j = 0; j < cl_out_dim; ++j) {
                float sum = 0.0f;
                for (int fi = 0; fi < cl_filter_dim; ++fi) {
                    for (int fj = 0; fj < cl_filter_dim; ++fj) {
                        for (int c = 0; c < cl_in_channels; ++c) sum += input[i * cl_stride + fi][j * cl_stride + fj][c] * filters[k][fi][fj][c];
                    }
                }
                output[i][j][k] = relu(sum + biases[k]);
            }
        }
    }
    return output;
}

float*** maxPool(float*** input) {
    // Allocate output array
    float*** output = (float***)malloc(mp_out_dim * sizeof(float**));
    for (int i = 0; i < mp_out_dim; ++i) {
        output[i] = (float**)malloc(mp_out_dim * sizeof(float*));
        for (int j = 0; j < mp_out_dim; ++j) output[i][j] = (float*)malloc(mp_out_channels * sizeof(float));
    }

    // Perform max pooling
    for (int c = 0; c < mp_out_channels; ++c) {
        for (int i = 0; i < mp_out_dim; ++i) {
            for (int j = 0; j < mp_out_dim; ++j) {
                float maxVal = -1e9;
                for (int di = 0; di < mp_kernel_dim; ++di) {
                    for (int dj = 0; dj < mp_kernel_dim; ++dj) {
                        maxVal = get_max(maxVal, input[i * mp_stride + di][j * mp_stride + dj][c]);
                    }
                }
                output[i][j][c] = maxVal;
            }
        }
    }
    return output;
}

float* flatten(float*** input) {
    // Allocate output array
    float* output = (float*)malloc(fl_out_dim * sizeof(float));
    int index = 0;
    for (int i = 0; i < fl_in_dim; ++i) {
        for (int j = 0; j < fl_in_dim; ++j) {
            for (int c = 0; c < fl_in_channels; ++c) {
                output[index++] = input[i][j][c];
            }
        }
    }
    return output;
}

float* dense(float* input, float** weights, float* biases) {
    // Allocate output array
    float* output = (float*)malloc(d_out_dim * sizeof(float));
    for (int i = 0; i < d_out_dim; ++i) {
        float sum = biases[i];
        for (int j = 0; j < d_in_dim; ++j) {
            sum += input[j] * weights[i][j];
        }
        output[i] = sum;
    }
    return output;
}


// Function to initialize convolutional filters and biases
void init_conv_filters() {
    conv_filters = (float****)malloc(cl_num_filters * sizeof(float***));
    for (int i = 0; i < cl_num_filters; ++i) {
        conv_filters[i] = (float***)malloc(cl_filter_dim * sizeof(float**));
        for (int j = 0; j < cl_filter_dim; ++j) {
            conv_filters[i][j] = (float**)malloc(cl_filter_dim * sizeof(float*));
            for (int k = 0; k < cl_filter_dim; ++k) {
                conv_filters[i][j][k] = (float*)malloc(cl_in_channels * sizeof(float));
                for (int c = 0; c < cl_in_channels; ++c) conv_filters[i][j][k][c] = 0.1f;
            }
        }
    }
}

// Function to initialize convolutional biases
void init_conv_biases() {
    conv_biases = (float*)malloc(cl_num_filters * sizeof(float));
    for (int i = 0; i < cl_num_filters; ++i) conv_biases[i] = 0.1f;
}

// Function to initialize dense weights and biases
void init_dense_weights() {
    dense_weights = (float**)malloc(d_out_dim * sizeof(float*));
    for (int i = 0; i < d_out_dim; ++i) {
        dense_weights[i] = (float*)malloc(d_in_dim * sizeof(float));
        for (int j = 0; j < d_in_dim; ++j) dense_weights[i][j] = 0.1f;
    }
}

// Function to initialize dense biases
void init_dense_biases() {
    dense_biases = (float*)malloc(d_out_dim * sizeof(float));
    for (int i = 0; i < d_out_dim; ++i) dense_biases[i] = 0.1f;
}

// Function to clean up allocated memory for convolutional filters, using free instead of delete
void clean_conv_filters() {
    for (int i = 0; i < cl_num_filters; ++i) {
        for (int j = 0; j < cl_filter_dim; ++j) {
            for (int k = 0; k < cl_filter_dim; ++k) free(conv_filters[i][j][k]);
            free(conv_filters[i][j]);
        }
        free(conv_filters[i]);
    }
    free(conv_filters);
}

// Function to clean up allocated memory for convolutional biases
void clean_conv_biases() { free(conv_biases); }

// Function to clean up allocated memory for dense weights and biases
void clean_dense_weights() {
    for (int i = 0; i < d_out_dim; ++i) free(dense_weights[i]);
    free(dense_weights);
}

// Function to clean up allocated memory for dense biases
void clean_dense_biases() { free(dense_biases); }


float* forward(float*** input) {
    // ---- PLACEHOLDER FOR WEIGHT INIT ----
    init_conv_filters();
    init_conv_biases();
    init_dense_weights();
    init_dense_biases();
    // -------------------------------------

    float*** conv_out = conv2d(input, conv_filters, conv_biases);
    float*** x = maxPool(conv_out);
    float* out = flatten(x);
    float* dense_out = dense(out, dense_weights, dense_biases);
    softmax(dense_out, d_out_dim);

    // -------------- CLEANUP --------------

    // Free allocated memory for conv_out
    for (int i = 0; i < cl_out_dim; ++i) {
        for (int j = 0; j < cl_out_dim; ++j) free(conv_out[i][j]);
        free(conv_out[i]);
    }
    free(conv_out);

    // Free allocated memory for x
    for (int i = 0; i < mp_out_dim; ++i) {
        for (int j = 0; j < mp_out_dim; ++j) free(x[i][j]);
        free(x[i]);
    }
    free(x);

    // Free allocated memory for out
    free(out);

    // -------------------------------------

    // Note: dense_out is returned, so it should not be deleted here
    return dense_out;
}


int main() {
    // Start the clock
    clock_t start = clock();
    int total_runs = 100000;


    for (int i = 0; i < total_runs; i++) {
        // Example MNIST-like input: 28x28 grayscale image
        float*** input = (float***)malloc(28 * sizeof(float**));
        for (int i = 0; i < 28; ++i) {
            input[i] = (float**)malloc(28 * sizeof(float*));
            for (int j = 0; j < 28; ++j) input[i][j] = (float*)malloc(1 * sizeof(float));
        }

        // Initialize input with some values (e.g., all pixels set to 0.5)
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) input[i][j][0] = 0.5f; // Example pixel value
        }

        float* output = forward(input);
        
        
        // Print the output
        // printf("Softmax output:\n");
        // for (int i = 0; i < 10; ++i) printf("%f\n", output[i]);

        // Clean up
        free(output);
        clean_conv_filters();
        clean_conv_biases();
        clean_dense_weights();
        clean_dense_biases();

        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) free(input[i][j]);
            free(input[i]);
        }
        free(input);
    }

    // Stop the clock
    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time taken for %d runs: %f seconds\n", total_runs, time_spent);
    time_spent /= total_runs;
    printf("Time taken per run: %f seconds\n", time_spent);

    return 0;
}
