/**
 * @todo: Clean up the code and remove unnecessary comments
 */

 // include for printf
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> // for int32_t
#include <time.h> // For clock_t and clock()
#include <string.h> // For strstr

// Conv2d parameters

const uint32_t cl_in_dim = 28;
const uint32_t cl_out_dim = 24;
const uint32_t cl_in_channels = 1;
const uint32_t cl_num_filters = 8;
const uint32_t cl_filter_dim = 5;
const uint32_t cl_stride = 1;

// MaxPool parameters
const uint32_t mp_in_dim = 24;
const uint32_t mp_out_dim = 12;
const uint32_t mp_in_channels = 8;
const uint32_t mp_out_channels = 8;
const uint32_t mp_kernel_dim = 2;
const uint32_t mp_stride = 2;

// Flatten parameters
const uint32_t fl_in_dim = 12;
const uint32_t fl_out_dim = 1152; // 12 * 12 * 8
const uint32_t fl_in_channels = 8;
const uint32_t fl_out_channels = 1;

// Dense parameters
const uint32_t d_in_dim = 1152;
const uint32_t d_out_dim = 10;

// Tunable parameters for each layer i.e weights and biases

// Conv2d
float* conv_filters; // 4D array for filters
float* conv_biases; // 1D array for biases

// MaxPool - No weights or biases
// Flatten - No weights or biases

// Dense
float* dense_weights; // 2D array for weights
float* dense_biases; // 1D array for biases


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

float exp_taylor(float x, uint32_t n_terms) {
    float result = 1.0f;
    float term = 1.0f;

    for (uint32_t i = 1; i <= n_terms; ++i) {
        term *= x / i;  // Compute x^i / i!
        result += term; // Add to the sum
    }

    return result;
}


float relu(float x) {
    return x > 0 ? x : 0;
}

void softmax(float* input, uint32_t size) {
    float maxVal = *get_max_in_range(input, input + size);
    float sum = 0.0f;
    for (uint32_t i = 0; i < size; ++i) {
        input[i] = exp_taylor(input[i] - maxVal, 10);
        sum += input[i];
    }
    for (uint32_t i = 0; i < size; ++i) input[i] /= sum;
}


float* conv2d(float* input, float* filters, float* biases) {
    // Allocate output array
    float* output = (float*)malloc(cl_out_dim * cl_out_dim * cl_num_filters * sizeof(float));

    // Perform convolution
    for (uint32_t k = 0; k < cl_num_filters; ++k) {
        for (uint32_t i = 0; i < cl_out_dim; ++i) {
            for (uint32_t j = 0; j < cl_out_dim; ++j) {
                float sum = 0.0f;
                for (uint32_t fi = 0; fi < cl_filter_dim; ++fi) {
                    for (uint32_t fj = 0; fj < cl_filter_dim; ++fj) {
                        for (uint32_t c = 0; c < cl_in_channels; ++c) {
                            uint32_t input_index = (i * cl_stride + fi) * (cl_in_dim * cl_in_channels) + (j * cl_stride + fj) * cl_in_channels + c;
                            uint32_t filter_index = k * (cl_filter_dim * cl_filter_dim * cl_in_channels) + fi * (cl_filter_dim * cl_in_channels) + fj * cl_in_channels + c;
                            sum += input[input_index] * filters[filter_index];
                        }
                    }
                }
                uint32_t output_index = i * (cl_out_dim * cl_num_filters) + j * cl_num_filters + k;
                output[output_index] = relu(sum + biases[k]);
            }
        }
    }

    return output;
}

float* maxPool(float* input) {
    // Allocate output array
    float* output = (float*)malloc(mp_out_dim * mp_out_dim * mp_out_channels * sizeof(float));

    // Perform max pooling
    for (uint32_t c = 0; c < mp_out_channels; ++c) {
        for (uint32_t i = 0; i < mp_out_dim; ++i) {
            for (uint32_t j = 0; j < mp_out_dim; ++j) {
                float maxVal = -1e9;
                for (uint32_t di = 0; di < mp_kernel_dim; ++di) {
                    for (uint32_t dj = 0; dj < mp_kernel_dim; ++dj) {
                        uint32_t input_index = (i * mp_stride + di) * (mp_in_dim * mp_in_channels) + (j * mp_stride + dj) * mp_in_channels + c;
                        maxVal = get_max(maxVal, input[input_index]);
                    }
                }
                uint32_t output_index = i * (mp_out_dim * mp_out_channels) + j * mp_out_channels + c;
                output[output_index] = maxVal;
            }
        }
    }
    return output;
}

float* flatten(float* input) {
    // Allocate output array
    float* output = (float*)malloc(fl_out_dim * sizeof(float));
    uint32_t index = 0;
    for (uint32_t i = 0; i < fl_in_dim; ++i) {
        for (uint32_t j = 0; j < fl_in_dim; ++j) {
            for (uint32_t c = 0; c < fl_in_channels; ++c) {
                uint32_t input_index = i * (fl_in_dim * fl_in_channels) + j * fl_in_channels + c;
                output[index++] = input[input_index];
            }
        }
    }
    return output;
}

float* dense(float* input, float* weights, float* biases) {
    // Allocate output array
    float* output = (float*)malloc(d_out_dim * sizeof(float));
    for (uint32_t i = 0; i < d_out_dim; ++i) {
        float sum = biases[i];
        for (uint32_t j = 0; j < d_in_dim; ++j) {
            sum += input[j] * weights[i * d_in_dim + j];
        }
        output[i] = sum;
    }
    return output;
}

// Function to print a vector (for debugging purposes)
// Takes in 1d array and it's Dimensions, then prints D, WxH grids
void print_vector(float* vec, uint32_t width, uint32_t height, uint32_t depth) {
    for (uint32_t c = 0; c < depth; ++c) {
        printf("Channel %d:\n", c);
        for (uint32_t i = 0; i < height; ++i) {
            for (uint32_t j = 0; j < width; ++j) {
                uint32_t index = c * (width * height) + i * width + j;
                printf("%f ", vec[index]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

// Function to initialize convolutional filters and biases as a 1D array
// Note: This is a 4D array in terms of filters, but we will represent it as a 1D array
// void init_conv_filters() {
//     conv_filters = (float*)malloc(cl_num_filters * cl_filter_dim * cl_filter_dim * cl_in_channels * sizeof(float));

//     // Open model_params.txt and read the values into conv_filters



//     for (uint32_t i = 0; i < cl_num_filters; ++i) {
//         for (uint32_t j = 0; j < cl_filter_dim; ++j) {
//             for (uint32_t k = 0; k < cl_filter_dim; ++k) {
//                 for (uint32_t c = 0; c < cl_in_channels; ++c) {
//                     uint32_t index = i * (cl_filter_dim * cl_filter_dim * cl_in_channels) + j * (cl_filter_dim * cl_in_channels) + k * cl_in_channels + c;
//                     conv_filters[index] = 0.1f;
//                 }
//             }
//         }
//     }
// }


#include <stdio.h>
#include <stdlib.h>

// void init_conv_filters() {
//     conv_filters = (float*)malloc(cl_num_filters * cl_filter_dim * cl_filter_dim * cl_in_channels * sizeof(float));
//     if (conv_filters == NULL) {
//         perror("malloc failed");
//         return;
//     }

//     FILE* fp = fopen("model_params.txt", "r");
//     if (fp == NULL) {
//         perror("Cannot open model_params.txt");
//         free(conv_filters);
//         return;
//     }

//     // Skip lines until reaching "--- Weights ---"
//     char line[256];
//     while (fgets(line, sizeof(line), fp)) {
//         if (strstr(line, "--- Weights ---")) {
//             break;
//         }
//     }

//     uint32_t total_filters = cl_num_filters * cl_filter_dim * cl_filter_dim * cl_in_channels;
//     uint32_t idx = 0;

//     while (idx < total_filters && fscanf(fp, "%f,", &conv_filters[idx]) == 1) {
//         idx++;
//     }

//     if (idx < total_filters) {
//         printf("Warning: Only loaded %u/%u filter values\n", idx, total_filters);
//     }

//     fclose(fp);
// }



// // Function to initialize convolutional biases
// void init_conv_biases() {
//     conv_biases = (float*)malloc(cl_num_filters * sizeof(float));
//     for (uint32_t i = 0; i < cl_num_filters; ++i) conv_biases[i] = 0.1f;
// }

// // Function to initialize dense weights as a 1D array
// // Note: This is a 2D array in terms of weights, but we will represent it as a 1D array
// void init_dense_weights() {
//     dense_weights = (float*)malloc(d_out_dim * d_in_dim * sizeof(float));
//     for (uint32_t i = 0; i < d_out_dim; ++i) {
//         for (uint32_t j = 0; j < d_in_dim; ++j) {
//             uint32_t index = i * d_in_dim + j;
//             dense_weights[index] = 0.1f;
//         }
//     }
// }

// // Function to initialize dense biases
// void init_dense_biases() {
//     dense_biases = (float*)malloc(d_out_dim * sizeof(float));
//     for (uint32_t i = 0; i < d_out_dim; ++i) dense_biases[i] = 0.1f;
// }


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void load_model_params() {
    FILE* fp = fopen("../models/model_params.txt", "r");
    if (fp == NULL) {
        perror("Cannot open model_params.txt");
        return;
    }

    // Allocate memory if not already allocated
    conv_filters = (float*)malloc(cl_num_filters * cl_filter_dim * cl_filter_dim * cl_in_channels * sizeof(float));
    conv_biases = (float*)malloc(cl_num_filters * sizeof(float));
    dense_weights = (float*)malloc(d_out_dim * d_in_dim * sizeof(float));
    dense_biases = (float*)malloc(d_out_dim * sizeof(float));

    if (!conv_filters || !conv_biases || !dense_weights || !dense_biases) {
        perror("malloc failed");
        fclose(fp);
        free(conv_filters); free(conv_biases); free(dense_weights); free(dense_biases);
        return;
    }

    char line[256];
    uint32_t idx = 0;

    // ===== Load conv_filters =====
    while (fgets(line, sizeof(line), fp)) if (strstr(line, "--- Weights ---")) break;

    uint32_t total_conv_filters = cl_num_filters * cl_filter_dim * cl_filter_dim * cl_in_channels;
    idx = 0;
    while (idx < total_conv_filters && fscanf(fp, "%f,", &conv_filters[idx]) == 1) idx++;
    if (idx < total_conv_filters) printf("Warning: Only loaded %u/%u conv filter values\n", idx, total_conv_filters);

    // ===== Load conv_biases =====
    while (fgets(line, sizeof(line), fp)) if (strstr(line, "--- Biases ---")) break;

    idx = 0;
    while (idx < cl_num_filters && fscanf(fp, "%f,", &conv_biases[idx]) == 1) idx++;
    if (idx < cl_num_filters) printf("Warning: Only loaded %u/%u conv bias values\n", idx, cl_num_filters);

    // ===== Load dense_weights =====
    while (fgets(line, sizeof(line), fp)) if (strstr(line, "Layer 4: DENSE")) break;
    while (fgets(line, sizeof(line), fp)) if (strstr(line, "--- Weights ---")) break;

    uint32_t total_dense_weights = d_out_dim * d_in_dim;
    idx = 0;
    while (idx < total_dense_weights && fscanf(fp, "%f,", &dense_weights[idx]) == 1) idx++;
    if (idx < total_dense_weights) printf("Warning: Only loaded %u/%u dense weight values\n", idx, total_dense_weights);

    // ===== Load dense_biases =====
    while (fgets(line, sizeof(line), fp)) if (strstr(line, "--- Biases ---")) break;

    idx = 0;
    while (idx < d_out_dim && fscanf(fp, "%f,", &dense_biases[idx]) == 1) idx++;
    if (idx < d_out_dim) printf("Warning: Only loaded %u/%u dense bias values\n", idx, d_out_dim);
    fclose(fp);


    // Print conv_filters for debugging
    // print_vector(conv_filters, cl_filter_dim, cl_filter_dim, cl_num_filters);
    // print_vector(conv_biases, 1, 1, cl_num_filters);
    print_vector(dense_weights, d_in_dim, d_out_dim, 1);
    print_vector(dense_biases, 1, 1, d_out_dim);
}



// Function to clean up allocated memory for convolutional filters, using free instead of delete.
// Note: This is a 4D array in terms of filters, but we will represent it as a 1D array
void clean_params() {
    free(conv_filters);
    free(conv_biases);
    free(dense_weights);
    free(dense_biases);

}

float* forward(float* input) {
    // ---- PLACEHOLDER FOR WEIGHT INIT ----
    // init_conv_filters();
    // init_conv_biases();
    // init_dense_weights();
    // init_dense_biases();
    // load_model_params();
    // -------------------------------------

    float* conv_out = conv2d(input, conv_filters, conv_biases);
    float* x = maxPool(conv_out);
    float* out = flatten(x);
    float* dense_out = dense(out, dense_weights, dense_biases);
    softmax(dense_out, d_out_dim);

    // -------------- CLEANUP --------------

    free(conv_out);
    free(x);
    free(out);

    // -------------------------------------

    // Note: dense_out is returned, so it should not be deleted here
    return dense_out;
}


// int main() {
//     // Start the clock
//     clock_t start = clock();
//     uint32_t total_runs = 10000;

//     for (int i = 0; i < total_runs; i++) {

//         // Example MNIST-like input: 28x28 grayscale image, represent it as a 1d array
//         float* input = (float*)malloc(cl_in_dim * cl_in_dim * cl_in_channels * sizeof(float));
//         for (uint32_t i = 0; i < cl_in_dim * cl_in_dim * cl_in_channels; ++i) input[i] = 0.5f; // Example pixel value
//         float* output = forward(input);


//         // Print the output
//         // printf("Softmax output:\n");
//         // for (uint32_t i = 0; i < 10; ++i) printf("%f\n", output[i]);

//         // Clean up
//         free(output);
//         free(input);
//         clean_params();
//     }

//     // Stop the clock
//     clock_t end = clock();
//     double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

//     printf("Time taken for %d runs: %f seconds\n", total_runs, time_spent);
//     time_spent /= total_runs;


//     printf("Time taken per run: %f seconds\n", time_spent);

//     return 0;
// }

// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>

#define IMAGE_SIZE (28 * 28)

// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>

// #define IMAGE_SIZE (28 * 28)

// Reads the `image_index`-th row from the CSV and fills the input array.
// Returns the label of the image.
// `input` must be a pointer to a `float*` (so we can reallocate it).
int read_mnist_sample(FILE* file, float* input, int image_index) {
    if (file == NULL) {
        fprintf(stderr, "Error: file pointer is NULL.\n");
        return -1;
    }
    if (input == NULL) {
        fprintf(stderr, "Error: input pointer is NULL.\n");
        return -1;
    }

    // Increase buffer size
    char line[8192] = { 0 };

    // Reset file to beginning
    fseek(file, 0, SEEK_SET);

    // Skip to the desired image_index row
    for (int i = 0; i <= image_index; ++i) {
        if (fgets(line, sizeof(line), file) == NULL) {
            fprintf(stderr, "Error: not enough rows in CSV (requested index %d).\n", image_index);
            return -1;
        }
    }

    char* token = strtok(line, ",");
    if (token == NULL) {
        fprintf(stderr, "Error: failed to read label.\n");
        return -1;
    }

    int label = atoi(token);

    for (int i = 0; i < IMAGE_SIZE; ++i) {
        token = strtok(NULL, ",");
        if (token == NULL) {
            fprintf(stderr, "Error: insufficient pixel values at index %d.\n", i);
            return -1;
        }
        input[i] = strtof(token, NULL);
    }

    // close the file
    fclose(file);

    return label;
}



int main() {

    float* input = (float*)malloc(cl_in_dim * cl_in_dim * cl_in_channels * sizeof(float));
    int correct_predictions = 0;
    load_model_params();

    // print conv_filters
    // print_vector(conv_filters, cl_filter_dim, cl_filter_dim, cl_num_filters);

    for (int i = 0; i < 100; i++) {
        int label = read_mnist_sample(fopen("../train/mnist_samples.csv", "r"), input, i);
        if (label < 0) {
            fprintf(stderr, "Error reading sample %d\n", i);
            free(input);
            return -1;
        }

        float* output = forward(input);

        // Get max value and index
        float max_value = output[0];
        int max_index = 0;
        for (int i = 1; i < d_out_dim; ++i) {
            if (output[i] > max_value) {
                max_value = output[i];
                max_index = i;
            }
        }
        // printf("Max value: %f at index: %d\n", max_value, max_index);

        if (max_index == label) {
            printf("Correct prediction for image %d: predicted %d, actual %d\n", i, max_index, label);
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
