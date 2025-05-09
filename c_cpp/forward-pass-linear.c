/*************************************************************
 * forward-pass-linear.c but uses float instead of float
**************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>     // for int32_t
#include <time.h>       // For clock_t and clock()
#include <string.h>     // For strstr
#include <stdbool.h>    // For bool type

// Conv2d parameters
#define CL_IN_DIM        28
#define CL_OUT_DIM       24
#define CL_IN_CHANNELS   1
#define CL_NUM_FILTERS   8
#define CL_FILTER_DIM    5
#define CL_STRIDE        1

// MaxPool parameters
#define MP_IN_DIM        24
#define MP_OUT_DIM       12
#define MP_IN_CHANNELS   8
#define MP_OUT_CHANNELS  8
#define MP_KERNEL_DIM    2
#define MP_STRIDE        2

// Flatten parameters
#define FL_IN_DIM        12
#define FL_OUT_DIM       1152 // 12 * 12 * 8
#define FL_IN_CHANNELS   8
#define FL_OUT_CHANNELS  1

// Dense parameters
#define D_IN_DIM         1152
#define D_OUT_DIM        10

// Euler's number for softmax calculation
#define EULER_NUMBER     2.718281828459045

/* Model parameters */
float* conv_filters;   // 4D array for filters
float* conv_biases;    // 1D array for biases
float* dense_weights;  // 2D array for weights
float* dense_biases;   // 1D array for biases

// Function to print a vector (for debugging purposes)
// Takes in 1d array and it's Dimensions, then prints D, WxH grids
void print_output_vector(float* vec, uint32_t width, uint32_t height, uint32_t depth) {
    // upto 2 decimal points
    for (uint32_t c = 0; c < depth; ++c) {
        printf("Channel %d:\n", c);
        for (uint32_t i = 0; i < height; ++i) {
            for (uint32_t j = 0; j < width; ++j) {
                uint32_t index = c * (width * height) + i * width + j;
                printf("%.3lf ", vec[index]);
            }
            printf("\n");
        }
        printf("\n");
    }
}


void print_vector(float* vec, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) printf("%.3lf ", vec[i]);
    printf("\n");
}

float* get_max_in_range(float* start, float* end) {
    float* maxPtr = start;
    for (float* ptr = start; ptr < end; ++ptr) if (*ptr > *maxPtr) maxPtr = ptr;
    return maxPtr;
}

float get_max(float a, float b) { return a > b ? a : b; }

// Function to calculate e^x for any integer or negative x
float exp_x(int x) {
    float result = 1.0;
    int abs_x = x;
    if (x < 0) abs_x = -x;
    for (int i = 0; i < abs_x; ++i) result *= EULER_NUMBER;
    if (x < 0) return 1.0 / result;
    else return result;
}

uint32_t idx_chw(uint32_t k, uint32_t i, uint32_t j, uint32_t height, uint32_t width) {
    return k * height * width + i * width + j;
}

// K = #filters, C = in-channels, F = filter dim
uint32_t idx_kcfhfw(uint32_t k, uint32_t c, uint32_t fh, uint32_t fw, uint32_t C, uint32_t F) {
    // flatten as: k*(C*F*F) + c*(F*F) + fh*F + fw
    return ((k * C + c) * F + fh) * F + fw;
}


/* If this gives a problem, use maxVal */
void softmax(float* input, uint32_t size) {
    // float maxVal = *get_max_in_range(input, input + size);
    float sum = 0.0f;
    for (uint32_t i = 0; i < size; ++i) {
        // input[i] = exp_x(input[i] - maxVal);
        input[i] = exp_x(input[i]);
        sum += input[i];
    }
    for (uint32_t i = 0; i < size; ++i) input[i] /= sum;
}



float* conv2d(float* input, float* filters, float* biases) {
    // Allocate output array
    float* output = (float*)malloc(CL_OUT_DIM * CL_OUT_DIM * CL_NUM_FILTERS * sizeof(float));

    // Perform convolution
    for (uint32_t k = 0; k < CL_NUM_FILTERS; ++k) {
        for (uint32_t i = 0; i < CL_OUT_DIM; ++i) {
            for (uint32_t j = 0; j < CL_OUT_DIM; ++j) {

                /********************************************************************************************************************/
                float sum = 0.0f;
                for (uint32_t fi = 0; fi < CL_FILTER_DIM; ++fi) {
                    for (uint32_t fj = 0; fj < CL_FILTER_DIM; ++fj) {
                        for (uint32_t c = 0; c < CL_IN_CHANNELS; ++c) {
                            uint32_t input_index = (i * CL_STRIDE + fi) * (CL_IN_DIM * CL_IN_CHANNELS) + (j * CL_STRIDE + fj) * CL_IN_CHANNELS + c;

                            uint32_t filter_index = k * (CL_FILTER_DIM * CL_FILTER_DIM * CL_IN_CHANNELS) + fi * (CL_FILTER_DIM * CL_IN_CHANNELS) + fj * CL_IN_CHANNELS + c;
                            sum += input[input_index] * filters[filter_index];
                        }
                    }
                }
                uint32_t output_index = k * (CL_OUT_DIM * CL_OUT_DIM) + i * CL_OUT_DIM + j;
                output[output_index] = sum + biases[k];
                /********************************************************************************************************************/

            }
        }
    }

    return output;
}



float* ReLU(float* input, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        input[i] = (input[i] > 0) ? input[i] : 0;
    }
    return input;
}

float* maxPool(float* input) {
    // Allocate output array
    float* output = (float*)malloc(MP_OUT_DIM * MP_OUT_DIM * MP_OUT_CHANNELS * sizeof(float));

    // Perform max pooling
    for (uint32_t c = 0; c < MP_OUT_CHANNELS; ++c) {
        for (uint32_t i = 0; i < MP_OUT_DIM; ++i) {
            for (uint32_t j = 0; j < MP_OUT_DIM; ++j) {


                float maxVal = -1e9;
                for (uint32_t di = 0; di < MP_KERNEL_DIM; ++di) {
                    for (uint32_t dj = 0; dj < MP_KERNEL_DIM; ++dj) {
                        uint32_t input_index = c * (MP_IN_DIM * MP_IN_DIM) + (i * MP_STRIDE + di) * MP_IN_DIM + (j * MP_STRIDE + dj);
                        maxVal = get_max(maxVal, input[input_index]);
                    }
                }

                uint32_t output_index = c * (MP_OUT_DIM * MP_OUT_DIM) + i * MP_OUT_DIM + j;
                output[output_index] = maxVal;


            }
        }
    }
    return output;
}

float* flatten(float* input) {
    // Allocate output array
    float* output = (float*)malloc(FL_OUT_DIM * sizeof(float));
    uint32_t index = 0;
    for (uint32_t i = 0; i < FL_IN_DIM; ++i) {
        for (uint32_t j = 0; j < FL_IN_DIM; ++j) {
            for (uint32_t c = 0; c < FL_IN_CHANNELS; ++c) {
                uint32_t input_index = c * (FL_IN_DIM * FL_IN_DIM) + i * FL_IN_DIM + j;

                output[index++] = input[input_index];
            }
        }
    }
    return output;
}

float* dense(float* input, float* weights, float* biases) {
    // Allocate output array
    float* output = (float*)malloc(D_OUT_DIM * sizeof(float));

    for (uint32_t i = 0; i < D_OUT_DIM; ++i) {

        float sum = biases[i];
        for (uint32_t j = 0; j < D_IN_DIM; ++j) {
            uint32_t index = j * D_OUT_DIM + i; // 1D index for weights
            sum += input[j] * weights[index];
        }
        output[i] = sum;

    }
    return output;
}


/**
 * @brief Load model parameters from a file.
 *
 * Loads convolutional filters, biases, dense weights, and biases from a text file.
 */
void load_model_params() {
    FILE* fp = fopen("../models/model_params.txt", "r");
    if (fp == NULL) {
        perror("Cannot open model_params.txt");
        return;
    }

    // Allocate memory if not already allocated
    conv_filters = (float*)malloc(CL_NUM_FILTERS * CL_FILTER_DIM * CL_FILTER_DIM * CL_IN_CHANNELS * sizeof(float));
    conv_biases = (float*)malloc(CL_NUM_FILTERS * sizeof(float));
    dense_weights = (float*)malloc(D_OUT_DIM * D_IN_DIM * sizeof(float));
    dense_biases = (float*)malloc(D_OUT_DIM * sizeof(float));

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

    uint32_t total_conv_filters = CL_NUM_FILTERS * CL_FILTER_DIM * CL_FILTER_DIM * CL_IN_CHANNELS;
    idx = 0;
    while (idx < total_conv_filters && fscanf(fp, "%f,", &conv_filters[idx]) == 1) idx++;
    if (idx < total_conv_filters) printf("Warning: Only loaded %u/%u conv filter values\n", idx, total_conv_filters);

    // ===== Load conv_biases =====
    while (fgets(line, sizeof(line), fp)) if (strstr(line, "--- Biases ---")) break;

    idx = 0;
    while (idx < CL_NUM_FILTERS && fscanf(fp, "%f,", &conv_biases[idx]) == 1) idx++;
    if (idx < CL_NUM_FILTERS) printf("Warning: Only loaded %u/%u conv bias values\n", idx, CL_NUM_FILTERS);

    // ===== Load dense_weights =====
    while (fgets(line, sizeof(line), fp)) if (strstr(line, "Layer 5: DENSE")) break;
    while (fgets(line, sizeof(line), fp)) if (strstr(line, "--- Weights ---")) break;

    uint32_t total_dense_weights = D_OUT_DIM * D_IN_DIM;
    idx = 0;
    while (idx < total_dense_weights && fscanf(fp, "%f,", &dense_weights[idx]) == 1) idx++;
    if (idx < total_dense_weights) printf("Warning: Only loaded %u/%u dense weight values\n", idx, total_dense_weights);

    // ===== Load dense_biases =====
    while (fgets(line, sizeof(line), fp)) if (strstr(line, "--- Biases ---")) break;

    idx = 0;
    while (idx < D_OUT_DIM && fscanf(fp, "%f,", &dense_biases[idx]) == 1) idx++;
    if (idx < D_OUT_DIM) printf("Warning: Only loaded %u/%u dense bias values\n", idx, D_OUT_DIM);
    fclose(fp);
}

void clean_params() {
    free(conv_filters);
    free(conv_biases);
    free(dense_weights);
    free(dense_biases);
}

// Function to check if parameters are loaded
bool params_loaded() {
    return conv_filters != NULL && conv_biases != NULL && dense_weights != NULL && dense_biases != NULL;
}


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

    for (int i = 0; i < CL_IN_DIM * CL_IN_DIM; ++i) {
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

        // Get max value and index
        float max_value = output[0];
        int max_index = 0;
        for (int i = 1; i < D_OUT_DIM; ++i) {
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
