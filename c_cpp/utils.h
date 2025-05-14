#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>     // for malloc
#include <stdint.h>     // for uint32_t
#include <time.h>       // For clock_t and clock()
#include <string.h>     // For strstr
#include <stdbool.h>    // For bool type

#include "config.h"    // for CL_IN_DIM, CL_OUT_DIM, etc.


/**
 * @brief Prints a 1D array as a 3D grid.
 *
 * This function takes a 1D array and its dimensions (width, height, depth)
 * and prints it in a grid format. Each channel is printed separately.
 *
 * @param vec Pointer to the 1D array to be printed.
 * @param width Width of the grid.
 * @param height Height of the grid.
 * @param depth Depth of the grid (number of channels).
 * @note The values are printed with 3 decimal points of precision.
 */
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


/**
 * @brief Finds the index of the maximum value in a range of a 1D array.
 *
 * This function takes a pointer to the start and end of a range in a 1D array
 * and returns the index of the maximum value in that range.
 *
 * @param start Pointer to the start of the range.
 * @param end Pointer to the end of the range.
 * @return Index of the maximum value in the range.
 */
int get_max_index_in_range(float* start, float* end) {
    float* maxPtr = start;
    int index = 0;
    for (float* ptr = start; ptr < end; ++ptr) {
        if (*ptr > *maxPtr) {
            maxPtr = ptr;
            index = ptr - start;
        }
    }
    return index;
}

/**
 * @brief Cleans up the allocated parameters.
 *
 * This function frees the memory allocated for the model parameters.
 *
 * @note This function should be called after the model is no longer needed.
 *      It is the caller's responsibility to ensure that the pointers are not NULL before calling this function.
 */
void clean_params() {
    free(conv_filters);
    free(conv_biases);
    free(dense_weights);
    free(dense_biases);
}

/**
 * brief Checks if the model parameters are loaded.
 *
 * This function checks if the model parameters are loaded by verifying if the pointers are not NULL.
 *
 * @return true if all parameters are loaded, false otherwise.
 * @note This function should be called before using the model parameters to ensure they are loaded.
 */
bool params_loaded() {
    return conv_filters != NULL && conv_biases != NULL && dense_weights != NULL && dense_biases != NULL;
}

/**
 * @brief Prints a 1D array.
 *
 * This function takes a 1D array and its size and prints it.
 *
 * @param vec Pointer to the 1D array to be printed.
 * @param size Size of the array.
 * @note The values are printed with 3 decimal points of precision.
 */
void print_vector(float* vec, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) printf("%.3lf ", vec[i]);
    printf("\n");
}


/**
 * @brief Loads model parameters from a file.
 *
 * This function loads the model parameters (filters, biases, weights) from a specified file.
 * It allocates memory for the parameters and reads them from the file.
 *
 * @note The function assumes the file is formatted correctly and contains the expected parameters.
 *       It is the caller's responsibility to free the allocated memory after use.
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



/**
 * @brief Reads a sample from the MNIST dataset.
 *
 * This function reads a sample from the MNIST dataset CSV file and fills the input array with pixel values.
 * It also returns the label of the sample.
 *
 * @param file Pointer to the opened CSV file.
 * @param input Pointer to the array where pixel values will be stored.
 * @param image_index Index of the image to read.
 * @return The label of the image, or -1 on error.
 * @note The input array should be allocated with size CL_IN_DIM * CL_IN_DIM.
 *      The function assumes the CSV file is formatted correctly.
 */
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

#endif // UTILS_H