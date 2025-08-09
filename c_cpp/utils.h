#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>     
#include <string.h>

#include "config.h"    // for CL_IN_DIM, CL_OUT_DIM, etc.


/**
 * @brief Gets the index of the maximum value in a range of a float array.
 *
 * This function iterates through a range of a float array to find the index of the maximum value.
 * If the range is invalid (start is NULL, end is NULL, or start >= end), it sets the prediction to -1.
 *
 * @param start Pointer to the start of the range.
 * @param end Pointer to the end of the range.
 * @note The prediction variable is expected to be defined globally, as it stores the index of the maximum value.
 */
void get_max_index_in_range(float* start, float* end) {
    if (start == NULL || end == NULL || start >= end) prediction = -1; // Invalid range

    float max_value = *start;
    prediction = 0; // Default index
    for (float* ptr = start; ptr < end; ++ptr) {
        if (*ptr > max_value) {
            max_value = *ptr;
            prediction = ptr - start; // Calculate index
        }
    }
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