
// include for printf
#include <stdio.h>


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

float exp_taylor(float x, int n_terms = 10) {
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
        input[i] = exp_taylor(input[i] - maxVal);
        sum += input[i];
    }
    for (int i = 0; i < size; ++i) input[i] /= sum;
}


float*** conv2d(float*** input, float**** filters, float* biases) {
    float*** output = new float** [cl_out_dim];
    for (int i = 0; i < cl_out_dim; ++i) {
        output[i] = new float* [cl_out_dim];
        for (int j = 0; j < cl_out_dim; ++j) output[i][j] = new float[cl_num_filters];
    }

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
    float*** output = new float** [mp_out_dim];
    for (int i = 0; i < mp_out_dim; ++i) {
        output[i] = new float* [mp_out_dim];
        for (int j = 0; j < mp_out_dim; ++j) output[i][j] = new float[mp_out_channels];
    }

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
    float* output = new float[fl_out_dim];
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
    float* output = new float[d_out_dim];
    for (int i = 0; i < d_out_dim; ++i) {
        float sum = biases[i];
        for (int j = 0; j < d_in_dim; ++j) {
            sum += input[j] * weights[i][j];
        }
        output[i] = sum;
    }
    return output;
}

void init_conv_filters() {
    conv_filters = new float*** [cl_num_filters];
    for (int i = 0; i < cl_num_filters; ++i) {
        conv_filters[i] = new float** [cl_filter_dim];
        for (int j = 0; j < cl_filter_dim; ++j) {
            conv_filters[i][j] = new float* [cl_filter_dim];
            for (int k = 0; k < cl_filter_dim; ++k) {
                conv_filters[i][j][k] = new float[cl_in_channels];
                for (int c = 0; c < cl_in_channels; ++c) conv_filters[i][j][k][c] = 0.1f;
            }
        }
    }
}

void init_conv_biases() {
    conv_biases = new float[cl_num_filters];
    for (int i = 0; i < cl_num_filters; ++i) conv_biases[i] = 0.1f;
}


void init_dense_weights() {
    dense_weights = new float* [d_out_dim];
    for (int i = 0; i < d_out_dim; ++i) {
        dense_weights[i] = new float[d_in_dim];
        for (int j = 0; j < d_in_dim; ++j) dense_weights[i][j] = 0.1f;
    }
}

void init_dense_biases() {
    dense_biases = new float[d_out_dim];
    for (int i = 0; i < d_out_dim; ++i) dense_biases[i] = 0.1f;
}

void clean_conv_filters() {
    for (int i = 0; i < cl_num_filters; ++i) {
        for (int j = 0; j < cl_filter_dim; ++j) {
            for (int k = 0; k < cl_filter_dim; ++k) delete[] conv_filters[i][j][k];
            delete[] conv_filters[i][j];
        }
        delete[] conv_filters[i];
    }
    delete[] conv_filters;
}

void clean_conv_biases() { delete[] conv_biases; }

void clean_dense_weights() {
    for (int i = 0; i < d_out_dim; ++i) delete[] dense_weights[i];
    delete[] dense_weights;
}

void clean_dense_biases() { delete[] dense_biases; }


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
    for (int i = 0; i < cl_out_dim; ++i) {
        for (int j = 0; j < cl_out_dim; ++j) delete[] conv_out[i][j];
        delete[] conv_out[i];
    }
    delete[] conv_out;

    for (int i = 0; i < mp_out_dim; ++i) {
        for (int j = 0; j < mp_out_dim; ++j) delete[] x[i][j];
        delete[] x[i];
    }
    delete[] x;

    delete[] out;
    // -------------------------------------

    // Note: dense_out is returned, so it should not be deleted here
    return dense_out;
}


int main() {
    // Example MNIST-like input: 28x28 grayscale image
    float*** input = new float** [28];
    for (int i = 0; i < 28; ++i) {
        input[i] = new float* [28];
        for (int j = 0; j < 28; ++j) {
            input[i][j] = new float[1];
            input[i][j][0] = 0.5f; // Example pixel value
        }
    }

    float* output = forward(input);
    printf("Softmax output:\n");


    // Print the output
    for (int i = 0; i < 10; ++i) printf("%f\n", output[i]);

    // Clean up
    delete[] output;
    clean_conv_filters();
    clean_conv_biases();
    clean_dense_weights();
    clean_dense_biases();

    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) delete[] input[i][j];
        delete[] input[i];
    }
    delete[] input;

    return 0;
}
