
#ifndef CONFIG_H
#define CONFIG_H

// Conv2d parameters
#define CL_IN_DIM        28     // Dimension of the input image
#define CL_OUT_DIM       24     // Dimension of the output feature map
#define CL_IN_CHANNELS   1      // Number of input channels (e.g., grayscale image has 1 channel)
#define CL_NUM_FILTERS   8      // Number of filters (output channels)
#define CL_FILTER_DIM    5      // Dimension of the filter (kernel size)
#define CL_STRIDE        1      // Stride of the convolution operation

#define MP_IN_DIM        24     // Dimension of the input feature map
#define MP_OUT_DIM       12     // Dimension of the output feature map
#define MP_IN_CHANNELS   8      // Number of input channels
#define MP_OUT_CHANNELS  8      // Number of output channels
#define MP_KERNEL_DIM    2      // Dimension of the pooling kernel
#define MP_STRIDE        2      // Stride of the pooling operation

// Flatten parameters
#define FL_IN_DIM        12     // Dimension of the input feature map
#define FL_OUT_DIM       1152   // Dimension of the output feature map. 12 * 12 * 8
#define FL_IN_CHANNELS   8      // Number of input channels
#define FL_OUT_CHANNELS  1      // Number of output channels

// Dense parameters
#define D_IN_DIM         1152   // Dimension of the input feature map (flattened)
#define D_OUT_DIM        10     // Dimension of the output (number of classes)

/* Model parameters */
float* conv_filters;   // 4D array for filters
float* conv_biases;    // 1D array for biases
float* dense_weights;  // 2D array for weights
float* dense_biases;   // 1D array for biases

#endif // CONFIG_H