# =================================================================================================
#
# // Conv2d parameters
# #define CL_IN_DIM        28
# #define CL_OUT_DIM       24
# #define CL_IN_CHANNELS   1
# #define CL_NUM_FILTERS   8
# #define CL_FILTER_DIM    5
# #define CL_STRIDE        1
#
# /* Model parameters */
# float* conv_filters;   // 4D array for filters: To be initialized as a 1D array, in the .data section
# float* conv_biases;    // 1D array for biases:  To be initialized as a 1D array, in the .data section
#
#
# float* conv2d(float* input, float* filters, float* biases) {
#     // Allocate output array
#     float* output = (float*)malloc(CL_OUT_DIM * CL_OUT_DIM * CL_NUM_FILTERS * sizeof(float));
#     float local_output[24 * 24 * 8] = { 0 };
#
#     // Perform convolution
#     for (uint32_t k = 0; k < CL_NUM_FILTERS; ++k) {
#         for (uint32_t i = 0; i < CL_OUT_DIM; ++i) {
#             for (uint32_t j = 0; j < CL_OUT_DIM; ++j) {
#                 float sum = 0.0f;
#                 for (uint32_t fi = 0; fi < CL_FILTER_DIM; ++fi) {
#                     for (uint32_t fj = 0; fj < CL_FILTER_DIM; ++fj) {
#                         for (uint32_t c = 0; c < CL_IN_CHANNELS; ++c) {
#                             uint32_t input_index = (i * CL_STRIDE + fi) * (CL_IN_DIM * CL_IN_CHANNELS) + (j * CL_STRIDE + fj) * CL_IN_CHANNELS + c;
#
#                             uint32_t filter_index = k * (CL_FILTER_DIM * CL_FILTER_DIM * CL_IN_CHANNELS) + fi * (CL_FILTER_DIM * CL_IN_CHANNELS) + fj * CL_IN_CHANNELS + c;
#                             sum += input[input_index] * filters[filter_index];
#                         }
#                     }
#                 }
#                 uint32_t output_index = k * (CL_OUT_DIM * CL_OUT_DIM) + i * CL_OUT_DIM + j;
#                 // uint32_t output_index = i * (cl_out_dim * cl_num_filters) + j * cl_num_filters + k;
#                 output[output_index] = sum + biases[k];
#                 local_output[output_index] = sum + biases[k]; // Store the result with the bias added
#             }
#         }
#     }
#
#     return output;
# }
#
# =================================================================================================
#define STDOUT 0xd0580000


# For now just using label as _start, will change it later (TODO)
.section .text
.global _start

conv2d:
    addi a0, x0, 20              # Number of elements
    li x1, 0xd0580000            # Load address for output
    la a1, a                     # Load address of first vector
    la a2, b                     # Load address of second vector
    la a3, result                # Load address for result


_finish:
    li x3, 0xd0580000
    addi x5, x0, 0xff
    sb x5, 0(x3)
    beq x0, x0, _finish
.rept 100
    nop
.endr


.data

# Definitions from #define (from the C code)
CL_IN_DIM:         .word 28              # Input dimension (28x28)
CL_OUT_DIM:        .word 24              # Output dimension (24x24)
CL_IN_CHANNELS:    .word 01              # Input channels (1)
CL_NUM_FILTERS:    .word 08              # Number of filters (8)
CL_FILTER_DIM:     .word 05              # Filter dimension (5x5)
CL_STRIDE:         .word 01              # Stride (1)

# Convolution parameters
conv_filters:      
    .float  -0.212320,  0.145384, -0.037321,  0.123764,  0.056655
    .float  -0.202739,  0.079783,  0.070867, -0.042695, -0.259383
    .float   0.049911,  0.145552, -0.020537, -0.187723, -0.131252
    .float   0.163713,  0.224705,  0.140069, -0.009085,  0.073234
    .float  -0.080659,  0.010824, -0.086331,  0.203459,  0.244409

    .float  -1.054972, -0.891024, -1.066550, -0.828543, -0.654060
    .float  -0.786513, -0.413009, -0.401602, -0.432393, -0.849205
    .float  -0.190523,  0.045326,  0.216727, -0.030382, -0.359319
    .float   0.350181,  0.340316,  0.487858,  0.364874,  0.259409
    .float   0.482986,  0.353294,  0.225865,  0.342781,  0.341477

    .float  -0.178495, -0.472485, -0.508939,  0.146645,  0.305071
    .float  -1.057735, -0.355156,  0.490745,  0.463817,  0.170792
    .float  -0.433517,  0.564238,  0.552975,  0.197741, -0.279824
    .float   0.272791,  0.584378,  0.121418, -0.373568, -0.231887
    .float   0.299926, -0.031639, -0.407140, -0.156032, -0.061682

    .float   0.273750,  0.310541,  0.470391, -0.055891, -0.402251
    .float   0.379908,  0.387477,  0.225295, -0.402449, -0.661950
    .float   0.408006,  0.495547, -0.003231, -0.497584, -0.559818
    .float   0.426277,  0.293883, -0.450071, -0.411655, -0.463014
    .float   0.308907, -0.218843, -0.678969, -0.654918, -0.878282

    .float   0.192461, -0.125563, -0.226632,  0.006193,  0.225292
    .float   0.208990,  0.043345, -0.004417, -0.264719, -0.372079
    .float   0.035871,  0.160617,  0.313575, -0.121912, -0.594081
    .float  -0.505590,  0.142334,  0.606456, -0.054825, -0.319844
    .float  -0.684762,  0.098203,  0.452869,  0.051462,  0.096053

    .float  -0.120633,  0.164645,  0.282223,  0.475194,  0.291666
    .float  -0.414444, -0.384494,  0.024740,  0.403475,  0.424486
    .float  -0.747294, -0.332456, -0.170921,  0.316539,  0.347904
    .float  -0.782172, -0.574138, -0.243847,  0.141671,  0.409410
    .float  -0.651157, -0.432430, -0.185768,  0.121586,  0.198998

    .float   0.168427, -0.004475, -0.116341, -0.465146, -0.472895
    .float   0.317201,  0.250817,  0.226554, -0.456673, -0.602053
    .float   0.078209,  0.507722,  0.514219, -0.068863, -0.439256
    .float  -0.231946,  0.160473,  0.504466,  0.311824, -0.176354
    .float  -0.263749, -0.127226,  0.223549,  0.458108,  0.486485

    .float   0.137269,  0.230076,  0.218404,  0.414103,  0.389986
    .float   0.270464,  0.570013,  0.722101,  0.660840,  0.469928
    .float  -0.033906,  0.095202,  0.032985,  0.161803,  0.184437
    .float  -0.835555, -0.592377, -0.858343, -0.693995, -0.071462
    .float  -0.532673, -0.534382, -0.728421, -0.574272, -0.164224

conv_biases:
    .float  -0.459817,  0.177505, -0.325453, -0.026671, -0.118034, -0.147214, -0.541557, -0.005194

input_image:
    # Initialize image here.
