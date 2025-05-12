# RISC-V Vectorized Convolutional Neural Network (RV-VCNN)

Implementation of a Convolutional Neural Network (CNN) **in RISC-V assembly**, using the **Vector Extension (RVV)** for parallel computation. This project demonstrates how a compact, trained CNN model can be mapped to low-level RISC-V operations for execution on custom or simulated hardware environments.

---

## Overview

This repository contains:

- A trained MNIST CNN model
- A notebook for training the model
- A parameter dump
- Assembly routines implementing (to be implemented):
  - Convolution
  - ReLU activation
  - Max pooling
  - Flattening
  - Fully connected layer
  - Softmax activation

The goal is to evaluate the **efficiency and feasibility** of deep learning inference using RISC-V with vector instructions.

---

## CNN Architecture

The model utilizes a compact and efficient Convolutional Neural Network (CNN) architecture, optimized for embedded systems, which is suitable for implementation in RISC-V assembly. The architecture consists of several layers, each designed to process the input in a specific way to extract meaningful features and classify the input effectively.

---

### Input Layer

The input to the network is a grayscale image of size $28 \times 28$. Each pixel in the image is represented as a single value in the range [0, 1], resulting in an input tensor $X$ of shape:

$$X \in \mathbb{R}^{28 \times 28 \times 1}$$

---

### Convolutional Layer

The first layer is a convolutional layer designed to extract features from the input image. It uses **8 filters**, each with a size of **$5 \times 5$**. The convolution operation performs as follows:

$$Z_{\text{cl}} = X * W_{\text{cl}} + b_{\text{cl}}$$

Where:

- $X$ is the input image (or the previous layer’s output),
- $W_{\text{cl}}$ represents the convolutional filters (or kernels),
- $b_{\text{cl}}$ is the bias term, and
- $ * $ represents the **convolution** operation, not a regular matrix multiplication.

#### Convolution Operation

The convolution operation slides each **$5 \times 5$ filter** across the input image (or feature map) to compute a **dot product** between the filter and the corresponding region of the image at each spatial location. The result is a new matrix representing the feature map that captures specific patterns such as edges or textures.

Since no padding is applied, the output size is smaller than the input size. Given the input size of **$28 \times 28$** (for MNIST images) and a **$5 \times 5$** filter, the output size is reduced to **$24 \times 24$** for each filter (after applying the stride of 1). 

The bias term $b_{\text{cl}}$ is added after the convolution, contributing an additional offset to each computed value in the resulting feature map.

#### ReLU Activation

After the convolution operation, the result is passed through a **ReLU (Rectified Linear Unit)** activation function to introduce non-linearity. The ReLU function is defined as:

$$
A_{\text{cl}} = \max(0, Z_{\text{cl}})
$$

This operation sets all negative values in $Z_{\text{cl}}$ to 0, while leaving positive values unchanged. This helps to introduce non-linearity into the network, which allows it to model more complex patterns.

#### Output Shape

The final output after the convolution and activation is a feature map with a shape of:

$$
A_{\text{cl}} \in \mathbb{R}^{24 \times 24 \times 8}
$$

Where:

- The height and width of the feature map are **$24 \times 24$**, which results from the convolution with a **$5 \times 5$** filter and a stride of 1.
- The depth (number of channels) is **8**, since we used 8 filters, each generating one feature map.

The output of this layer is then passed to the next layer, which in our case is a max pooling layer.

---

### Max Pooling

The next operation is max pooling, which is used to downsample the spatial dimensions of the feature map. In this layer, a pooling window of size $2 \times 2$ is applied with a stride of 2. This means that for each output element, a $2 \times 2$ region is selected from the input feature map, and the maximum value within that region is chosen.

A more mathematical formulation of the max pooling operation is:

$$A_{\text{mp}}(i,j,k) = \max_{(m,n) \in \{0,1\}^2} A_{\text{cl}}(2i + m,\, 2j + n,\, k)$$

In this equation:

- $A_{\text{cl}}(x, y, k)$ denotes the activation at position $(x,y)$ in the $k$-th channel of the input feature map (output from the convolutional layer).
- The indices $i$ and $j$ are the spatial coordinates for the output feature map $A_{\text{mp}}$; $k$ indexes the channels.
- The expression $2i + m$ and $2j + n$ implements the stride of 2. For each output location $(i,j)$, the corresponding $2 \times 2$ window in $A_{\text{cl}}$ starts at position $(2i, 2j)$ and spans the indices determined by $m,n \in \{0,1\}$.
- $\max_{(m,n) \in \{0,1\}^2}$ computes the maximum value within that $2 \times 2$ window.

As a result of this operation, each $2 \times 2$ block of the input is reduced to a single output value. For example, given that the output from the convolutional layer $A_{\text{cl}}$ has a shape of $24 \times 24 \times 8$, applying this max pooling operation reduces the spatial dimensions by half, yielding an output:

$$A_{\text{mp}} \in \mathbb{R}^{12 \times 12 \times 8}$$

This reduction in spatial size helps decrease computational load in subsequent layers and makes the model more robust to minor shifts in the input.

---

### Flattening

After the max pooling layer, the feature map is flattened into a one-dimensional vector to be passed to the fully connected layer. The flattening operation transforms the $12 \times 12 \times 8$ output tensor into a 1D vector of size 1152, represented as:

$$A' \in \mathbb{R}^{1152}$$

---

### Fully Connected Layer

The flattened output is then passed through a fully connected (dense) layer. This layer computes a linear transformation of the input vector $A'$, followed by a bias term, resulting in an intermediate vector $Z_{\text{fc}}$:

$$Z_{\text{fc}} = W_{\text{fc}} \cdot  A' + b_{\text{fc}}$$

where $W_{\text{fc}}$ is the weight matrix, and $b_{\text{fc}}$ is the bias vector for the fully connected layer. The output of this layer has a size of $10$, representing the 10 class scores, one for each digit from 0 to 9. Thus, the output of the fully connected layer is:

$$Z_{\text{fc}} \in \mathbb{R}^{10}$$

---

### Softmax Output

Finally, the output $Z_{\text{fc}}$ is passed through a softmax activation function to obtain the class probabilities. The softmax function for the $i$-th class is defined as:

$$\hat{y}_ i = \frac{e^{Z_{\text{fc},i}}}{\sum_{j=1}^{10} e^{Z_{\text{fc},j}}}$$

where $\hat{y}_i$ is the predicted probability for class $i$, and the denominator ensures that the sum of all predicted probabilities equals 1.

---

### Complete Transformation from Input to Output

The transformation from the input image $X$ to the predicted class probabilities $\hat{y}$ is expressed as:

$$\hat{y}_ i = \frac{e^{Z_{\text{fc},i}}}{\sum_{j=1}^{10} e^{Z_{\text{fc},j}}}$$

Where:

$$
Z_{\text{fc}} = W_{\text{fc}} 
    \left(
        \text{Flatten} 
        \left(
            \text{MaxPool}
            \left( 
                \text{ReLU}
                \left(
                    X * W_{\text{cl}} + b_{\text{cl}} 
                \right) 
            \right) 
        \right) 
    \right) + b_{\text{fc}}
$$

In this final equation, the input image $X$ undergoes the following transformations:

1. **Convolution**: Apply filters $W_{\text{cl}}$ and bias $b_{\text{cl}}$ to the input $X$.
   
   $$Z_{\text{cl}} = X * W_{\text{cl}} + b_{\text{cl}}$$

2. **ReLU Activation**: Apply the ReLU activation element-wise.
   
   $$A_{\text{cl}} = \text{ReLU}(Z_{\text{cl}})$$

3. **Max Pooling**: Perform max pooling with a $2 \times 2$ window and stride 2.
   
   $$A_{\text{mp}}(i,j,k) = \max_{(m,n) \in \{0,1\}^2} A_{\text{cl}}(2i + m,\, 2j + n,\, k)$$

4. **Flattening**: Flatten the pooled output into a 1D vector.

    
   $$A' = \text{Flatten}(A_{\text{mp}})$$

5. **Fully Connected Layer**: Perform a linear transformation in the fully connected layer.
   
   $$Z_{\text{fc}} = W_{\text{fc}} A' + b_{\text{fc}}$$

6. **Softmax Activation**: Apply softmax to obtain the final class probabilities, $\hat{y}$.
   

This is the full forward pass from the input image $X$ to the output probabilities $\hat{y}$.

---

## Goals

- Demonstrate that a complete forward pass of a CNN can be executed in RISC-V assembly
- Utilize RVV for efficient vectorized computation of convolutions and dense operations

---

## Requirements

- RISC-V Vector Extension simulator or hardware
- TensorFlow (for training and parameter export)
- Python 3.7+
