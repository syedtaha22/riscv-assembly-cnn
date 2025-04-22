#include <iostream>


int main() {
    int H = 28; // Input height
    int W = 28; // Input width
    int C = 1;  // Input channels
    int K = 8;  // Number of filters
    int F = 5;  // Filter size
    int stride = 1; // Stride

    int outH = (H - F) / stride + 1;
    int outW = (W - F) / stride + 1;

    std::cout << "Output shape: " << outH << "x" << outW << "x" << K << std::endl;
}