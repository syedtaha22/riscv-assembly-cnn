## Frontend

### Overview

This component provides an interactive interface that allows users to draw numerical digits on the screen. The drawn image is then processed and passed into a RISC-V assembly pipeline for digit recognition using a trained machine learning model.

### Motivation

Manually encoding image data directly into RISC-V assembly is inefficient and impractical. To streamline this process, this frontend automates the following tasks:

1. Captures user-drawn digits through a GUI.
2. Converts the drawing into an image format compatible with the backend.
3. Writes the processed image into RISC-V assembly code.
4. Compiles and runs the RISC-V assembly, feeding the image into the model.
5. Retrieves the output and displays the prediction on screen.

### Current Functionality

At this stage, the program executes the model using TensorFlow with the input image, simulating the end-to-end prediction pipeline.