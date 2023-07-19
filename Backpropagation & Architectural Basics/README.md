
# Session 6 - Backpropagation & Architectural Basics

![Instruction for Assignment](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Backpropagation%20&%20Architectural%20Basics/Images/Assignment.PNG?raw=true)
Task is to build a convolution architecture for MNIST which is under **20k parameters** and should achieve **99.4% accuracy** under **20 Epochs**.

## My Solution

[Colab Notebook](#https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Backpropagation%20%26%20Architectural%20Basics/Session_6.ipynb)


#### Model Architecture

![Instruction for Assignment](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Backpropagation%20&%20Architectural%20Basics/Images/architecture.PNG?raw=true)

#### Receptive Field and Jump parameter Calculation

| Layer | Channel Size Change   | Receptive Field Change    | Jump Parameter Change |
| :-------- | :------- | :------------------------- | :------------------------- |
| `conv1block` | 28 -> 26 | 1 -> 3 | 1 -> 1 |
| `conv2block` | 26 -> 24 | 3 -> 5 | 1 -> 1 |
| `conv3block` | 24 -> 22 | 5 -> 7 | 1 -> 1 |
| `trans1block` | 22 -> 11 -> 11 | 7 -> 8 -> 8 | 1 -> 2 -> 2 |
| `conv4block` | 11 -> 9 | 8 -> 12 | 2 -> 2 |
| `conv5block` | 9 -> 7 | 12 -> 16 | 2 -> 2 |
| `conv6block` | 7 -> 5 | 16 -> 20 | 2 -> 2 |
| `outputblock` | 5 -> 1 | 20 -> 28 | 2 -> 2 -> 2 |



## Things Used for Building the Architecture
- Convolution Blocks (3x3)
- Relu
- Maxpool (used in transition layer for reducing the size and increasing the Receptive field)
- Convolution Blocks (3x3) (used in transition layer for decreasing the channel size to aggregrate channels)
- BatchNorm (Architecture Gradient Stability and better convergence)
- DropOut (Regularization - To avoid overfitting)
- AvgPool (To keep the model Fully convolutional and building better architecture)

## Learning from this Session 
- Backpropagation
- Fully Connected Layers
- SoftMax
- Architectural Blocks
    - MAX POOLING
    - BATCH NORMALIZATION
    - DROPOUT
    - LEARNING RATE


## Authors

- [@darshanvjani](https://github.com/darshanvjani)

