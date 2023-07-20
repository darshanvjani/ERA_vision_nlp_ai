
# Session 8 - Batch Normalization & Regularization

![Instruction for Assignment](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Batch%20Normalization%20and%20Regularization/Images/assignment_screenshot.PNG?raw=true)

## My Solution

The solution for this assignment is modularise into 4 different files.
- utils.py
- model.py
- train.py
- *notebook.ipynb*

[Colab Notebook](#https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Batch%20Normalization%20and%20Regularization/Session_8.ipynb)


#### Model Architecture

The same architecture is used to build 3 different model with different normalization strategy
    
    1. Batch Normalization

    2. Group Normalization  
    
    3. Layer Normalization


Model Architecture (BatchNorm)
![BN](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Batch%20Normalization%20and%20Regularization/Images/model_bn.PNG?raw=true)

Model Architecture (GroupNorm)
![GN](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Batch%20Normalization%20and%20Regularization/Images/model_gn.PNG?raw=true)

Model Architecture (LayerNorm)
![LN](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Batch%20Normalization%20and%20Regularization/Images/model_ln.PNG?raw=true)


#### Receptive Field and Jump parameter Calculation


| Block | Layer | Channel Size Change | Receptive Field Change | Jump Parameter Change |
| :-------- | :-------- | :------- | :------------------------- | :------------------------- |
| `convblock1` | Conv2d_1 (3x3) | 32 -> 32 | 1 -> 3 | 1 -> 1 |
| - | Conv2d_2 (3x3) | 32 -> 32 | 3 -> 5 | 1 -> 1 |
| `transblock1` | Conv2d (1x1) | 32 -> 32 | 5 -> 5 | 1 -> 1 |
| - | MaxPool2d | 32 -> 16 | 5 -> 6 | 1 -> 2 |
| `convblock2` | Conv2d_1 (3x3) | 16 -> 16 | 6 -> 10 | 2 -> 2 |
| - | Conv2d_2 (3x3) | 16 -> 16 | 10 -> 14 | 2 -> 2 |
| - | Conv2d_3 (3x3) | 16 -> 16 | 14 -> 18 | 2 -> 2 |
| `transblock2` | Conv2d (1x1) | 16 -> 16 | 18 -> 18 | 2 -> 2 |
| - | MaxPool2d | 16 -> 8 | 18 -> 20 | 2 -> 4 |
| `convblock3` | Conv2d_1 (3x3) | 8 -> 8 | 20 -> 24 | 4 -> 4 |
| - | Conv2d_2 (3x3) | 8 -> 8 | 24 -> 28 | 4 -> 4 |
| - | Conv2d_3 (3x3) | 8 -> 8 | 28 -> 32 | 4 -> 4 |
| `gap` | AdaptiveAvgPool2d | 8 -> 1 | 32 -> 60 | 4 -> 4 |
| `fc1` | Conv2d (1x1) | 1 -> 1 | 60 -> 60 | 4 -> 4 |


## Results

#### Accuracy Logs with different normalization strategies

![accuracy](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Batch%20Normalization%20and%20Regularization/Images/accuracy.png?raw=true)

#### Loss Logs with different normalization strategies

![loss](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Batch%20Normalization%20and%20Regularization/Images/loss.png?raw=true)


## Things Used for Building the Architecture
- Convolution Blocks (3x3)
- Relu
- Maxpool (used in transition layer for reducing the size and increasing the Receptive field)
- Convolution Blocks (3x3) (used in transition layer for decreasing the channel size to aggregrate channels)
- BatchNorm, Group Norm & Layer Norm (Architecture Gradient Stability and better convergence)
- DropOut (Regularization - To avoid overfitting)
- AvgPool (To keep the model Fully convolutional and building better architecture)

## Learning from this Session 
- Image Normalizing
- Why redistribution of data?
- Normalization is not Equalization
- How to normalize?
- Loss & Weights with/without normalization
- **Batch Normalization**
- Batch Normalization Mathematics
- What are SOTAs using today?
- **Group, Instance, and Layers Normalization**
- Regularization
- L1 & L2 Regularization


## Authors

- [@darshanvjani](https://github.com/darshanvjani)

