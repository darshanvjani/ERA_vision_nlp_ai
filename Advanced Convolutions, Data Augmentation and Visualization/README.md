
# Session 9 - Advanced Convolutions, Data Augmentation and Visualization

![Instruction for Assignment](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Advanced%20Convolutions,%20Data%20Augmentation%20and%20Visualization/Images/assignment.png?raw=true)


## My Solution

Model With Strided Convolution in the Transition Block :  
[Model 1](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Advanced%20Convolutions%2C%20Data%20Augmentation%20and%20Visualization/Session_9_using_model.ipynb)

Model With Strided - Dilated Convolution in the Transition Block:  
[Model 2](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Advanced%20Convolutions%2C%20Data%20Augmentation%20and%20Visualization/Session_9.ipynb)

#### Model Architecture

Model 1 (Trans Block - Strided Conv)
![Instruction for Assignment](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Advanced%20Convolutions,%20Data%20Augmentation%20and%20Visualization/Images/model.png?raw=true)

Model 2 (Trans Block - Strided and Dilated Conv)
![Instruction for Assignment](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Advanced%20Convolutions,%20Data%20Augmentation%20and%20Visualization/Images/model_1.png?raw=true)

#### Receptive Field and Jump parameter Calculation


For Model 1 (Strided Conv in the transition Block)
| Block | Layer | Channel Size Change | Receptive Field Change | Jump Parameter Change |
| :-------- | :-------- | :------- | :------------------------- | :------------------------- |
| `convblock1` | Conv2d_1 (3x3) | 32 -> 32 | 1 -> 3 | 1 -> 1 |
| - | Conv2d_2 (3x3 - dilation 2) | 32 -> 32 | 3 -> 7 | 1 -> 1 |
| - | Conv2d_3 (3x3 - stride 2) | 32 -> 16 | 7 -> 9 | 1 -> 2 |
| `convblock2` | Conv2d_4 (3x3) | 16 -> 16 | 9 -> 13 | 2 -> 2 |
| - | Conv2d_5 (3x3) | 16 -> 16 | 13 -> 17 | 2 -> 2 |
| - | Conv2d_6 (3x3 - stride 2) | 16 -> 8 | 17 -> 21 | 2 -> 4 |
| `convblock3` | Conv2d_7 (3x3) | 8 -> 8 | 21 -> 29 | 4 -> 4 |
| - | Conv2d_8 (3x3) | 8 -> 8 | 29 -> 37 | 4 -> 4 |
| - | Conv2d_9 (3x3 - stride 2) | 8 -> 4 | 37 -> 45 | 4 -> 8 |
| `gap` | AdaptiveAvgPool2d | 4 -> 1 | 45 -> 69 | 8 -> 8 |
| `fc1` | Conv2d_10 (1x1) | 1 -> 1 | 69 -> 69 | 8 -> 8 |

For Model 2 (Dilated Strided Conv in the transition Block)
| Block | Layer | Channel Size Change | Receptive Field Change | Jump Parameter Change |
| :-------- | :-------- | :------- | :------------------------- | :------------------------- |
| `convblock1` | Conv2d_1 (3x3) | 3 -> 32 | 1 -> 3 | 1 -> 1 |
| - | Conv2d_2 (3x3) | 32 -> 32 | 3 -> 5 | 1 -> 1 |
| - | Conv2d_3 (3x3 - dilation 2, stride 2) | 32 -> 16 | 5 -> 9 | 1 -> 2 |
| `convblock2` | Conv2d_4 (3x3) | 16 -> 16 | 9 -> 13 | 2 -> 2 |
| - | Conv2d_5 (3x3) | 16 -> 16 | 13 -> 17 | 2 -> 2 |
| - | Conv2d_6 (3x3 - dilation 2, stride 2) | 16 -> 8 | 17 -> 25 | 2 -> 4 |
| `convblock3` | Conv2d_7 (3x3) | 8 -> 8 | 25 -> 33 | 4 -> 4 |
| - | Conv2d_8 (3x3) | 8 -> 8 | 33 -> 39 | 4 -> 4 |
| - | Conv2d_9 (3x3 - dilation 2, stride 2) | 8 -> 4 | 39 -> 55 | 4 -> 8 |
| `gap` | AdaptiveAvgPool2d | 4 -> 1 | 55 -> 79 | 8 -> 8 |
| `fc1` | Conv2d_10 (1x1) | 1 -> 1 | 79 -> 79 | 8 -> 8 |




## Things Used for Building the Architecture
- Convolution Blocks (3x3)
- Relu
- **Conv Layers (with Strided = 2) in the transition layers for model 1.**
- **Dilated Conv ( with Strided = 2) in the trans layer for model 2.**
- **Depthwise seperable kernals**
- Normal Dilated Conv to capture bigger context
- BatchNorm (Architecture Gradient Stability and better convergence)
- DropOut (Regularization - To avoid overfitting)
- AvgPool (To keep the model Fully convolutional and building better architecture)

## Learning from this Session 
- Normal Convolutions
- Pointwise Convolutions
- Concept of Channels
- Receptive Fields
- Strides & Checkerboard Issue
- How do we get 5x5 RF with one 3x3 Kernel? or *Atrous* or **Dilated Convolutions**
- Dence Problems in Computer Vision
- How do we increase channel size after Convolution or *Transpose Convolution* or *Deconvolution* or *Fractionally Strided Convolution*
- **Pixel Shuffle**
- **Depthwise Separable Convolution**
- **Spatially Separable Convolutions**
- **Grouped Convolutions**
- **Data Augmentation** Any why we should fall in love with it!
    - PMDA (Poor Mans Data Augmentation)
        - Scale; Translation; Rotation; Blurring; Image Mirroring; Color Shifting / Whitening.
    - **MMDA (Middle Mans Data Augmentation)** 
        - *Elastic Distortions*
        - *CutOut*
        - *MixUp*
        - *RICAP*
        - (and more) ...
    - RMDA (Rich Mans Data Augmentation)**
        - NAS 
        - Population Based Augmentation


[A Comprehensive Survey of Image Augmentation 2022](https://arxiv.org/pdf/2205.01491.pdf) 
## Authors

- [@darshanvjani](https://github.com/darshanvjani)

