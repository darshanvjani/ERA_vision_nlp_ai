
# Session 10 - Residual Connections and One Cycle Policy

![Instruction for Assignment](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Advanced%20Convolutions,%20Data%20Augmentation%20and%20Visualization/Images/assignment.png?raw=true)


The Assignment is inspired by the [Stanford DAWNBench Competition](https://dawn.cs.stanford.edu/benchmark/) which strive for achiving more then 93% test accuracy on CIFAR10 under as less of computation Cost as possible.

I am recreating [@davidcpage](https://github.com/davidcpage) custom resnet model which use A compination of Learning Stratigies like OneCyclePolicy for convergence as quickly as possible to a **relatively stable local minima**.



## My Solution

My Implementation of Davidcpage Custom Resnet Model Implementation (Stanford DAWN Benchmark Rank 1 Model)
[Model](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Residual%20Connections%20and%20One%20Cycle%20Policy/Session_10.ipynb)

#### Model Architecture

Custom Resnet Model Architecture
![Instruction for Assignment](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Residual%20Connections%20and%20One%20Cycle%20Policy/Images/model_architecture.PNG?raw=true)


## Results

Achieved **90.42% test accuracy under 25 epochs** using this model.

Accuracy Logs

![accuracy](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Residual%20Connections%20and%20One%20Cycle%20Policy/Images/accuracy_log.PNG?raw=true)

Loss Logs

![loss](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Residual%20Connections%20and%20One%20Cycle%20Policy/Images/loss_log.PNG?raw=true)

Confusion Matrix

![matrix](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Residual%20Connections%20and%20One%20Cycle%20Policy/Images/confusion_matrix.PNG?raw=true)


## Misclassified Images and Class Activation Maps

Incorrect Prediction

![incorrect_pred](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Residual%20Connections%20and%20One%20Cycle%20Policy/Images/incorrect_images.PNG?raw=true)

GradCAM (Class Activation Map)

![cam](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Residual%20Connections%20and%20One%20Cycle%20Policy/Images/CAM_incorrect_images.png?raw=true)


## Things Used for Building the Architecture
- Convolution Blocks (3x3)
- Relu
- **Residual Blocks**
- MaxPool
- **Strided Convolution**
- Normal Dilated Conv to capture bigger context
- BatchNorm (Architecture Gradient Stability and better convergence)
- DropOut (Regularization - To avoid overfitting)

## Learning from this Session 
- VGG Architecture
- Many Receptive Fields
- Beyond Image Size Receptive Fields
- Change in "Current Views" As we do down the layers
- Inception
- ResNet V1,V2 & V3
- One Cycle Policy

## Authors

- [@darshanvjani](https://github.com/darshanvjani)

