
# Session 10 - Residual Connections and One Cycle Policy

![Instruction for Assignment](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/CAMs,%20LRs,%20and%20Optimizers/Images/assignment.png?raw=true)

The Assignment aims to make us familior to Class Activation Maps (CAM) and how it works.

#### GradCam (Class Activation Map)
![CAM](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/CAMs,%20LRs,%20and%20Optimizers/Images/cam.jpg?raw=true)


## My Solution

My Submittion
[Colab Notebook](#https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/CAMs%2C%20LRs%2C%20and%20Optimizers/Session_11.ipynb)

#### Model Architecture

The Model used for this work is **Resnet 18**
![Instruction for Assignment](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/CAMs,%20LRs,%20and%20Optimizers/Images/resnet_arc.png?raw=true)


## Results

Achieved **87.01% test accuracy under 20 epochs** using this model.

Accuracy Logs

![accuracy](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/CAMs,%20LRs,%20and%20Optimizers/Images/accuracy%20logs.png?raw=true)

Loss Logs

![loss](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/CAMs,%20LRs,%20and%20Optimizers/Images/loss%20logs.png?raw=true)

Confusion Matrix

![matrix](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/CAMs,%20LRs,%20and%20Optimizers/Images/confusion%20matrix.png?raw=true)


## Misclassified Images and Class Activation Maps

Incorrect Prediction

![incorrect_pred](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/CAMs,%20LRs,%20and%20Optimizers/Images/misclassified%20images.png?raw=true)

GradCAM (Class Activation Map)

![cam](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/CAMs,%20LRs,%20and%20Optimizers/Images/cam.png?raw=true)


## Things Used for Building the Architecture
- Resnet structure is replicated

## Learning from this Session 
- Class activation maps
- GradCAM
- Learning Rates
- Weight updates
- Constant vs Adaptive Learning Rates
- SGD
    - Gradient Perturbation
    - Momentum & Nesterov Momentum
- RMSProp
- Adam
- Best Optimizer
- LRs
    - One Cycle Policy
    - Reduce LR on Plateau
- What kind of minima do we want?

## Authors

- [@darshanvjani](https://github.com/darshanvjani)

