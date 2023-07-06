
# Session 4 - Building the First Neural Network Assignment

![Instruction for Assignment](https://raw.githubusercontent.com/darshanvjani/ERA_vision_nlp_ai/main/Building%20the%20First%20Neural%20Network/Images/assignment.PNG)
Task is to take the [Faulty Colab Notebook](https://colab.research.google.com/github/darshanvjani/ERA_vision_nlp_ai/blob/main/Building%20the%20First%20Neural%20Network/Faulty_Notebook.ipynb) and rewrite it such that all the Errors (Syntexually and Understanding wise) are resolved.

## My Solution

[Corrected Notebook](https://colab.research.google.com/github/darshanvjani/ERA_vision_nlp_ai/blob/main/Building%20the%20First%20Neural%20Network/Corrected_Notebook.ipynb)

## Issues Encountered and Resolution
- Code Block 3 > Different Normalization value between train and test || Made the Normalization values same for both train and test
- Code Block 4 > Error in the dataset download || Resolved it by seperating train and test, make sure that there is no DATA LEAKAGE
- Code Block 5 > Incorrect Argument for dataloader (train_data in both train and test dataloader) || Corrected the dataloader
- Code Block 5 > Dataloader **kwargs (shuffle == False) || Changed to True
- Code Block 7(**BIGGEST ISSUE**) > Faulty in the neural network architecture, incorrect transition from 2D (Convolution Layer) to 1D (Fully Connected Layer) || Correct the Structure
- Code Block 10 > lr==10.01 and test(model, device, train_loader, criterion) || changed lr == 0.01 and test(model, device, test_loader, criterion)

## Learning from this Session - Session 4 - Building the First Neural Network Assignment
- Non-linearity of Real-world Data
- Data Decomposition into Features
- Feed-forward Networks
- Role of Neurons
- Activation Functions
- Understanding Kernels
- Importance of Receptive Field Size
- Extraction Order
- Convolution with Strides
- Multi-Channel Convolution


## Authors

- [@darshanvjani](https://github.com/darshanvjani)

