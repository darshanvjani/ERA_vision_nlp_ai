# Session 12

## Introduction
This assignment is focused towards getting acquainted with Pytorch Lightning, Gradio and HuggingFace🤗 Spaces.

## Target
- Port training code to Pytorch Lightning
- Use Gradio to create a simple app visualizing the classification output, GradCAM and the misclassified images.
- Host the app using HuggingFace Spaces.

[Link to HuggingFace Space](https://huggingface.co/spaces/darshanjani/demo0)

## Architecture
```----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,792
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
           Dropout-4           [-1, 64, 32, 32]               0
            Conv2d-5          [-1, 128, 32, 32]          73,856
         MaxPool2d-6          [-1, 128, 16, 16]               0
       BatchNorm2d-7          [-1, 128, 16, 16]             256
              ReLU-8          [-1, 128, 16, 16]               0
           Dropout-9          [-1, 128, 16, 16]               0
           Conv2d-10          [-1, 128, 16, 16]         147,584
      BatchNorm2d-11          [-1, 128, 16, 16]             256
             ReLU-12          [-1, 128, 16, 16]               0
          Dropout-13          [-1, 128, 16, 16]               0
           Conv2d-14          [-1, 128, 16, 16]         147,584
      BatchNorm2d-15          [-1, 128, 16, 16]             256
             ReLU-16          [-1, 128, 16, 16]               0
          Dropout-17          [-1, 128, 16, 16]               0
           Conv2d-18          [-1, 256, 16, 16]         295,168
        MaxPool2d-19            [-1, 256, 8, 8]               0
      BatchNorm2d-20            [-1, 256, 8, 8]             512
             ReLU-21            [-1, 256, 8, 8]               0
          Dropout-22            [-1, 256, 8, 8]               0
           Conv2d-23            [-1, 512, 8, 8]       1,180,160
        MaxPool2d-24            [-1, 512, 4, 4]               0
      BatchNorm2d-25            [-1, 512, 4, 4]           1,024
             ReLU-26            [-1, 512, 4, 4]               0
          Dropout-27            [-1, 512, 4, 4]               0
           Conv2d-28            [-1, 512, 4, 4]       2,359,808
      BatchNorm2d-29            [-1, 512, 4, 4]           1,024
             ReLU-30            [-1, 512, 4, 4]               0
          Dropout-31            [-1, 512, 4, 4]               0
           Conv2d-32            [-1, 512, 4, 4]       2,359,808
      BatchNorm2d-33            [-1, 512, 4, 4]           1,024
             ReLU-34            [-1, 512, 4, 4]               0
          Dropout-35            [-1, 512, 4, 4]               0
        MaxPool2d-36            [-1, 512, 1, 1]               0
           Linear-37                   [-1, 10]           5,130
================================================================
Total params: 6,575,370
Trainable params: 6,575,370
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 8.00
Params size (MB): 25.08
Estimated Total Size (MB): 33.10
----------------------------------------------------------------
```

### Metrics
| Train Acc | Test Acc | Train Loss | Test Loss |
|-----------|----------|------------|-----------|
| 96.34     | 92.98    | 0.09       | 0.22      |

### Performance Curve
![training and test](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Pytorch%20Lightning%20&%20Spaces/Images/training_curves.png?raw=true)

### Confusion Matrix
![confusion matrix](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Pytorch%20Lightning%20&%20Spaces/Images/conf.png?raw=true)

### Data Exploration
```
train_transforms = A.Compose([
    A.PadIfNeeded(min_height=48, min_width=48, always_apply=True, border_mode=0),
    A.RandomCrop(height=32, width=32, always_apply=True),
    A.HorizontalFlip(p=0.5),
    # A.PadIfNeeded(min_height=64, min_width=64, always_apply=True, border_mode=0),
    A.CoarseDropout(
        p=0.2,
        max_holes=1,
        max_height=8,
        max_width=8,
        min_holes=1,
        min_height=8,
        min_width=8,
        fill_value=(0.4914, 0.4822, 0.4465),
        mask_fill_value=None,
    ),
    # A.CenterCrop(height=32, width=32, always_apply=True),
    A.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ToTensorV2()
])

test_transforms = A.Compose([
    A.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ToTensorV2()
])
```
The data exploration section includes code for data transformations using the Albumentations library. The transformations used are RandomCrop, HorizontalFlip, and CoarseDropout.

### LR Finder
![lr](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Pytorch%20Lightning%20&%20Spaces/Images/lr.png?raw=true)
The LR Finder section includes code for finding the optimal learning rate using the torch_lr_finder library. The suggested learning rate is 2.56E-03.

### Misclassified Images
![miss_images](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/Pytorch%20Lightning%20&%20Spaces/Images/mis_images.png?raw=true)
The total number of incorrect predictions is 701. The misclassified images in all three models have classes very close to each other as misclassified. These misclassified images would be hard for a human to classify correctly too!

### Training Log
The training log section includes the training and validation loss and accuracy for each epoch. The training goes up to 23 epochs.
```
Sanity Checking: 0it [00:00, ?it/s]
Epoch: 0, Val Loss: 2.303018569946289, Val Accuracy: 0.0927734375
Training: 0it [00:00, ?it/s]
Validation: 0it [00:00, ?it/s]
Epoch: 0, Val Loss: 1.6438324451446533, Val Accuracy: 0.4316999912261963

Epoch: 0, Train Loss: 1.818788766860962, Train Accuracy: 0.3441599905490875
Validation: 0it [00:00, ?it/s]
Epoch: 1, Val Loss: 1.158406376838684, Val Accuracy: 0.5909000039100647

Epoch: 1, Train Loss: 1.208844780921936, Train Accuracy: 0.5672199726104736
Validation: 0it [00:00, ?it/s]
Epoch: 2, Val Loss: 0.8067277073860168, Val Accuracy: 0.7204999923706055

Epoch: 2, Train Loss: 0.8991088271141052, Train Accuracy: 0.6844000220298767
Validation: 0it [00:00, ?it/s]
Epoch: 3, Val Loss: 0.7408087849617004, Val Accuracy: 0.7458000183105469

Epoch: 3, Train Loss: 0.727645993232727, Train Accuracy: 0.7465599775314331
Validation: 0it [00:00, ?it/s]
Epoch: 4, Val Loss: 0.6198990941047668, Val Accuracy: 0.7919999957084656

Epoch: 4, Train Loss: 0.6317057609558105, Train Accuracy: 0.7823200225830078
Validation: 0it [00:00, ?it/s]
Epoch: 5, Val Loss: 0.6251322627067566, Val Accuracy: 0.7928000092506409

Epoch: 5, Train Loss: 0.5662901401519775, Train Accuracy: 0.8035200238227844
Validation: 0it [00:00, ?it/s]
Epoch: 6, Val Loss: 0.458161860704422, Val Accuracy: 0.8432999849319458

Epoch: 6, Train Loss: 0.49268826842308044, Train Accuracy: 0.8307600021362305
Validation: 0it [00:00, ?it/s]
Epoch: 7, Val Loss: 0.44248712062835693, Val Accuracy: 0.8492000102996826

Epoch: 7, Train Loss: 0.433277428150177, Train Accuracy: 0.8492599725723267
Validation: 0it [00:00, ?it/s]
Epoch: 8, Val Loss: 0.4037456214427948, Val Accuracy: 0.8632000088691711

Epoch: 8, Train Loss: 0.39565661549568176, Train Accuracy: 0.8627399802207947
Validation: 0it [00:00, ?it/s]
Epoch: 9, Val Loss: 0.4889300763607025, Val Accuracy: 0.8299999833106995

Epoch: 9, Train Loss: 0.357261598110199, Train Accuracy: 0.875220000743866
Validation: 0it [00:00, ?it/s]
Epoch: 10, Val Loss: 0.38768595457077026, Val Accuracy: 0.8680999875068665

Epoch: 10, Train Loss: 0.33682936429977417, Train Accuracy: 0.8836600184440613
Validation: 0it [00:00, ?it/s]
Epoch: 11, Val Loss: 0.36559027433395386, Val Accuracy: 0.882099986076355

Epoch: 11, Train Loss: 0.30156561732292175, Train Accuracy: 0.8941199779510498
Validation: 0it [00:00, ?it/s]
Epoch: 12, Val Loss: 0.3478858470916748, Val Accuracy: 0.8866000175476074

Epoch: 12, Train Loss: 0.279257595539093, Train Accuracy: 0.9043200016021729
Validation: 0it [00:00, ?it/s]
Epoch: 13, Val Loss: 0.3850046992301941, Val Accuracy: 0.8755000233650208

Epoch: 13, Train Loss: 0.2567802667617798, Train Accuracy: 0.9107800126075745
Validation: 0it [00:00, ?it/s]
Epoch: 14, Val Loss: 0.3424720764160156, Val Accuracy: 0.8895999789237976

Epoch: 14, Train Loss: 0.23939351737499237, Train Accuracy: 0.916700005531311
Validation: 0it [00:00, ?it/s]
Epoch: 15, Val Loss: 0.3428153693675995, Val Accuracy: 0.8913999795913696

Epoch: 15, Train Loss: 0.22301723062992096, Train Accuracy: 0.9211000204086304
Validation: 0it [00:00, ?it/s]
Epoch: 16, Val Loss: 0.276138573884964, Val Accuracy: 0.9060999751091003

Epoch: 16, Train Loss: 0.20357853174209595, Train Accuracy: 0.9287999868392944
Validation: 0it [00:00, ?it/s]
Epoch: 17, Val Loss: 0.3122042417526245, Val Accuracy: 0.9003000259399414

Epoch: 17, Train Loss: 0.18075187504291534, Train Accuracy: 0.9375799894332886
Validation: 0it [00:00, ?it/s]
Epoch: 18, Val Loss: 0.292987197637558, Val Accuracy: 0.910099983215332

Epoch: 18, Train Loss: 0.16429181396961212, Train Accuracy: 0.942080020904541
Validation: 0it [00:00, ?it/s]
Epoch: 19, Val Loss: 0.26266226172447205, Val Accuracy: 0.9182000160217285

Epoch: 19, Train Loss: 0.1525600552558899, Train Accuracy: 0.9477999806404114
Validation: 0it [00:00, ?it/s]
Epoch: 20, Val Loss: 0.2535788416862488, Val Accuracy: 0.917900025844574

Epoch: 20, Train Loss: 0.13234469294548035, Train Accuracy: 0.9539399743080139
Validation: 0it [00:00, ?it/s]
Epoch: 21, Val Loss: 0.2403881847858429, Val Accuracy: 0.9243999719619751

Epoch: 21, Train Loss: 0.11851569265127182, Train Accuracy: 0.9589999914169312
Validation: 0it [00:00, ?it/s]
Epoch: 22, Val Loss: 0.23120273649692535, Val Accuracy: 0.9265000224113464

Epoch: 22, Train Loss: 0.1018628403544426, Train Accuracy: 0.9656000137329102
Validation: 0it [00:00, ?it/s]
Epoch: 23, Val Loss: 0.2248384803533554, Val Accuracy: 0.9298999905586243

Epoch: 23, Train Loss: 0.09355912357568741, Train Accuracy: 0.9693400263786316
```
