
# Session 13 - YoloV3 from Scratch

The assignment is about training YoloV3 from Scratch to achieve 

- *Class accuracy is more than 75%*
- *No Obj accuracy of more than 95%*
- *Object Accuracy of more than 70%*

the job is to replicate the original YoloV3 paper whil implementing 4 image mosaic augmentation.

## YoloV3 Model Architecture
Architecture has been replicated form the original [YoloV3](https://arxiv.org/pdf/1804.02767.pdf) paper.

![Architecture](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/YoloV3%20from%20Scratch/Images/architecture.png?raw=true)

## Lr Finder 
We have used the suggest Lr of 0.0031622 using the lr Finder

![LR_finder](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/YoloV3%20from%20Scratch/Images/lr_finder.png?raw=true)

## Training

Trained the model on Tesla P200. 
Took around 10hr for the model to train till 40 epochs.

#### Logs

```
Epoch: 0, Loss: 10.928609848022461
Epoch: 1, Loss: 8.026472091674805
Epoch: 2, Loss: 7.41795015335083

Class accuracy is: 41.016903%
No obj accuracy is: 97.683266%
Obj accuracy is: 31.571072%
```
![log1](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/YoloV3%20from%20Scratch/Images/log1.png?raw=true)

```
Epoch: 3, Loss: 6.867676734924316
Epoch: 4, Loss: 6.553554534912109
Epoch: 5, Loss: 6.341279029846191
Epoch: 6, Loss: 6.1299309730529785

Class accuracy is: 44.801571%
No obj accuracy is: 96.033981%
Obj accuracy is: 50.657391%

Epoch: 7, Loss: 5.986669540405273
Epoch: 8, Loss: 5.882408142089844

Class accuracy is: 53.837627%
No obj accuracy is: 96.736259%
Obj accuracy is: 47.531174%
```
![log2](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/YoloV3%20from%20Scratch/Images/log2.png?raw=true)

```
Epoch: 9, Loss: 5.698788166046143
Epoch: 10, Loss: 5.607735633850098

Class accuracy is: 61.482407%
No obj accuracy is: 95.958916%
Obj accuracy is: 58.210033%

Epoch: 11, Loss: 5.4896039962768555
Epoch: 12, Loss: 5.371456146240234
Epoch: 13, Loss: 5.247055530548096
```
![log3](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/YoloV3%20from%20Scratch/Images/log3.png?raw=true)

```
Epoch: 14, Loss: 5.145310878753662

Class accuracy is: 58.730984%
No obj accuracy is: 96.177765%
Obj accuracy is: 64.808258%

Epoch: 15, Loss: 5.004179000854492
Epoch: 16, Loss: 4.854735851287842
Epoch: 17, Loss: 4.776919364929199
Epoch: 18, Loss: 4.684315204620361

Class accuracy is: 68.295929%
No obj accuracy is: 97.073303%
Obj accuracy is: 59.390415%
```
![log4](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/YoloV3%20from%20Scratch/Images/log4.png?raw=true)
```
Epoch: 15, Loss: 5.004179000854492
Epoch: 16, Loss: 4.854735851287842
Epoch: 17, Loss: 4.776919364929199
Epoch: 18, Loss: 4.684315204620361

Class accuracy is: 70.834030%
No obj accuracy is: 96.883621%
Obj accuracy is: 68.185089%

Epoch: 19, Loss: 4.524842739105225
Epoch: 20, Loss: 4.402139186859131
Epoch: 21, Loss: 4.315126419067383
Epoch: 22, Loss: 4.205116271972656

Class accuracy is: 77.700195%
No obj accuracy is: 96.674088%
Obj accuracy is: 71.964531%

Epoch: 23, Loss: 4.103285312652588
Epoch: 24, Loss: 4.014401435852051
Epoch: 25, Loss: 3.9082982540130615
Epoch: 26, Loss: 3.8202638626098633

Class accuracy is: 78.905518%
No obj accuracy is: 97.232094%
Obj accuracy is: 73.280685%

Epoch: 27, Loss: 3.7256128787994385
Epoch: 28, Loss: 3.6390113830566406

```
![log5](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/YoloV3%20from%20Scratch/Images/log5.png?raw=true)

```
Epoch: 29, Loss: 3.52851939201355
Epoch: 30, Loss: 3.4508590698242188

Class accuracy is: 73.068008%
No obj accuracy is: 95.833298%
Obj accuracy is: 77.883049%

Class accuracy is: 81.773346%
No obj accuracy is: 96.864510%
Obj accuracy is: 75.272926%

Epoch: 31, Loss: 3.3701071739196777
Epoch: 32, Loss: 3.290649652481079
Epoch: 33, Loss: 3.2031702995300293

```
![log6](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/YoloV3%20from%20Scratch/Images/log6.png?raw=true)

```
Epoch: 34, Loss: 3.1084444522857666

Class accuracy is: 85.336655%
No obj accuracy is: 97.280716%
Obj accuracy is: 77.104462%

Epoch: 35, Loss: 3.020203113555908
Epoch: 36, Loss: 2.930464744567871
Epoch: 37, Loss: 2.8547286987304688
Epoch: 38, Loss: 2.7558252811431885

```
![log7](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/YoloV3%20from%20Scratch/Images/log7.png?raw=true)

```
Class accuracy is: 81.400658%
No obj accuracy is: 96.601974%
Obj accuracy is: 79.738083%

Class accuracy is: 87.315048%
No obj accuracy is: 97.628883%
Obj accuracy is: 76.982544%

Epoch: 39, Loss: 2.718630075454712

```

**MAP:  0.33122602105140686**
![Final Results](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/YoloV3%20from%20Scratch/Images/log8.png?raw=true)
## Loss Function
[loss.py](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/CAMs,%20LRs,%20and%20Optimizers/Images/resnet_arc.png?raw=true)

![loss.py](https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/YoloV3%20from%20Scratch/Images/loss_function.png?raw=true)

## Things Used for Building the Architecture
- Replicated YoloV3 PAPER

## Learning from this Session 
- Everything starts from CVPR 2016
- They did it to beat SSD
- Recognition vs Detection
- Sliding Window - Obsolete
- Region Proposal [RCNN] (Alternative to YOLO approach)
- Anchor Boxes
- Anchor Boxes Mathematics
- I owe You! (Intersection Over Union)
- The most beautiful loss function! (Yolo Loss Function)
- Yolo V3 (Architecture)
- NMS
- Training & Architectural Details of Yolo V3
- Speeding up the training of Yolo V3

## Authors

- [@darshanvjani](https://github.com/darshanvjani)
