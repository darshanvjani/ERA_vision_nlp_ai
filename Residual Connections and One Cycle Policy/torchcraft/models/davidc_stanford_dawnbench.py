# PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
# Layer1 -
# X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
# R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
## Add(X, R1)
# Layer 2 -
# Conv 3x3 [256k]
# MaxPooling2D
# BN
# ReLU
# Layer 3 -
# X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
# R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
# Add(X, R2)
# MaxPooling with Kernel Size 4
# FC Layer 
# SoftMax

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        
        self.prelayer = nn.Sequential(
            
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding = 1, stride = 1, bias = False), #32 -> 32 | 1 -> 3 | 1 -> 1
            nn.BatchNorm2d(64),
            nn.ReLU()
    
            )
            
        self.layer1 = nn.Sequential(
            
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1, stride = 1, bias = False), #32 -> 32 -> 16 | 1 -> 3 -> 4 | 1 -> 1 -> 2
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(128),
            nn.ReLU()
    
            )
            
        self.r1 = nn.Sequential(
            
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1, stride = 1, bias = False), #32 -> 32 | 3 -> 5 | 1 -> 1
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1, stride = 2, bias = False), #32 -> 16 | 5 -> 7 | 1 -> 2
            nn.BatchNorm2d(128),
            nn.ReLU()
    
            )
            
        self.layer2 = nn.Sequential(
            
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1, stride = 1, bias = False), #16 -> 16 -> 8 | 4 -> 8 -> 10 +++ 7 -> 11 -> 13| 2 -> 2 -> 4
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(256),
            nn.ReLU()
    
            )
        
        self.layer3 = nn.Sequential(
            
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, padding = 1, stride = 1, bias = False), #8 -> 8 -> 4 | 10 -> 18 -> 22 +++ 13 -> 21 -> 25| 4 -> 4 -> 8
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(512),
            nn.ReLU()
    
            )
            
            
        self.r2 = nn.Sequential(
            
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, padding = 1, stride = 1, bias = False), #8 -> 8 | 10 -> 18 +++ 13 -> 21 | 4 -> 4
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1, stride = 2, bias = False), #8 -> 4 | 18 -> 26 +++ 21 -> 29| 4 -> 8
            nn.BatchNorm2d(512),
            nn.ReLU()
    
            )
            
        self.output = nn.Sequential(
            
            nn.MaxPool2d(4,4),  #4 -> 1 | 26 -> 50 +++ 29 -> 53 | 8 -> 16
            nn.Conv2d(in_channels = 512, out_channels = 10, kernel_size = 1, bias = False)
            
            )
            
        self.softmax = nn.Softmax(dim = 1)
            
    def forward(self, x):
        
        x = self.prelayer(x)
        
        x = self.layer1(x) + self.r1(x)
        
        x = self.layer2(x)
                    
        x = self.layer3(x) + self.r2(x)
        
        x = self.output(x)
        x = x.view(-1, 10)
        
        x = self.softmax(x)
        
        return x         
        