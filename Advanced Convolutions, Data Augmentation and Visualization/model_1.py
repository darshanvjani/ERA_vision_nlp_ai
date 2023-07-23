# has the architecture to C1C2C3C40 (No MaxPooling, but 3 convolutions, where the last one has a stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
# total RF must be more than 44
# one of the layers must use Depthwise Separable Convolution
# one of the layers must use Dilated Convolution
# use GAP (compulsory):- add FC after GAP to target #of classes (optional)
# achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.




import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, dropout_value = 0.01):
        super(Net, self).__init__()
        
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1, bias = False),  # 3 -> 32 | 1 -> 3 | 1  > 1
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
            
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 2, bias = False), # 32 -> 32 | 3 -> 5 | 1 -> 1
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            
            nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 3, padding = 2, bias = False, dilation = 2, stride = 2) # 32 -> 16 | 5 -> 9 | 1 -> 2
            )
            
            
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1, bias = False),  # 16 -> 16 | 9 -> 13 | 2 -> 2
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1, bias = False, groups=32), # 16 -> 16 | 13 -> 17 | 2 -> 2
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            
            nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, padding = 1, bias = False, dilation = 2, stride = 2) # 16 -> 8 | 17 -> 25 | 2 -> 4
            )
        
        
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1, bias = False),  # 8 -> 8 | 25 -> 33 | 4 -> 4
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            
            nn.Conv2d(in_channels = 64, out_channels = 120, kernel_size = 3, padding = 1, bias = False), # 8 -> 8 | 33 -> 39 | 4 -> 4
            nn.ReLU(),
            nn.BatchNorm2d(120),
            nn.Dropout(dropout_value),
            
            nn.Conv2d(in_channels = 120, out_channels = 64, kernel_size = 3, padding = 1, bias = False, dilation = 2, stride = 2) # 8 -> 4 | 39 -> 55 | 4 -> 8
            )
        
        
        self.convblock4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 4 -> 1 | 55 -> 79 | 8 -> 8
            
            nn.Conv2d(in_channels = 64, out_channels = 10, kernel_size = 1, bias = False)
            )
        
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        
        x = x.view(-1, 10)
        return x
        
#net.load_state_dict(torch.load('/content/96K-model.pth'))






