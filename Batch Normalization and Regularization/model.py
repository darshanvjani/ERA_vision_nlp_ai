
# Make this network:
# C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10
# Keep the parameter count less than 50000
# Try and add one layer to another
# Max Epochs is 20



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, norm, dropout_value = 0.01):
        super(Net, self).__init__()
        
        if norm == "BN":
            self.convblock1 = nn.Sequential(
                
                nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1, bias = False), # 32 -> 32 | 1 -> 3 | 1 -> 1
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.Dropout(dropout_value),
                
                nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 1, bias = False), # 32 -> 32 | 3 -> 5 | 1 -> 1
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.Dropout(dropout_value),
                
                )
            
            self.transblock1 = nn.Sequential(
                
                nn.Conv2d(in_channels = 16, out_channels = 10, kernel_size = 1, bias = False), # 32 -> 32 | 5 -> 5 | 1 -> 1
                nn.ReLU(),
                
                nn.MaxPool2d(2, 2) # 32 -> 16 | 5 -> 6 | 1 -> 2
                
                )
        
            self.convblock2 = nn.Sequential(
                
                nn.Conv2d(in_channels = 10, out_channels = 32, kernel_size = 3, padding = 1, bias = False), # 16 -> 16 | 6 -> 10 | 2 -> 2
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Dropout(dropout_value),
                
                nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1, bias = False), # 16 -> 16 | 10 -> 14 | 2 -> 2
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Dropout(dropout_value),
                
                nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1, bias = False), # 16 -> 16 | 14 -> 18 | 2 -> 2
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Dropout(dropout_value),
                
                )
            
            self.transblock2 = nn.Sequential(
                
                nn.Conv2d(in_channels = 32, out_channels = 20, kernel_size = 1, bias = False), # 16 -> 16 | 18 -> 18 | 2 -> 2
                nn.ReLU(),
                
                nn.MaxPool2d(2, 2), # 16 -> 8 | 18 -> 20 | 2 -> 4
                
                )
            
            self.convblock3 = nn.Sequential(
                
                nn.Conv2d(in_channels = 20, out_channels = 32, kernel_size = 3, padding = 1, bias = False), # 8 -> 8 | 20 -> 24 | 4 -> 4
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Dropout(dropout_value),
                
                nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1, bias = False), # 8 -> 8 | 24 -> 28 | 4 -> 4
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Dropout(dropout_value),
                
                nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1, bias = False), # 8 -> 8 | 28 -> 32 | 4 -> 4
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Dropout(dropout_value),
                
                )
            
            self.gap = nn.Sequential(
                
                nn.AdaptiveAvgPool2d((1,1)) # 8 -> 1 | 32 -> 60(32 + 28) | 4 -> 4
                
                )
            
            self.fc1 = nn.Sequential(
                
                nn.Conv2d(in_channels = 32, out_channels = 10, kernel_size = 1, bias = False)
                
                )
            
        elif norm == "GN":
            self.convblock1 = nn.Sequential(
                
                nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1, bias = False), # 32 -> 32 | 1 -> 3 | 1 -> 1
                nn.ReLU(),
                nn.GroupNorm(4, 16),
                nn.Dropout(dropout_value),
                
                nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 1, bias = False), # 32 -> 32 | 3 -> 5 | 1 -> 1
                nn.ReLU(),
                nn.GroupNorm(4, 16),
                nn.Dropout(dropout_value)
                
                )
            
            self.transblock1 = nn.Sequential(
                
                nn.Conv2d(in_channels = 16, out_channels = 10, kernel_size = 1, bias = False), # 32 -> 32 | 5 -> 5 | 1 -> 1
                nn.ReLU(),
                
                nn.MaxPool2d(2, 2) # 32 -> 16 | 5 -> 6 | 1 -> 2
                
                )
        
            self.convblock2 = nn.Sequential(
                
                nn.Conv2d(in_channels = 10, out_channels = 32, kernel_size = 3, padding = 1, bias = False), # 16 -> 16 | 6 -> 10 | 2 -> 2
                nn.ReLU(),
                nn.GroupNorm(8, 32),
                nn.Dropout(dropout_value),
                
                nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1, bias = False), # 16 -> 16 | 10 -> 14 | 2 -> 2
                nn.ReLU(),
                nn.GroupNorm(8, 32),
                nn.Dropout(dropout_value),
                
                nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1, bias = False), # 16 -> 16 | 14 -> 18 | 2 -> 2
                nn.ReLU(),
                nn.GroupNorm(8, 32),
                nn.Dropout(dropout_value)
                
                )
            
            self.transblock2 = nn.Sequential(
                
                nn.Conv2d(in_channels = 32, out_channels = 20, kernel_size = 1, bias = False), # 16 -> 16 | 18 -> 18 | 2 -> 2
                nn.ReLU(),
                
                nn.MaxPool2d(2, 2) # 16 -> 8 | 18 -> 20 | 2 -> 4
                
                )
            
            self.convblock3 = nn.Sequential(
                
                nn.Conv2d(in_channels = 20, out_channels = 32, kernel_size = 3, padding = 1, bias = False), # 8 -> 8 | 20 -> 24 | 4 -> 4
                nn.ReLU(),
                nn.GroupNorm(8, 32),
                nn.Dropout(dropout_value),
                
                nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1, bias = False), # 8 -> 8 | 24 -> 28 | 4 -> 4
                nn.ReLU(),
                nn.GroupNorm(8, 32),
                nn.Dropout(dropout_value),
                
                nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1, bias = False), # 8 -> 8 | 28 -> 32 | 4 -> 4
                nn.ReLU(),
                nn.GroupNorm(8, 32),
                nn.Dropout(dropout_value)
                
                )
            
            self.gap = nn.Sequential(
                
                nn.AdaptiveAvgPool2d((1, 1)) # 8 -> 1 | 32 -> 60(32 + 28) | 4 -> 4
                
                )
            
            self.fc1 = nn.Sequential(
                
                nn.Conv2d(in_channels = 32, out_channels = 10, kernel_size = 1, bias = False)
                
                )
        
        else:
            self.convblock1 = nn.Sequential(
                
                nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1, bias = False), # 32 -> 32 | 1 -> 3 | 1 -> 1
                nn.ReLU(),
                nn.LayerNorm([16, 32, 32]),
                nn.Dropout(dropout_value),
                
                nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 1, bias = False), # 32 -> 32 | 3 -> 5 | 1 -> 1
                nn.ReLU(),
                nn.LayerNorm([16, 32, 32]),
                nn.Dropout(dropout_value)
                
                )
            
            self.transblock1 = nn.Sequential(
                
                nn.Conv2d(in_channels = 16, out_channels = 10, kernel_size = 1, bias = False), # 32 -> 32 | 5 -> 5 | 1 -> 1
                nn.ReLU(),
                
                nn.MaxPool2d(2, 2) # 32 -> 16 | 5 -> 6 | 1 -> 2
                
                )
        
            self.convblock2 = nn.Sequential(
                
                nn.Conv2d(in_channels = 10, out_channels = 32, kernel_size = 3, padding = 1, bias = False), # 16 -> 16 | 6 -> 10 | 2 -> 2
                nn.ReLU(),
                nn.LayerNorm([32, 16, 16]),
                nn.Dropout(dropout_value),
                
                nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1, bias = False), # 16 -> 16 | 10 -> 14 | 2 -> 2
                nn.ReLU(),
                nn.LayerNorm([32, 16, 16]),
                nn.Dropout(dropout_value),
                
                nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1, bias = False), # 16 -> 16 | 14 -> 18 | 2 -> 2
                nn.ReLU(),
                nn.LayerNorm([32, 16, 16]),
                nn.Dropout(dropout_value)
                
                )
            
            self.transblock2 = nn.Sequential(
                
                nn.Conv2d(in_channels = 32, out_channels = 20, kernel_size = 1, bias = False), # 16 -> 16 | 18 -> 18 | 2 -> 2
                nn.ReLU(),
                
                nn.MaxPool2d(2, 2) # 16 -> 8 | 18 -> 20 | 2 -> 4
                
                )
            
            self.convblock3 = nn.Sequential(
                
                nn.Conv2d(in_channels = 20, out_channels = 32, kernel_size = 3, padding = 1, bias = False), # 8 -> 8 | 20 -> 24 | 4 -> 4
                nn.ReLU(),
                nn.LayerNorm([32, 8, 8]),
                nn.Dropout(dropout_value),
                
                nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1, bias = False), # 8 -> 8 | 24 -> 28 | 4 -> 4
                nn.ReLU(),
                nn.LayerNorm([32, 8, 8]),
                nn.Dropout(dropout_value),
                
                nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1, bias = False), # 8 -> 8 | 28 -> 32 | 4 -> 4
                nn.ReLU(),
                nn.LayerNorm([32, 8, 8]),
                nn.Dropout(dropout_value)
                
                )
            
            self.gap = nn.Sequential(
                
                nn.AdaptiveAvgPool2d((1, 1)) # 8 -> 1 | 32 -> 60(32 + 28) | 4 -> 4
                
                )
            
            self.fc1 = nn.Sequential(
                
                nn.Conv2d(in_channels = 32, out_channels = 10, kernel_size = 1, bias = False)
                
                )

    def forward(self, x):
        
        x = self.convblock1(x)
        x = self.transblock1(x)
        
        x = self.convblock2(x)
        x = self.transblock2(x)

        x = self.convblock3(x)
        x = self.gap(x)
        
        x = self.fc1(x)
        
        x = x.view(-1, 10)
        return x
        
#net.load_state_dict(torch.load('/content/96K-model.pth'))






