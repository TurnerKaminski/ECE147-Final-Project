# This is the file in which we will define our models

# Importing
import torch
import torch.nn as nn
import torch.nn.functional as F

# Here we define a seperate class for each model
## TO DO:

# it will be something like this once we start defining models:

# class $modeltype$(nn.Module):
#     def __init__(self, inputsize, outputsize, hiddensize):
#         define the layers/convolutions
#     def forward(self, x):
#         calc forward passes

# Simple model BASED on the paper EEGnet from the project guidelines
# Table 2
# https://iopscience.iop.org/article/10.1088/1741-2552/aace8c#jneaace8cs2-2-1
class CNN(nn.Module):
    def __init__(self, input_size, N, dropout_p=0.5):
        super(CNN, self).__init__()
        
        C = 22 # num electrodes
        F1 = 8
        D = 2
        F2 = F1 * D
        SampleRate = 32

        # Batch 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, 32), bias=False),
            nn.BatchNorm2d(F1)
        )
        
        self.depthwise_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=F2, kernel_size=(C, 1), groups=F1, padding=(0, 0), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU()
        )
        
        self.avgpool1 = nn.AvgPool2d(kernel_size=(1, 2))
        self.dropout1 = nn.Dropout(p=dropout_p)
        
        # Batch 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=F2, out_channels=F2, kernel_size=(1, SampleRate//4), bias=False),
            nn.Conv2d(in_channels=F2, out_channels=F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=dropout_p)
        )
    
        
        w1 = int((input_size[1] - SampleRate + 1) / 1)
        w2 = int(w1 / 2)
        w3 = w2 - SampleRate // 4 + 1
        w4 = int(w3 / 4)
        # N is output size of model (number of classification optopns)
        self.dense = nn.Linear(16 * 1 * w4 , N)
        
    def forward(self, x):
        x = self.conv1(x.unsqueeze(1))
        x = self.depthwise_conv1(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        x = self.dense(x)
        
        return x
    
