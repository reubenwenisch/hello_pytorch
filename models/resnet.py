import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ResBlock(nn.Module):
    """ Layer1 -
    X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
    R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
    Add(X, R1) """
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.X1 = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, bias=True, kernel_size=3, padding=1, stride=1),
                    nn.MaxPool2d(kernel_size = 2),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(),
                    ) # 32 -> 16
        self.R1 = nn.Sequential(
                    nn.Conv2d(out_channel, out_channel, bias=True, kernel_size=3, padding=1, stride=1),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(),
                    nn.Conv2d(out_channel, out_channel, bias=True, kernel_size=3, padding=1, stride=1),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU()) #16

    def forward(self, x):
        x1 = self.X1(x)
        r1 = self.R1(x1)
        return x1+r1

class ResNet(nn.Module):
    """Some Information about MyModule"""
    def __init__(self):
        super(ResNet, self).__init__()
        self.PrepLayer = nn.Sequential(nn.Conv2d(3, 64, bias=True, kernel_size=3, padding=1, stride=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU()) # 32 -> 32
        self.Layer1 = ResBlock(in_channel = 64, out_channel=128) #32 -> 16
        self.Layer2 = nn.Sequential(nn.Conv2d(128, 256, bias=True, kernel_size=3, padding=1, stride=1),
                    nn.MaxPool2d(kernel_size = 2),
                    nn.BatchNorm2d(256),
                    nn.ReLU()) # 16 -> 8
        self.Layer3 = ResBlock(in_channel = 256, out_channel=512) # 4
        self.max = nn.MaxPool2d(kernel_size = 4) # 4 -> 1
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features = 512, out_features = 10, bias=True)
        self.softmax = nn.Softmax(dim=1) # dim = 1
        
    def forward(self, x):
        x = self.PrepLayer(x)
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)
        x = self.max(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.softmax(x)
        return x