import torch
import torch.nn as nn
from torchvision import models

class Model(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        model = models.resnet18(pretrained=pretrained)
        model = list(model.children())[:-2]
        model.append(nn.AdaptiveAvgPool2d(7))
        
        self.relu = nn.ReLU()
        self.model = nn.Sequential(*model)
        self.pred = nn.Conv2d(512, 6, 1)

    def forward(self, x):
        x = self.model(x)
        x = self.pred(x)
        return x
        