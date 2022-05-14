import torch
import torch.nn as nn
from torchvision import models

class Model(nn.Module):
    def __init__(self, backbone, out_channels):
        super().__init__()
        self.model = backbone
        self.pred = nn.Conv2d(out_channels, 5, 1)

    def forward(self, x):
        x = self.model(x)
        x = self.pred(x)
        return x
        
#model = models.resnet18(pretrained=pretrained)
#model = list(model.children())[:-2]
#model.append(nn.AdaptiveAvgPool2d(7))