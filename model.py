import torch
import torch.nn as nn
from torchvision import models

class Model(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        model = models.resnet18(pretrained=pretrained)
        model = list(model.children())[:-1]
        model.append(nn.AdaptiveAvgPool2d(7))
        

        self.model = nn.Sequential(*model)
        self.pred = nn.Conv2d(1, 4, 1)

    def forward(self, x):
        x = torch.mean(self.model(x), dim=1, keepdim=True)
        x = self.pred(x)
        return x
        