import torch
import torch.nn as nn
from torchvision import models

class Model(nn.Module):
    def __init__(self, backbone, out_channels):
        super().__init__()
        self.model = backbone
        self.pred = nn.Conv2d(out_channels, 20, 1)
        

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.model(x)
        x = self.pred(x) # shape: (batch_size, 4, 5, 7, 7)
        h, w = x.shape[2:]
        x = x.reshape(batch_size, -1, 5, h*w).permute(0, 1, 3, 2).reshape(batch_size, -1, 5)
        loc = torch.sigmoid(x[:, :, :4])
        conf = x[:, :, 4]
        return loc, conf