from tracemalloc import start
from cv2 import selectROIs
import torch
import torch.nn as nn
from torchvision import models


def conv(in_channel, out_channel, kernel_size=3,
         stride=1, padding=0
         ):
    conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                     stride=stride, padding=padding)
    bn = nn.BatchNorm2d(out_channel)
    relu = nn.ReLU()
    return nn.Sequential(conv, bn, relu)


class OcrModel(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv1 = conv(3, 16)
        self.conv2 = conv(16, 32, stride=2)
        self.conv3 = conv(32, 64)
        self.conv4 = conv(64, 64, stride=2) # size : (b, c, h, w)
        self.conv5 = conv(64, 64, stride=2)
        self.conv6 = conv(64, 64, stride=2)
        self.lstm = nn.LSTM(384, 512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.permute((3, 0, 1, 2))
        x = torch.flatten(x, start_dim=2)
        out = self.lstm(x)
        return out
