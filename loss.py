import torch
import torch.nn.functional as F
from utils import assign_cell
import torch.nn as nn

# function to compute loss

"""def loss_fn(preds, labels, alpha=0.):
    # bounding box loss
    bbox_loss = (1 - alpha) * F.mse_loss(preds[:, :4, :, :], labels[:, :4, :, :])

    # objectness loss
    object_loss = alpha * F.binary_cross_entropy(torch.sigmoid(preds[:, 4, :, :]), labels[:, 4, :, :])
    return bbox_loss + object_loss"""

criterion = nn.SmoothL1Loss()

def loss_fn(preds, labels):
    # bbox loss
    loss = 0
    for i, label in enumerate(labels):
        pos = assign_cell(label)
        a, b = pos
        loss1 = criterion(preds[i, :, a, b], label)
        loss += loss1

    return loss
