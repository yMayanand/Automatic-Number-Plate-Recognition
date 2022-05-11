import torch
import torch.nn.functional as F
from utils import assign_cell

# function to compute loss

def loss_fn(preds, labels):
    # bounding box loss
    bbox_loss = F.mse_loss(preds[:, :4, :, :], labels[:, :4, :, :])

    # objectness loss
    object_loss = F.binary_cross_entropy(torch.sigmoid(preds[:, 4, :, :]), labels[:, 4, :, :])
    return bbox_loss + object_loss


"""def loss_fn(preds, labels):
    losses = []
    for i, label in enumerate(labels):
        pos = assign_cell(label)
        a, b = pos
        loss = F.mse_loss(preds[i, :, a, b], label)
        losses.append(loss)
    avg_loss = 0
    for i in losses:
        avg_loss += i

    return avg_loss"""
