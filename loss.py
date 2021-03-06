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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion1 = nn.SmoothL1Loss(reduction='none')
criterion2 = nn.BCELoss()

def loss_fn(preds, labels):
    # bbox loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss = 0
    for i, label in enumerate(labels):
        pos = assign_cell(label)
        a, b = pos
        obj = criterion1(preds[i, :4, a, b], label[:4])
        temp = torch.zeros(7, 7).to(device)
        temp[a, b] = torch.tensor([1.]).to(device)
        conf = criterion2(torch.sigmoid(preds[i, 4, :, :]), temp)
        loss += obj + conf

    return loss.mean(dim=0)
