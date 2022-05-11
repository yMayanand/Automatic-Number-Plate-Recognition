import torch
import torch.nn.functional as F
from utils import assign_cell

# function to compute loss

"""def loss_fn(preds, labels, alpha=0.):
    # bounding box loss
    bbox_loss = (1 - alpha) * F.mse_loss(preds[:, :4, :, :], labels[:, :4, :, :])

    # objectness loss
    object_loss = alpha * F.binary_cross_entropy(torch.sigmoid(preds[:, 4, :, :]), labels[:, 4, :, :])
    return bbox_loss + object_loss"""


def loss_fn(preds, labels, alpha=1, beta=1):
    # bbox loss
    loss = 0
    count = 0
    for i, label in enumerate(labels):
        pos = assign_cell(label)
        a, b = pos
        loss1 = F.mse_loss(preds[i, :4, a, b], label[:4])
        loss2 = F.binary_cross_entropy(torch.sigmoid(preds[i, 4, a, b]), label[4])
        loss += beta * loss1 + alpha * loss2
        count += 1

    return loss/count
