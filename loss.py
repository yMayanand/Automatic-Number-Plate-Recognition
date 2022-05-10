import torch.nn.functional as F
from utils import assign_cell

# function to compute loss


def loss_fn(preds, labels):
    losses = []
    for i, label in enumerate(labels):
        pos = assign_cell(label)
        a, b = pos
        loss = F.mse_loss(preds[i, :, a, b], label)
        losses.append(loss)
    avg_loss = 0
    for i in losses:
        avg_loss += i

    return avg_loss
