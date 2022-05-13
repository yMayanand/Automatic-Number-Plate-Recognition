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

criterion1 = nn.SmoothL1Loss()
criterion2 = nn.CrossEntropyLoss(weight=torch.tensor([0., 48.]))

def loss_fn(preds, labels):
    # bbox loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss = 0
    for i, label in enumerate(labels):
        pos = assign_cell(label)
        a, b = pos
        obj = criterion1(preds[i, :4, a, b], label[:4])
        temp = torch.zeros(1, 7, 7, dtype=torch.long).to(device)
        temp[:, a, b] = torch.tensor([1]).to(device)
        print(temp.device)
        temp1 = preds[i, 4:, :, :].reshape(2, 49).permute(1, 0).to(device)
        temp2 = temp.reshape(-1).to(device)
        loss2 = criterion2(temp1, temp2)
        #conf = criterion2(torch.sigmoid(preds[i, 4, a, b]), label[4])
        loss += obj + loss2

    return loss
