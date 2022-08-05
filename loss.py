import torch
import torch.nn.functional as F
from utils import assign_cell
import torch.nn as nn

# function to compute loss

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

class DetectionLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.ce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, ploc, plabel, gloc, glabel):
        self.loc_loss = self.mse_loss(ploc, gloc)
        self.conf_loss = self.ce_loss(plabel, glabel)
        total_loss = self.loc_loss + (0.5 * self.conf_loss)
        return total_loss