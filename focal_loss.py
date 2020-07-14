import torch
from torch import nn
from torch.nn import functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, smooth_eps=0.0, logits=False,
                 reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.smooth_eps = smooth_eps
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.smooth_eps > 0:
            # From https://arxiv.org/pdf/1512.00567.pdf
            targets = targets.float()
            targets = (1-self.smooth_eps)*targets + self.smooth_eps/2
        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise NotImplementedError('Only mean reduction is implemented')
