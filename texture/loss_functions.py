import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and focusing on hard texture samples.
    FL(pt) = - alpha * (1 - pt)^gamma * log(pt)
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_loss_function(algorithm_name="cross_entropy"):
    """
    Factory function for loss algorithms.
    """
    if algorithm_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif algorithm_name == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    elif algorithm_name == "focal_loss":
        return FocalLoss(gamma=2.0)
    else:
        raise ValueError(f"Unknown loss algorithm: {algorithm_name}")
