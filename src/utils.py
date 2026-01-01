import torch
import torch.nn as nn

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # BCE Part
        bce = self.bce(inputs, targets)
        
        # Dice Part
        inputs = torch.sigmoid(inputs)
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return bce + (1 - dice)

def calculate_metrics(pred, target, threshold=0.5):
    """计算常用的医学分割指标"""
    # pred是logits，先sigmoid再二值化
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    
    tp = (pred * target).sum()
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    
    epsilon = 1e-7
    
    iou = tp / (tp + fp + fn + epsilon)
    dice = 2 * tp / (2 * tp + fp + fn + epsilon)
    acc = (tp + tn) / (tp + tn + fp + fn + epsilon)
    sensitivity = tp / (tp + fn + epsilon) # 召回率
    specificity = tn / (tn + fp + epsilon) # 特异性
    
    return iou.item(), dice.item(), acc.item(), sensitivity.item(), specificity.item()