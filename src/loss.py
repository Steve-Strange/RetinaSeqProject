"""
损失函数模块
支持: vessel (Focal Tversky + Boundary), dice, dice_bce
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss
    特别适合类别极不平衡 + 微结构（血管）

    Args:
        alpha: FP权重 (默认0.7)
        beta: FN权重 (默认0.3)
        gamma: focal参数 (默认0.75)
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        loss = torch.pow((1 - tversky), self.gamma)
        return loss


class BoundaryLoss(nn.Module):
    """
    Boundary Loss via Sobel
    强制预测边界贴近 GT 边界
    """
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)

        def get_boundary(x):
            sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=x.device).float()
            sobel_y = sobel_x.t()
            sobel_x = sobel_x.view(1, 1, 3, 3)
            sobel_y = sobel_y.view(1, 1, 3, 3)
            grad_x = F.conv2d(x, sobel_x, padding=1)
            grad_y = F.conv2d(x, sobel_y, padding=1)
            return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        pred_boundary = get_boundary(inputs)
        gt_boundary = get_boundary(targets)
        return F.l1_loss(pred_boundary, gt_boundary)


class VesselSegmentationLoss(nn.Module):
    """
    血管分割联合损失：Focal Tversky + Boundary

    Args:
        w_tversky: Tversky损失权重 (默认1.0)
        w_boundary: Boundary损失权重 (默认0.1)
    """
    def __init__(self, w_tversky=1.0, w_boundary=0.1):
        super(VesselSegmentationLoss, self).__init__()
        self.tversky = FocalTverskyLoss()
        self.boundary = BoundaryLoss()
        self.w_tversky = w_tversky
        self.w_boundary = w_boundary

    def forward(self, inputs, targets):
        loss_t = self.tversky(inputs, targets)
        loss_b = self.boundary(inputs, targets)
        return self.w_tversky * loss_t + self.w_boundary * loss_b


class DiceLoss(nn.Module):
    """
    Dice Loss
    适用于二分类分割任务
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class DiceBCELoss(nn.Module):
    """
    Dice + BCE Loss
    结合Dice和二元交叉熵
    """
    def __init__(self, smooth=1.0):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs_sig = torch.sigmoid(inputs)
        inputs_flat = inputs_sig.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        bce = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean')

        return bce + dice_loss


class BCEWithLogitsLoss(nn.Module):
    """
    标准BCE Loss (带Logits)
    """
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)


# ==========================================
# 损失函数工厂
# ==========================================

LOSS_REGISTRY = {
    'vessel': VesselSegmentationLoss,      # Focal Tversky + Boundary (推荐)
    'dice': DiceLoss,                       # 纯Dice
    'dice_bce': DiceBCELoss,               # Dice + BCE
    'bce': BCEWithLogitsLoss,              # 纯BCE
    'focal_tversky': FocalTverskyLoss,     # 纯Focal Tversky
}


def get_loss(loss_type='vessel', **kwargs):
    """
    获取损失函数实例

    Args:
        loss_type: 损失函数类型 ('vessel', 'dice', 'dice_bce', 'bce', 'focal_tversky')
        **kwargs: 传递给损失函数的参数

    Returns:
        损失函数实例
    """
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[loss_type](**kwargs)
