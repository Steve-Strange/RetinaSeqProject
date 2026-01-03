import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
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