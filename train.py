import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from glob import glob
from tqdm import tqdm
import numpy as np

# 导入上面的模块
from src.dataset import DriveDataset, get_transforms
from src.model import UNet

# 简单的 DiceLoss 实现
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        bce = self.bce(inputs, targets)
        
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return bce + (1 - dice)

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0
    
    for x, y in tqdm(loader, desc="Training"):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        
    return epoch_loss / len(loader)

def main():
    # 配置
    DATA_PATH = "./data"
    IMG_SIZE = 512
    BATCH_SIZE = 4
    EPOCHS = 50
    LR = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 路径准备
    train_x = sorted(glob(os.path.join(DATA_PATH, "images", "*.tif"))) # 根据实际后缀修改
    train_y = sorted(glob(os.path.join(DATA_PATH, "mask", "*.gif")))
    od_mask_path = os.path.join(DATA_PATH, "od_mask")
    
    # 切分数据集 (简单切分，保留最后几张验证)
    val_split = 4
    train_imgs, val_imgs = train_x[:-val_split], train_x[-val_split:]
    train_masks, val_masks = train_y[:-val_split], train_y[-val_split:]
    
    # 数据集
    # 这里设置 use_od_guidance=True 开启你的创新点
    train_ds = DriveDataset(train_imgs, train_masks, od_mask_path, 
                            transform=get_transforms(IMG_SIZE), use_od_guidance=True)
    val_ds = DriveDataset(val_imgs, val_masks, od_mask_path, 
                          transform=get_transforms(IMG_SIZE), use_od_guidance=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # 模型
    print(f"初始化模型... 设备: {DEVICE}")
    model = UNet(in_channels=4).to(DEVICE) # 4通道！
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = DiceBCELoss()
    
    best_loss = float('inf')
    
    # 训练循环
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        
        # 简单验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                pred = model(vx)
                val_loss += loss_fn(pred, vy).item()
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model_od_guided.pth")
            print("模型已保存!")

if __name__ == "__main__":
    main()