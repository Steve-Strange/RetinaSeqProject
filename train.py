import os
import torch
import glob
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import DriveDataset
from src.model import UNet
from src.transforms import get_train_transforms, get_val_transforms
from src.utils import DiceBCELoss, calculate_metrics

def train_one_epoch(model, loader, optimizer, loss_fn, device, use_mixup=True):
    model.train()
    total_loss = 0
    
    for x, y in tqdm(loader, desc="Train"):
        x, y = x.to(device), y.to(device)
        
        # MixUp Augmentation
        if use_mixup and np.random.random() > 0.5:
            lam = np.random.beta(1.0, 1.0)
            index = torch.randperm(x.size(0)).to(device)
            mixed_x = lam * x + (1 - lam) * x[index, :]
            mixed_y = lam * y + (1 - lam) * y[index, :]
            pred = model(mixed_x)
            loss = loss_fn(pred, mixed_y)
        else:
            pred = model(x)
            loss = loss_fn(pred, y)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    metrics = np.zeros(5) # IoU, Dice, Acc, Sen, Spe
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Valid"):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            
            # 累加指标
            batch_metrics = calculate_metrics(pred, y)
            metrics += np.array(batch_metrics)
            
    return total_loss/len(loader), metrics/len(loader)

def main():
    # 配置
    DATA_PATH = "./data/unified_data"
    BATCH_SIZE = 4
    EPOCHS = 50
    LR = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OD_GUIDANCE = True  # 你的创新点开关！
    
    # 路径
    imgs = sorted(glob.glob(os.path.join(DATA_PATH, "images", "*.png")))
    masks = sorted(glob.glob(os.path.join(DATA_PATH, "masks", "*.png")))
    od_path = os.path.join(DATA_PATH, "od_masks")
    
    # 切分
    split = int(len(imgs) * 0.8)
    train_ds = DriveDataset(imgs[:split], masks[:split], od_path, 
                            get_train_transforms(), use_od_guidance=OD_GUIDANCE)
    val_ds = DriveDataset(imgs[split:], masks[split:], od_path, 
                          get_val_transforms(), use_od_guidance=OD_GUIDANCE)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    # 模型 (如果开启 OD Guidance，输入通道为 4)
    in_channels = 4 if OD_GUIDANCE else 3
    print(f"初始化模型, 输入通道: {in_channels}...")
    model = UNet(in_channels=in_channels).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = DiceBCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    best_dice = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        val_loss, val_metrics = validate(model, val_loader, loss_fn, DEVICE)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Dice: {val_metrics[1]:.4f} | IoU: {val_metrics[0]:.4f} | Sen: {val_metrics[3]:.4f}")
        
        if val_metrics[1] > best_dice:
            best_dice = val_metrics[1]
            torch.save(model.state_dict(), "best_model.pth")
            print(">>> 模型已保存 (Best Dice)")

if __name__ == "__main__":
    main()