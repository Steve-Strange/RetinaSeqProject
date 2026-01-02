import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
import cv2
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 引入 TransUNet
from src.transunet import TransUNet

# === 配置 ===
# 1. 换回最原始的统一数据集 (RGB)
DATA_DIR = "./data/unified_data"
BATCH_SIZE = 64  # TransUNet比较吃显存，先用24试试，稳了再加
LR = 1e-4
EPOCHS = 100
IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === 2. 定义实时数据增强 (Online Augmentation) ===
def get_train_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        # 几何变换
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        # 简单的光照变化 (不要太强)
        A.RandomBrightnessContrast(p=0.2),
        # 归一化 (关键！TransUNet/ResNet 喜欢标准化输入)
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

# === 3. 数据集类 ===
class StandardDataset(Dataset):
    def __init__(self, image_paths, mask_dir, transform=None):
        self.image_paths = image_paths
        self.mask_dir = mask_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        fname = os.path.basename(img_path)
        mask_path = os.path.join(self.mask_dir, fname)
        
        # 读取图片 (RGB)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取 Mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 二值化 Mask (确保只有 0 和 255)
        # 有些数据集resize后会有中间值，强制二值化
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 应用增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # 处理 Mask (0-255 -> 0-1 float)
        mask = mask.float() / 255.0
        mask = mask.unsqueeze(0) # [1, H, W]
        
        return image, mask

# === 4. 验证与保存 ===
class EarlyStopping:
    def __init__(self, patience=15, delta=0.001, path='best_transunet_standard.pth'):
        self.patience = patience
        self.best_score = None
        self.counter = 0
        self.path = path
        self.early_stop = False

    def __call__(self, val_iou, model):
        if self.best_score is None:
            self.best_score = val_iou
            self.save_checkpoint(model)
        elif val_iou < self.best_score + 0.001:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_iou
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), self.path)
        else:
            torch.save(model.state_dict(), self.path)
        print(f">>> Model Saved! Best IoU: {self.best_score:.4f}")

def calculate_metrics(preds, targets):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (preds.sum() + targets.sum() + 1e-6)
    return iou.item(), dice.item()

# === 5. 训练主循环 ===
def main():
    # 路径检查
    img_dir = os.path.join(DATA_DIR, "images")
    mask_dir = os.path.join(DATA_DIR, "masks")
    all_imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    
    if len(all_imgs) == 0:
        print(f"❌ 错误: 在 {img_dir} 没找到图片。请先运行 prepare_data.py")
        return

    # 切分
    split_idx = int(len(all_imgs) * 0.8)
    train_imgs = all_imgs[:split_idx]
    val_imgs = all_imgs[split_idx:]
    
    print(f"训练集: {len(train_imgs)} | 验证集: {len(val_imgs)}")
    
    train_ds = StandardDataset(train_imgs, mask_dir, transform=get_train_transforms())
    val_ds = StandardDataset(val_imgs, mask_dir, transform=get_val_transforms())
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    # 模型
    model = TransUNet(img_size=IMG_SIZE, in_channels=3, out_channels=1)
    model = model.to(DEVICE)
    
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 张显卡并行")
        model = nn.DataParallel(model)
        
    # 损失函数 (Dice + BCE)
    bce_loss = nn.BCEWithLogitsLoss()
    
    def criterion(pred, target):
        bce = bce_loss(pred, target)
        pred_prob = torch.sigmoid(pred)
        dice = (2. * (pred_prob * target).sum() + 1) / (pred_prob.sum() + target.sum() + 1)
        return 0.5 * bce + 0.5 * (1 - dice)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5, verbose=True)
    early_stopping = EarlyStopping(patience=15, path="./checkpoints/best_transunet_std.pth")
    
    os.makedirs("./checkpoints", exist_ok=True)
    
    # 检查数据 (Sanity Check)
    # 取一个batch看看是不是全黑的
    debug_img, debug_mask = next(iter(train_loader))
    print(f"数据检查: Image Max={debug_img.max():.2f}, Min={debug_img.min():.2f}")
    print(f"数据检查: Mask Max={debug_mask.max()}, Min={debug_mask.min()}, Mean={debug_mask.mean():.4f}")
    if debug_mask.max() == 0:
        print("❌ 警告: Mask 全黑！检查 prepare_data.py 是否正确生成了 mask")
        return

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, masks in loop:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        # 验证
        model.eval()
        val_iou = 0
        val_dice = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs)
                iou, dice = calculate_metrics(preds, masks)
                val_iou += iou
                val_dice += dice
                
        avg_iou = val_iou / len(val_loader)
        avg_dice = val_dice / len(val_loader)
        scheduler.step(avg_iou)
        
        print(f"Val IoU: {avg_iou:.4f} | Val Dice: {avg_dice:.4f}")
        
        early_stopping(avg_iou, model)
        if early_stopping.early_stop:
            print("早停！")
            break

if __name__ == "__main__":
    main()