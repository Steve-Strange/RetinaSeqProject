import os
import torch
import torch.nn as nn
import glob
import numpy as np
import random
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler # 混合精度训练

# 引入你的自定义模块
from src.dataset import SimpleDataset
from src.model import UNet as CustomUNet # 你之前手写的 U-Net
from src.utils import DiceBCELoss, calculate_metrics

# 引入强大的第三方库
import segmentation_models_pytorch as smp

def get_args():
    parser = argparse.ArgumentParser(description="Retinal Vessel Segmentation Training Pipeline")
    
    # === 路径配置 ===
    parser.add_argument("--data_path", type=str, default="./data/final_augmented_dataset", help="数据路径")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="模型保存路径")
    parser.add_argument("--log_dir", type=str, default="./logs", help="日志保存路径")
    
    # === 模型选择 (核心差异) ===
    # custom: 你之前手写的轻量级 U-Net (适合做 Baseline 对比)
    # resnet34: 工业界标准 Baseline，带 ImageNet 预训练
    # efficientnet-b4: 进阶模型，更强
    parser.add_argument("--model_arch", type=str, default="resnet34", 
                        choices=["custom", "resnet34", "resnet50", "efficientnet-b4"],
                        help="选择模型架构")
    
    # === 训练超参数 ===
    parser.add_argument("--batch_size", type=int, default=48, help="Batch Size")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程")
    
    # === 功能开关 ===
    parser.add_argument("--no_od", action="store_true", help="如果加了这个flag，就关闭视盘引导(变成3通道)")
    
    return parser.parse_args()

def get_model(arch, in_channels):
    """模型工厂函数"""
    print(f"正在构建模型: {arch} (Input Channels: {in_channels})...")
    
    if arch == "custom":
        # 你手写的 U-Net (从零训练)
        model = CustomUNet(in_channels=in_channels, num_classes=1)
        print(">>> 使用自定义 U-Net (Scratch)")
        
    else:
        # 使用 SMP 库构建带预训练权重的模型 (Transfer Learning)
        # encoder_weights="imagenet" 是快速收敛的关键
        model = smp.Unet(
            encoder_name=arch,        
            encoder_weights="imagenet",     
            in_channels=in_channels,                  
            classes=1,
            activation=None
        )
        print(f">>> 使用 SMP {arch} (Pretrained on ImageNet)")
        
    return model

def get_train_val_split(data_dir, val_ratio=0.2):
    """防止数据泄露的切分逻辑"""
    img_dir = os.path.join(data_dir, "images")
    all_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    
    unique_ids = set()
    for f in all_files:
        fname = os.path.basename(f)
        if '_aug_' in fname:
            base_id = fname.split('_aug_')[0]
        else:
            base_id = os.path.splitext(fname)[0]
        unique_ids.add(base_id)
    
    unique_ids = list(unique_ids)
    random.shuffle(unique_ids)
    
    split_idx = int(len(unique_ids) * (1 - val_ratio))
    train_ids = set(unique_ids[:split_idx])
    val_ids = set(unique_ids[split_idx:])
    
    train_paths = []
    val_paths = []
    
    for f in all_files:
        fname = os.path.basename(f)
        base_id = fname.split('_aug_')[0] if '_aug_' in fname else os.path.splitext(fname)[0]
        
        if base_id in train_ids:
            train_paths.append(f)
        else:
            val_paths.append(f)
            
    print(f"数据集切分: 训练集 {len(train_paths)} | 验证集 {len(val_paths)}")
    return train_paths, val_paths

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler):
    model.train()
    total_loss = 0
    
    for x, y in tqdm(loader, desc="Train"):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # 混合精度训练上下文
        with autocast():
            pred = model(x)
            loss = loss_fn(pred, y)
            loss = loss.mean() # Handle DataParallel
        
        # 缩放梯度
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    metrics = np.zeros(5) 
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Valid"):
            x, y = x.to(device), y.to(device)
            # 验证时不需要 autocast，或者是可选的
            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.mean().item()
            
            batch_metrics = calculate_metrics(pred, y)
            metrics += np.array(batch_metrics)
            
    return total_loss/len(loader), metrics/len(loader)

def main():
    # 1. 初始化
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_od = not args.no_od
    
    # 记录日志
    log_file = os.path.join(args.log_dir, f"log_{args.model_arch}.csv")
    history = []

    print(f"=== 开始训练任务 ===")
    print(f"架构: {args.model_arch}")
    print(f"OD引导: {use_od}")
    print(f"显卡: {os.environ.get('CUDA_VISIBLE_DEVICES', 'All')}")

    # 2. 数据准备
    train_imgs, val_imgs = get_train_val_split(args.data_path)
    masks_dir = os.path.join(args.data_path, "masks")
    od_masks_dir = os.path.join(args.data_path, "od_masks")
    
    train_ds = SimpleDataset(train_imgs, masks_dir, od_masks_dir, use_od)
    val_ds = SimpleDataset(val_imgs, masks_dir, od_masks_dir, use_od)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)
    
    # 3. 模型构建
    in_channels = 4 if use_od else 3
    model = get_model(args.model_arch, in_channels)
    model.to(device)
    
    # 多卡处理
    if torch.cuda.device_count() > 1:
        print(f">>> 启用 {torch.cuda.device_count()} 卡并行!")
        model = nn.DataParallel(model)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = DiceBCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5, verbose=True)
    scaler = GradScaler() # 混合精度
    
    best_dice = 0.0
    
    # 4. 训练循环
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        val_loss, val_metrics = validate(model, val_loader, loss_fn, device)
        
        dice = val_metrics[1]
        scheduler.step(dice)
        
        # 打印信息
        print(f"Loss: Train {train_loss:.4f} | Val {val_loss:.4f}")
        print(f"Metrics: Dice {dice:.4f} | IoU {val_metrics[0]:.4f} | Sen {val_metrics[3]:.4f}")
        
        # 记录日志
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "dice": dice,
            "iou": val_metrics[0],
            "sensitivity": val_metrics[3],
            "specificity": val_metrics[4]
        })
        pd.DataFrame(history).to_csv(log_file, index=False)
        
        # 保存最佳模型
        if dice > best_dice:
            best_dice = dice
            # 根据架构命名，防止覆盖
            save_name = f"best_{args.model_arch}_od{int(use_od)}.pth"
            save_path = os.path.join(args.save_dir, save_name)
            
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f">>> Best Model Saved: {save_name} (Dice: {best_dice:.4f})")

if __name__ == "__main__":
    main()