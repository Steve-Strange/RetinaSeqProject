"""
训练模块
包含：训练循环、验证、模型保存、损失曲线绘制
"""
import os
import time
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import PROCESSED_DATA_ROOT, CHECKPOINT_DIR, LOG_DIR, BATCH_SIZE, NUM_EPOCHS, LR, NUM_WORKERS
from .dataset import DriveDataset
from .model import get_model
from .loss import get_loss
from .utils import create_dir


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """训练一个epoch"""
    model.train()
    epoch_loss = 0.0
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def evaluate(model, loader, loss_fn, device):
    """验证"""
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)


def save_loss_plot(log_path, save_plot_path):
    """读取CSV并画损失曲线"""
    try:
        epochs, train_losses, valid_losses = [], [], []
        with open(log_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                epochs.append(int(row[0]))
                train_losses.append(float(row[1]))
                valid_losses.append(float(row[2]))

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label='Train Loss', color='blue')
        plt.plot(epochs, valid_losses, label='Valid Loss', color='red')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_plot_path)
        plt.close()
    except Exception as e:
        print(f"Error plotting loss: {e}")


def run_training(dataset_name, epochs=None, batch_size=None, lr=None, model_type='baseline', loss_type='vessel'):
    """
    执行训练流程

    Args:
        dataset_name: 数据集名称 (DRIVE/CHASE_DB/STARE)
        epochs: 训练轮数 (可选，默认使用config中的值)
        batch_size: 批次大小 (可选)
        lr: 学习率 (可选)
        model_type: 模型类型 ('baseline', 'dcn_sp', 'aspp')
        loss_type: 损失函数类型 ('vessel', 'dice', 'dice_bce', 'bce', 'focal_tversky')
    """
    print(f"\n[Train] Starting training for {dataset_name}...")
    print(f"  Model: {model_type} | Loss: {loss_type}")

    # 使用传入参数或默认值
    _epochs = epochs or NUM_EPOCHS
    _batch_size = batch_size or BATCH_SIZE
    _lr = lr or LR

    # 准备路径 (包含模型类型后缀以区分不同实验)
    create_dir(CHECKPOINT_DIR)
    create_dir(LOG_DIR)

    suffix = f"_{model_type}" if model_type != 'baseline' else ""
    data_path = os.path.join(PROCESSED_DATA_ROOT, dataset_name)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{dataset_name.lower()}{suffix}_checkpoint.pth")
    log_csv_path = os.path.join(LOG_DIR, f"{dataset_name}{suffix}_log.csv")
    plot_path = os.path.join(LOG_DIR, f"{dataset_name}{suffix}_loss.png")

    # 初始化CSV
    with open(log_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "valid_loss"])

    # 数据加载
    train_x = sorted(glob(os.path.join(data_path, "train", "image", "*")))
    train_y = sorted(glob(os.path.join(data_path, "train", "mask", "*")))
    valid_x = sorted(glob(os.path.join(data_path, "test", "image", "*")))
    valid_y = sorted(glob(os.path.join(data_path, "test", "mask", "*")))

    print(f"Train samples: {len(train_x)} | Valid samples: {len(valid_x)}")

    train_ds = DriveDataset(train_x, train_y)
    valid_ds = DriveDataset(valid_x, valid_y)

    train_loader = DataLoader(train_ds, batch_size=_batch_size, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(valid_ds, batch_size=_batch_size, shuffle=False, num_workers=NUM_WORKERS)

    # 模型准备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_type)

    if torch.cuda.device_count() > 1:
        print(f"[System] {torch.cuda.device_count()} GPUs detected. Using DataParallel.")
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=_lr)
    loss_fn = get_loss(loss_type)

    best_valid_loss = float("inf")

    # 训练循环
    for epoch in range(_epochs):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)

        end_time = time.time()
        epoch_mins, epoch_secs = int((end_time - start_time) // 60), int((end_time - start_time) % 60)

        print(f"Epoch {epoch+1:03d}/{_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f} | Time: {epoch_mins}m {epoch_secs}s")

        # 保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), ckpt_path)
            else:
                torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Best model saved! Val Loss: {valid_loss:.4f}")

        # 写入日志
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, valid_loss])

    print(f"[Train] Finished {dataset_name}. Best Loss: {best_valid_loss:.4f}")

    # 画损失曲线
    save_loss_plot(log_csv_path, plot_path)
    print(f"[Train] Loss plot saved to {plot_path}")

    # 清理显存
    del model, optimizer, train_loader, valid_loader
    torch.cuda.empty_cache()
