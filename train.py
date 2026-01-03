import os
import time
import csv
import matplotlib.pyplot as plt
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import PROCESSED_DATA_ROOT, CHECKPOINT_DIR, BATCH_SIZE, NUM_EPOCHS, LR, NUM_WORKERS
from src.dataset import DriveDataset
from src.model import build_unet
from src.loss import VesselSegmentationLoss
from src.utils import create_dir, epoch_time

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0.0
    for i, (x, y) in enumerate(loader):
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
    """ 读取CSV并画图 """
    try:
        epochs = []
        train_losses = []
        valid_losses = []
        
        with open(log_path, "r") as f:
            reader = csv.reader(f)
            next(reader) # 跳过表头
            for row in reader:
                epochs.append(int(row[0]))
                train_losses.append(float(row[1]))
                valid_losses.append(float(row[2]))
        
        # 切换后端，防止在服务器无头模式下报错
        plt.switch_backend('agg') 
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

def run_training(dataset_name):
    print(f"\n[Train] Starting training for {dataset_name}...")
    
    # === 1. 准备路径 ===
    create_dir(CHECKPOINT_DIR)
    
    # 定义日志保存目录 (新建一个 logs 文件夹)
    LOG_DIR = "working/logs"
    create_dir(LOG_DIR)
    
    # 定义文件路径
    data_path = os.path.join(PROCESSED_DATA_ROOT, dataset_name)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{dataset_name.lower()}_checkpoint.pth")
    log_csv_path = os.path.join(LOG_DIR, f"{dataset_name}_log.csv")
    plot_path = os.path.join(LOG_DIR, f"{dataset_name}_loss.png")
    
    # === 2. 初始化 CSV 文件 ===
    # 每次重新训练都会覆盖旧的日志
    with open(log_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "valid_loss"])
    
    # === 3. 数据加载 ===
    train_x = sorted(glob(os.path.join(data_path, "train", "image", "*")))
    train_y = sorted(glob(os.path.join(data_path, "train", "mask", "*")))
    valid_x = sorted(glob(os.path.join(data_path, "test", "image", "*")))
    valid_y = sorted(glob(os.path.join(data_path, "test", "mask", "*")))
    
    print(f"Train samples: {len(train_x)} | Valid samples: {len(valid_x)}")

    train_ds = DriveDataset(train_x, train_y)
    valid_ds = DriveDataset(valid_x, valid_y)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # === 4. 模型准备 ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_unet()
    
    # 依然保留 DataParallel 逻辑，虽然在 main.py 里我们用单卡多进程，
    # 但如果为了兼容单独运行这个脚本，保留它没坏处。
    if torch.cuda.device_count() > 1:
        print(f"[System] {torch.cuda.device_count()} GPUs detected. Using DataParallel.")
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = VesselSegmentationLoss()

    best_valid_loss = float("inf")

    # === 5. 训练循环 ===
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)
        print(f"epoch:{epoch} train_loss:{train_loss} valid_loss:{valid_loss}")
        
        # --- 保存最佳模型 ---
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), ckpt_path)
            else:
                torch.save(model.state_dict(), ckpt_path)
            # print(f"Epoch {epoch+1} Saved Best. Val Loss: {valid_loss:.4f}")
            
        # --- 写入日志 ---
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, valid_loss])
            
        # end_time = time.time()
        # mins, secs = epoch_time(start_time, end_time)

    print(f"[Train] Finished {dataset_name}. Best Loss: {best_valid_loss:.4f}")
    
    # === 6. 训练结束后画图 ===
    save_loss_plot(log_csv_path, plot_path)
    print(f"[Train] Loss plot saved to {plot_path}")
    
    # 清理显存
    del model, optimizer, train_loader, valid_loader
    torch.cuda.empty_cache()