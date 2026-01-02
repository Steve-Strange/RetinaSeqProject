import os
import argparse
import time
from glob import glob
import torch
from torch.utils.data import DataLoader
from src.dataset import DriveDataset
from src.model import LFANet
from src.loss import VesselSegmentationLoss
from src.utils import seeding, create_dir, epoch_time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--data_path", type=str, default="output/processed_data")
    parser.add_argument("--checkpoint_dir", type=str, default="output/checkpoints")
    parser.add_argument("--use_od_input", action="store_true", help="是否将OD Mask作为第4个通道输入")
    return parser.parse_args()

def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0
    model.train()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)

if __name__ == "__main__":
    args = get_args()
    seeding(42)
    create_dir(args.checkpoint_dir)
    
    # Paths
    train_x = sorted(glob(os.path.join(args.data_path, "training/image/*")))
    train_y = sorted(glob(os.path.join(args.data_path, "training/mask/*")))
    train_od = sorted(glob(os.path.join(args.data_path, "training/od_mask/*")))
    
    valid_x = sorted(glob(os.path.join(args.data_path, "test/image/*")))
    valid_y = sorted(glob(os.path.join(args.data_path, "test/mask/*")))
    valid_od = sorted(glob(os.path.join(args.data_path, "test/od_mask/*")))
    
    print(f"Train Size: {len(train_x)}, Valid Size: {len(valid_x)}")

    # Dataset
    train_ds = DriveDataset(train_x, train_y, train_od, use_od_input=args.use_od_input)
    valid_ds = DriveDataset(valid_x, valid_y, valid_od, use_od_input=args.use_od_input)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_channels = 4 if args.use_od_input else 3 # 3通道(特征栈) or 4通道(特征栈+OD)
    
    model = LFANet(input_channels=input_channels)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = VesselSegmentationLoss()
    
    best_valid_loss = float("inf")
    ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)
        
        if valid_loss < best_valid_loss:
            print(f"Valid Loss Improved: {best_valid_loss:.4f} -> {valid_loss:.4f}. Saving...")
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), ckpt_path)
            
        end_time = time.time()
        m, s = epoch_time(start_time, end_time)
        print(f"Epoch {epoch+1}/{args.epochs} | Time: {m}m {s}s | Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f}")