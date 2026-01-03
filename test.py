import os
import time
import numpy as np
import cv2
import torch
from glob import glob
from tqdm import tqdm
from operator import add

from src.config import PROCESSED_DATA_ROOT, CHECKPOINT_DIR, RESULT_DIR
from src.model import build_unet
from src.utils import create_dir, calculate_metrics
from src.od_utils import ODSegmenter

def mask_parse(mask):
    """单通道转3通道"""
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=-1)
    if mask.shape[-1] == 1:
        mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask

def draw_error_map(gt, pred):
    """
    绘制差异图：
    - 绿色 (Green): 正确 (TP)
    - 红色 (Red): 少了 (FN)
    - 蓝色 (Blue): 多了 (FP)
    """
    gt = (gt > 0.5).astype(np.uint8)
    pred = (pred > 0.5).astype(np.uint8)
    h, w = gt.shape
    # OpenCV 使用 BGR
    error_map = np.zeros((h, w, 3), dtype=np.uint8)
    # TP: Green [0, 255, 0]
    error_map[(gt == 1) & (pred == 1)] = [0, 255, 0]
    # FN (少): Red [0, 0, 255]
    error_map[(gt == 1) & (pred == 0)] = [0, 0, 255]
    # FP (多): Blue [255, 0, 0]
    error_map[(gt == 0) & (pred == 1)] = [255, 0, 0]
    return error_map

def run_testing(dataset_name):
    print(f"\n[Test] Starting testing for {dataset_name}...")
    
    data_path = os.path.join(PROCESSED_DATA_ROOT, dataset_name)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{dataset_name.lower()}_checkpoint.pth")
    save_path = os.path.join(RESULT_DIR, dataset_name)
    create_dir(save_path)
    
    test_x = sorted(glob(os.path.join(data_path, "test", "image", "*")))
    test_y = sorted(glob(os.path.join(data_path, "test", "mask", "*")))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载血管分割模型
    model = build_unet().to(device)
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found for {dataset_name}")
        return
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # 2. 初始化视盘分割模型
    od_segmenter = ODSegmenter(device=device)
    
    metrics_score = [0.0] * 6
    
    for i, (x_p, y_p) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = os.path.basename(x_p).split(".")[0]
        
        # 读取数据
        image_bgr = cv2.imread(x_p, cv2.IMREAD_COLOR)
        x = np.transpose(image_bgr, (2, 0, 1)) / 255.0
        x = np.expand_dims(x, axis=0).astype(np.float32)
        x = torch.from_numpy(x).to(device)
        
        mask_gray = cv2.imread(y_p, cv2.IMREAD_GRAYSCALE)
        y = np.expand_dims(mask_gray, axis=0) / 255.0
        y = np.expand_dims(y, axis=0).astype(np.float32)
        y = torch.from_numpy(y).to(device)
        
        with torch.no_grad():
            # 血管预测
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            
            score = calculate_metrics(y, pred_y)
            metrics_score = list(map(add, metrics_score, score))
            
            # 转 numpy
            pred_y_np = pred_y[0].cpu().numpy().squeeze(0)
            pred_binary = (pred_y_np > 0.5).astype(np.uint8)
            
            # 视盘预测
            od_mask = od_segmenter.process(image_bgr)

        # 绘图
        gt_show = mask_parse(mask_gray)
        od_show = mask_parse(od_mask)
        gt_binary = (mask_gray > 127).astype(np.uint8)
        error_map = draw_error_map(gt_binary, pred_binary)
        
        # 拼接: 原图 | 真值 | OD | 误差图
        h, w, _ = image_bgr.shape
        sep = np.ones((h, 10, 3), dtype=np.uint8) * 255
        final_img = np.concatenate([image_bgr, sep, gt_show, sep, od_show, sep, error_map], axis=1)
        
        cv2.imwrite(os.path.join(save_path, f"{name}.png"), final_img)
        
    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    acc = metrics_score[4]/len(test_x)
    print(f"[{dataset_name} Result] Jaccard: {jaccard:.4f} | F1: {f1:.4f} | Acc: {acc:.4f}")
    
    del model, od_segmenter
    torch.cuda.empty_cache()