"""
工具函数模块
包含：随机种子、目录创建、时间计算、指标计算、可视化辅助函数
"""
import os
import random
import numpy as np
import cv2
import torch
from sklearn.metrics import confusion_matrix


def seeding(seed):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_dir(path):
    """创建目录"""
    if not os.path.exists(path):
        os.makedirs(path)


def epoch_time(start_time, end_time):
    """计算epoch耗时"""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def calculate_metrics(y_true, y_pred):
    """
    计算分割指标
    返回: [jaccard, f1, recall/sensitivity, precision, accuracy, specificity]
    """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8).reshape(-1)

    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8).reshape(-1)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    score_f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
    score_jaccard = tp / (tp + fp + fn + 1e-6)
    score_recall = tp / (tp + fn + 1e-6)
    score_specificity = tn / (tn + fp + 1e-6)
    score_precision = tp / (tp + fp + 1e-6)
    score_acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_specificity]


# ==========================================
# 可视化辅助函数
# ==========================================

def mask_to_bgr(mask):
    """单通道掩码转BGR格式用于显示"""
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
    h, w = gt.shape[:2]
    error_map = np.zeros((h, w, 3), dtype=np.uint8)
    error_map[(gt == 1) & (pred == 1)] = [0, 255, 0]   # TP: Green
    error_map[(gt == 1) & (pred == 0)] = [0, 0, 255]   # FN: Red
    error_map[(gt == 0) & (pred == 1)] = [255, 0, 0]   # FP: Blue
    return error_map


def add_label(image, text):
    """在图像上添加文字标签"""
    img_copy = image.copy()
    cv2.putText(img_copy, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return img_copy


# ==========================================
# OD区域处理 (V6: 距离场 + Gamma控制)
# ==========================================

def harmonize_od_edge(image_bgr, od_mask,
                      dist_out_px=30, gamma_out=1.0,
                      dist_in_px=10, gamma_in=4.0,
                      intensity=50):
    """
    视盘边缘增强处理
    使用距离场和Gamma控制来平滑OD边缘
    """
    if np.sum(od_mask) == 0:
        return image_bgr

    # A. 计算距离场
    dist_map_out = cv2.distanceTransform(cv2.bitwise_not(od_mask), cv2.DIST_L2, 5)
    dist_map_in = cv2.distanceTransform(od_mask, cv2.DIST_L2, 5)

    # B. 生成权重 (指数衰减)
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_dist_out = dist_map_out / dist_out_px
        mask_out_zone = (dist_map_out > 0) & (dist_map_out <= dist_out_px)
        weight_out = np.zeros_like(dist_map_out, dtype=np.float32)
        weight_out[mask_out_zone] = np.power(np.clip(1.0 - norm_dist_out[mask_out_zone], 0, 1), gamma_out)

    with np.errstate(divide='ignore', invalid='ignore'):
        norm_dist_in = dist_map_in / dist_in_px
        mask_in_zone = (dist_map_in > 0) & (dist_map_in <= dist_in_px)
        weight_in = np.zeros_like(dist_map_in, dtype=np.float32)
        weight_in[mask_in_zone] = np.power(np.clip(1.0 - norm_dist_in[mask_in_zone], 0, 1), gamma_in)

    weight_map = weight_out + weight_in

    # C. 血管保护 (黑帽变换)
    green = image_bgr[:, :, 1]
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, morph_kernel)
    _, vessel_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    vessel_float = cv2.GaussianBlur(vessel_mask.astype(np.float32) / 255.0, (3, 3), 0)
    weight_map = weight_map * (1.0 - vessel_float)
    weight_map = cv2.GaussianBlur(weight_map, (3, 3), 0)

    # D. 应用提亮
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_enhanced = v.astype(np.float32) + (weight_map * float(intensity))
    v_final = np.clip(v_enhanced, 0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge([h, s, v_final]), cv2.COLOR_HSV2BGR)
