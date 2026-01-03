import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from glob import glob
from tqdm import tqdm
from operator import add
from collections import OrderedDict
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from src.config import PROCESSED_DATA_ROOT, CHECKPOINT_DIR, RESULT_DIR, TARGET_DATASETS
from src.model import build_unet
from src.utils import create_dir, calculate_metrics

# ==========================================
# 1. 视盘分割器
# ==========================================
class ODSegmenter:
    def __init__(self, device=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "pamixsun/segformer_for_optic_disc_cup_segmentation"
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading OD model: {e}")
            self.model = None

    def process(self, image_bgr):
        if self.model is None: return np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(img_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        upsampled = torch.nn.functional.interpolate(
            logits, size=image_bgr.shape[:2], mode="bilinear", align_corners=False
        )
        pred = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        od_mask = np.where(pred > 0, 255, 0).astype(np.uint8)
        return od_mask

# ==========================================
# 2. 图像处理核心 (V6: 距离场 + Gamma控制)
# ==========================================
def harmonize_od_edge(image_bgr, od_mask, 
                      dist_out_px=30, gamma_out=1.0, 
                      dist_in_px=10,  gamma_in=4.0, 
                      intensity=50):
    if np.sum(od_mask) == 0: return image_bgr

    # A. 计算距离场
    dist_map_out = cv2.distanceTransform(cv2.bitwise_not(od_mask), cv2.DIST_L2, 5)
    dist_map_in = cv2.distanceTransform(od_mask, cv2.DIST_L2, 5)

    # B. 生成权重 (指数衰减)
    # 1. 向外
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_dist_out = dist_map_out / dist_out_px
        mask_out_zone = (dist_map_out > 0) & (dist_map_out <= dist_out_px)
        weight_out = np.zeros_like(dist_map_out, dtype=np.float32)
        weight_out[mask_out_zone] = np.power(np.clip(1.0 - norm_dist_out[mask_out_zone], 0, 1), gamma_out)

    # 2. 向内
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
    
    # 柔化血管边缘
    vessel_float = cv2.GaussianBlur(vessel_mask.astype(np.float32) / 255.0, (3, 3), 0)
    weight_map = weight_map * (1.0 - vessel_float)
    
    # 平滑权重图
    weight_map = cv2.GaussianBlur(weight_map, (3, 3), 0)

    # D. 应用提亮
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_enhanced = v.astype(np.float32) + (weight_map * float(intensity))
    v_final = np.clip(v_enhanced, 0, 255).astype(np.uint8)
    
    return cv2.cvtColor(cv2.merge([h, s, v_final]), cv2.COLOR_HSV2BGR)

# ==========================================
# 3. 辅助绘图函数
# ==========================================
def mask_to_bgr(mask):
    if len(mask.shape) == 2: mask = np.expand_dims(mask, axis=-1)
    return np.concatenate([mask, mask, mask], axis=-1)

def draw_error_map(gt, pred):
    gt = (gt > 0.5).astype(np.uint8)
    pred = (pred > 0.5).astype(np.uint8)
    h, w = gt.shape
    error_map = np.zeros((h, w, 3), dtype=np.uint8)
    error_map[(gt == 1) & (pred == 1)] = [0, 255, 0] # TP Green
    error_map[(gt == 1) & (pred == 0)] = [0, 0, 255] # FN Red
    error_map[(gt == 0) & (pred == 1)] = [255, 0, 0] # FP Blue
    return error_map

def add_label(image, text):
    img_copy = image.copy()
    cv2.putText(img_copy, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return img_copy

# ==========================================
# 4. 对比测试主程序
# ==========================================
def run_comparative_test(dataset_name):
    print(f"\n{'='*60}")
    print(f"Running COMPARATIVE Test (6 Columns): {dataset_name}")
    print(f"{'='*60}")
    
    data_path = os.path.join(PROCESSED_DATA_ROOT, dataset_name)
    save_path = os.path.join(RESULT_DIR, f"{dataset_name}_Final_Visual_V6")
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{dataset_name.lower()}_checkpoint.pth")
    create_dir(save_path)

    test_x = sorted(glob(os.path.join(data_path, "test", "image", "*")))
    test_y = sorted(glob(os.path.join(data_path, "test", "mask", "*")))

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载血管模型
    model = build_unet().to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items(): new_state_dict[k.replace("module.", "")] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    # 加载OD模型
    od_segmenter = ODSegmenter(device=device)

    metrics_base = [0.0] * 6
    metrics_opt = [0.0] * 6
    
    for x_path, y_path in tqdm(zip(test_x, test_y), total=len(test_x)):
        name = os.path.basename(x_path).split(".")[0]
        
        img_orig = cv2.imread(x_path)
        mask_gt = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
        y_tensor = torch.from_numpy(np.expand_dims(np.expand_dims(mask_gt,0),0)/255.0).float().to(device)
        
        # 1. Path A: Baseline
        x_base = np.transpose(img_orig, (2, 0, 1)) / 255.0
        x_base_t = torch.from_numpy(np.expand_dims(x_base, 0).astype(np.float32)).to(device)
        
        with torch.no_grad():
            pred_base = torch.sigmoid(model(x_base_t))
            metrics_base = list(map(add, metrics_base, calculate_metrics(y_tensor, pred_base)))
            pred_base_np = (pred_base[0].cpu().numpy().squeeze(0) > 0.5).astype(np.uint8)

        # 2. Path B: Optimized
        od_mask = od_segmenter.process(img_orig)
        
        # === 核心参数调节 ===
        img_opt = harmonize_od_edge(
            img_orig, od_mask, 
            dist_out_px=30,  gamma_out=1.0, 
            dist_in_px=10,   gamma_in=4.0, 
            intensity=60
        )
        # =================
        
        x_opt = np.transpose(img_opt, (2, 0, 1)) / 255.0
        x_opt_t = torch.from_numpy(np.expand_dims(x_opt, 0).astype(np.float32)).to(device)
        
        with torch.no_grad():
            pred_opt = torch.sigmoid(model(x_opt_t))
            metrics_opt = list(map(add, metrics_opt, calculate_metrics(y_tensor, pred_opt)))
            pred_opt_np = (pred_opt[0].cpu().numpy().squeeze(0) > 0.5).astype(np.uint8)

        # ---------------------------
        # Visualization (6列对比)
        # ---------------------------
        # 1. 原图
        col1 = add_label(img_orig, "Original")
        
        # 2. OD Mask (新增)
        col2 = add_label(mask_to_bgr(od_mask), "OD Mask")
        
        # 6. 真值
        col3 = add_label(mask_to_bgr(mask_gt), "GT")
        
        # 3. Baseline 误差
        col4 = add_label(draw_error_map((mask_gt>127), pred_base_np), "Results")
        
        # # 4. 处理后的图
        # col5 = add_label(img_opt, "Processed")
        
        # # 5. Optimized 误差
        # col6 = add_label(draw_error_map((mask_gt>127), pred_opt_np), "Opt Error")


        h = col1.shape[0]
        sep = np.ones((h, 5, 3), dtype=np.uint8) * 100
        
        final_row = np.concatenate([col1, sep, col2, sep, col3, sep, col4], axis=1)
        
        cv2.imwrite(os.path.join(save_path, f"{name}.png"), final_row)

    # 打印结果
    count = len(test_x)
    def print_res(name, m):
        print(f"--- {name} ---")
        print(f"IoU (Jaccard): {m[0]/count:.4f}")
        print(f"F1 (Dice)    : {m[1]/count:.4f}")
        print(f"Accuracy     : {m[4]/count:.4f}")

    print(f"\n>>> [{dataset_name}] RESULTS <<<")
    print_res("Baseline", metrics_base)
    print_res("Optimized", metrics_opt)
    
    diff_iou = (metrics_opt[0] - metrics_base[0]) / count
    print(f"\nIoU Change: {diff_iou:+.4f}")
    print(f"Images saved to: {save_path}\n")

if __name__ == "__main__":
    for ds in TARGET_DATASETS:
         run_comparative_test(ds)