import os
import cv2
import torch
import numpy as np
from glob import glob
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from tqdm import tqdm

# 获取绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # 回退一级
INPUT_DIR = os.path.join(project_root, "data", "unified_data", "images")
OUTPUT_DIR = os.path.join(project_root, "data", "unified_data", "od_masks")

def generate_masks():
    if not os.path.exists(INPUT_DIR):
        print(f"错误: 找不到数据目录 {INPUT_DIR}，请先运行 prepare_data.py")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_name = "pamixsun/segformer_for_optic_disc_cup_segmentation"
    print("加载视盘分割模型...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    img_paths = sorted(glob(os.path.join(INPUT_DIR, "*.png")))
    print(f"开始生成视盘Mask，共 {len(img_paths)} 张...")

    for path in tqdm(img_paths):
        img = cv2.imread(path)
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        inputs = processor(img_rgb, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        
        upsampled = torch.nn.functional.interpolate(
            logits, size=img.shape[:2], mode="bilinear", align_corners=False
        )
        pred = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        
        # 提取视盘 (Label 1) 和 视杯 (Label 2)
        od_mask = np.where(pred > 0, 255, 0).astype(np.uint8)
        
        save_name = os.path.basename(path)
        cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), od_mask)

if __name__ == "__main__":
    generate_masks()