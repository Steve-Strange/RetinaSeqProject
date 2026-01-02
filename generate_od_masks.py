import os
import cv2
import torch
import numpy as np
import gzip
from glob import glob
from tqdm import tqdm
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

# 1. 设置路径 (请根据你的实际 DRIVE 数据路径修改)
DATA_PATH = "data/DRIVE/" 
# 输出路径
OD_MASK_PATH = os.path.join(DATA_PATH, "od_masks") 

# 2. 加载 SegFormer 模型
print("Loading SegFormer model...")
model_name = "pamixsun/segformer_for_optic_disc_cup_segmentation"
processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
model = SegformerForSemanticSegmentation.from_pretrained(model_name).eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def get_od_mask(image_path):
    # 读取图像 (兼容普通图片和 .ppm.gz)
    if image_path.endswith(".gz"):
        with gzip.open(image_path, "rb") as f:
            data = f.read()
        img_array = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(image_path)
        if image is None: return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 模型需要 RGB

    # 推理
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # 上采样回原图尺寸
    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=image.shape[:2], mode="bilinear", align_corners=False
    )
    
    # 获取预测结果: 0=背景, 1=视盘, 2=视杯
    pred = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    
    # 生成二值 Mask：我们将 视盘(1) 和 视杯(2) 都视为视盘区域
    # 也可以只取视盘，取决于具体定义，通常 OD 包含 Cup
    od_mask = (pred > 0).astype(np.uint8) * 255 
    
    return od_mask

def process_set(subset_name):
    # 处理 training 或 test 文件夹
    img_dir = os.path.join(DATA_PATH, subset_name, "images")
    save_dir = os.path.join(DATA_PATH, subset_name, "od_masks")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    images = sorted(glob(os.path.join(img_dir, "*"))) # 读取 .tif 或 .png
    
    print(f"Processing {subset_name}...")
    for img_path in tqdm(images):
        name = os.path.basename(img_path).split(".")[0]
        mask = get_od_mask(img_path)
        
        if mask is not None:
            # 保存为 png
            cv2.imwrite(os.path.join(save_dir, f"{name}.png"), mask)

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        process_set("training")
        process_set("test")
        print("OD Mask generation complete.")
    else:
        print(f"Path not found: {DATA_PATH}")