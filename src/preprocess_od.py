import os
import cv2
import torch
import numpy as np
from glob import glob
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from tqdm import tqdm

# --- 路径解析逻辑 ---
# 获取当前脚本文件 (preprocess_od.py) 所在的目录路径 (例如: .../RetinaSegProject/src)
script_dir = os.path.dirname(os.path.abspath(__file__))

# 向上跳一级目录，到达项目根目录 (例如: .../RetinaSegProject)
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# 定义输入和输出路径，它们现在都相对于项目根目录
INPUT_IMAGES_PATH = os.path.join(project_root, 'data', 'images')
OUTPUT_MASKS_PATH = os.path.join(project_root, 'data', 'od_mask')
# --------------------

def generate_od_masks(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载预训练模型
    model_name = "pamixsun/segformer_for_optic_disc_cup_segmentation"
    print(f"正在加载视盘分割模型: {model_name}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # 使用已经解析好的绝对路径进行文件查找
    img_paths = sorted(glob(os.path.join(input_dir, "*"))) # "*" 匹配所有文件
    
    # 添加调试信息，帮助你确认路径是否正确
    print(f"DEBUG: 正在搜索的输入目录: {input_dir}")
    print(f"DEBUG: glob 找到的图片路径: {img_paths}")
    
    print(f"开始处理 {len(img_paths)} 张图片...")
    
    if not img_paths:
        print("警告: 未找到任何图片文件，请检查 INPUT_IMAGES_PATH 是否正确，以及该目录下是否有图片。")
        return

    for img_path in tqdm(img_paths):
        # 读取图片
        image = cv2.imread(img_path)
        if image is None: 
            print(f"警告: 无法读取图像 {img_path}, 可能文件损坏或格式不支持，跳过。")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 预处理
        inputs = processor(image, return_tensors="pt").to(device)
        
        # 推理
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        # 上采样回原始尺寸
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        
        # 获取预测结果 (0:背景, 1:视盘, 2:视杯)
        pred = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        
        # 我们只需要视盘区域 (视盘和视杯) -> 二值化
        od_mask = np.where(pred > 0, 255, 0).astype(np.uint8)
        
        # 保存
        filename = os.path.basename(img_path).split('.')[0] + ".png"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, od_mask)

if __name__ == "__main__":
    generate_od_masks(INPUT_IMAGES_PATH, OUTPUT_MASKS_PATH)