import cv2
import glob
import os
import numpy as np

DATA_DIR = "./data/final_augmented_dataset"

def check():
    img_path = glob.glob(os.path.join(DATA_DIR, "images", "*.png"))[0]
    mask_path = os.path.join(DATA_DIR, "masks", os.path.basename(img_path))
    
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # 读取时不转换通道
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    print(f"检查图片: {img_path}")
    print(f"图片形状: {img.shape}")
    print(f"图片数值范围: min={img.min()}, max={img.max()}, mean={img.mean():.2f}")
    
    print(f"检查Mask: {mask_path}")
    print(f"Mask数值范围: min={mask.min()}, max={mask.max()}")
    print(f"Mask非零像素占比: {(mask>0).sum() / mask.size :.4f}")

    if img.max() == 0:
        print("❌ 警告：输入图片是全黑的！生成脚本有问题。")
    elif mask.max() == 0:
        print("❌ 警告：Mask是全黑的！")
    else:
        print("✅ 数据看起来正常。")

if __name__ == "__main__":
    check()