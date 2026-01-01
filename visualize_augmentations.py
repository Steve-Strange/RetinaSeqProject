import os
import cv2
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
# 确保引用的是刚才修改过的 src/transforms.py
from src.transforms import RetinaPreprocess, get_offline_transforms

# === 配置 ===
DATA_DIR = "./data/unified_data"
SAVE_DIR = "./debug_results_pipeline" # 结果保存在这里
NUM_SAMPLES = 5  # 展示几组图片

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 获取所有图片路径
    all_images = sorted(glob.glob(os.path.join(DATA_DIR, "images", "*.png")))
    
    if len(all_images) == 0:
        print("错误: 找不到数据，请检查 DATA_DIR 路径")
        return

    # 随机选几张
    sample_paths = random.sample(all_images, min(NUM_SAMPLES, len(all_images)))
    
    # 获取我们刚修改过的、温和的 Transform
    transform = get_offline_transforms(512)
    
    print(f"正在生成 {len(sample_paths)} 组可视化对比图...")
    print("流程: 原图 -> OD颜色修复 -> 特征栈(伪彩) -> 几何增强(微调)")

    for idx, img_path in enumerate(sample_paths):
        fname = os.path.basename(img_path)
        
        # 1. 读取基础数据
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        mask_path = os.path.join(DATA_DIR, "masks", fname)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        od_path = os.path.join(DATA_DIR, "od_masks", fname)
        if os.path.exists(od_path):
            od_mask = cv2.imread(od_path, cv2.IMREAD_GRAYSCALE)
        else:
            od_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)

        # 2. 颜色工程处理
        # 步骤 A: 视盘颜色同化 (OD Color Harmonization)
        rgb_harmonized = RetinaPreprocess.harmonize_od_color(rgb, od_mask)
        
        # 步骤 B: 生成绿色增强栈 [Green, CLAHE, TopHat]
        # 结果是 3通道特征图
        feature_stack = RetinaPreprocess.get_green_enhanced_stack(rgb_harmonized)
        
        # 3. 几何增强 (Augmentation)
        # 对 "特征栈" 进行变形
        augmented = transform(image=feature_stack, mask=mask, mask0=od_mask)
        aug_img = augmented['image']
        aug_mask = augmented['mask']
        
        # 4. 绘图 (Matplotlib)
        # 创建一个 2行4列 的大图
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        fig.suptitle(f"Sample {idx+1}: {fname}", fontsize=20, weight='bold')
        
        # --- Row 1: 预处理前后的 RGB 对比 ---
        
        # 1-1. 原图
        axes[0,0].imshow(rgb)
        axes[0,0].set_title("1. Original RGB", fontsize=14)
        axes[0,0].axis('off')
        
        # 1-2. OD Mask
        axes[0,1].imshow(od_mask, cmap='gray')
        axes[0,1].set_title("2. Optic Disc Mask", fontsize=14)
        axes[0,1].axis('off')
        
        # 1-3. 修复后的图 (注意看视盘颜色变红了)
        axes[0,2].imshow(rgb_harmonized)
        axes[0,2].set_title("3. OD Harmonized (Color Fixed)", fontsize=14)
        axes[0,2].axis('off')
        
        # 1-4. 血管标签
        axes[0,3].imshow(mask, cmap='gray')
        axes[0,3].set_title("4. Ground Truth", fontsize=14)
        axes[0,3].axis('off')

        # --- Row 2: 特征工程与最终输出 ---
        
        # 2-1. 特征通道1: 原始绿色
        axes[1,0].imshow(feature_stack[:,:,0], cmap='gray')
        axes[1,0].set_title("5. Feature Ch1: Raw Green\n(From Harmonized)", fontsize=14)
        axes[1,0].axis('off')
        
        # 2-2. 特征通道3: TopHat (这个最厉害，专门看细血管)
        axes[1,1].imshow(feature_stack[:,:,2], cmap='gray')
        axes[1,1].set_title("6. Feature Ch3: Morph TopHat\n(Micro-Vessels)", fontsize=14)
        axes[1,1].axis('off')
        
        # 2-3. 组合特征图 (伪彩色)
        axes[1,2].imshow(feature_stack) 
        axes[1,2].set_title("7. Stacked Input (False Color)\n[G, CLAHE, TopHat]", fontsize=14)
        axes[1,2].axis('off')
        
        # 2-4. 最终增强 (带几何变换)
        # 也就是模型真正吃到的数据
        axes[1,3].imshow(aug_img)
        axes[1,3].set_title("8. Final Augmented Input\n(Mild Affine + Clean)", fontsize=14)
        axes[1,3].axis('off')
        
        plt.tight_layout()
        save_p = os.path.join(SAVE_DIR, f"pipeline_vis_{idx}.png")
        plt.savefig(save_p, dpi=150) # 提高一点分辨率
        plt.close()
        print(f"  已保存: {save_p}")

    print(f"\n可视化完成！请打开文件夹 {SAVE_DIR} 查看图片。")
    print("这次的形变应该是很微小的，且没有噪点。")

if __name__ == "__main__":
    main()