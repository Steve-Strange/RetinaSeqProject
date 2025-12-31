import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
import albumentations as A
from src.transforms import RetinaPreprocess

# === 配置 ===
INPUT_DIR = "./data/unified_data"
OUTPUT_DIR = "./data/final_augmented_dataset" # 改个名，区分旧数据
AUGMENT_TIMES = 10  # 10倍扩充

# === 1. 更新后的增强策略 (去噪点 + 仿射变换) ===
def get_offline_transforms(img_size=512):
    """
    离线增强策略 (微调版 - 更加温和)
    """
    return A.Compose([
        # 1. 基础操作 (无损)
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # 2. 仿射变换 (幅度大幅降低)
        A.Affine(
            scale=(0.95, 1.00),   # 缩放仅限 ±5%
            translate_percent=(0.00, 0.00), # 平移仅限 2%
            rotate=(-90, 90),     # 旋转仅限 ±10度
            shear=(-2, 2),        # 剪切仅限 ±2度 (之前是10，太歪了)
            mode=cv2.BORDER_CONSTANT, 
            cval=0, 
            p=0.8
        ),
        
        # 3. 透视变换 (幅度大幅降低)
        # 模拟轻微的摄像头角度不正，而不是鱼眼镜头
        A.Perspective(scale=(0.01, 0.02), keep_size=True, p=0.3), # scale降到0.02

        # 4. 光照微调 (无噪点)
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.8),
            A.RandomGamma(gamma_limit=(95, 105), p=0.5),
        ], p=0.5),
        
    ], additional_targets={'mask0': 'mask'})

def main():
    # 1. 创建目录
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "masks"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "od_masks"), exist_ok=True)
    
    # 2. 获取路径
    img_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "images", "*.png")))
    mask_path_root = os.path.join(INPUT_DIR, "masks")
    od_path_root = os.path.join(INPUT_DIR, "od_masks")
    
    transform = get_offline_transforms(512)
    
    print(f"开始生成最终数据集...")
    print(f"策略: OD同化 -> [Green, CLAHE, TopHat]特征栈 -> 仿射增强")
    print(f"原图 {len(img_paths)} 张 -> 目标 {len(img_paths) * AUGMENT_TIMES} 张")
    
    for img_path in tqdm(img_paths):
        # 读取基础数据
        fname = os.path.basename(img_path)
        base_name = fname.split('.')[0]
        
        # 读图 (BGR -> RGB)
        bgr = cv2.imread(img_path)
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # 读 Masks
        mask = cv2.imread(os.path.join(mask_path_root, fname), cv2.IMREAD_GRAYSCALE)
        
        od_p = os.path.join(od_path_root, fname)
        if os.path.exists(od_p):
            od_mask = cv2.imread(od_p, cv2.IMREAD_GRAYSCALE)
        else:
            od_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
        # === 步骤1: 物理预处理 (确定性) ===
        
        # A. 视盘颜色同化 (Color Harmonization)
        # 这一步必须在 RGB 空间做
        image = RetinaPreprocess.harmonize_od_color(image, od_mask)
        
        # B. 生成特征栈 (Feature Stacking)
        # 输入 RGB，输出 [Green, CLAHE, TopHat] 的 3通道图
        # 注意：这一步之后，图像不再是 RGB 颜色空间了！
        image_engineered = RetinaPreprocess.get_green_enhanced_stack(image)
        
        # === 步骤2: 循环增强 ===
        for i in range(AUGMENT_TIMES):
            # 对特征栈进行几何变换
            # Albumentations 不在乎你是不是真 RGB，只要是 3通道 它都能转
            augmented = transform(image=image_engineered, mask=mask, mask0=od_mask)
            
            aug_img = augmented['image']
            aug_mask = augmented['mask']
            aug_od = augmented['mask0']
            
            # === 保存 ===
            # 注意：这里不需要再转 BGR 了！
            # 我们的 aug_img 通道顺序是 [G, CLAHE, TopHat]
            # 直接保存，cv2.imwrite 会按顺序写入文件的 Ch0, Ch1, Ch2
            # 训练时 cv2.imread 读出来还是 Ch0, Ch1, Ch2，完美对应。
            
            new_name = f"{base_name}_aug_{i}.png"
            
            cv2.imwrite(os.path.join(OUTPUT_DIR, "images", new_name), aug_img)
            cv2.imwrite(os.path.join(OUTPUT_DIR, "masks", new_name), aug_mask)
            cv2.imwrite(os.path.join(OUTPUT_DIR, "od_masks", new_name), aug_od)

    print(f"\n增强完成！数据已保存到 {OUTPUT_DIR}")
    print("注意：生成的图片看起来颜色会很怪（紫绿色），这是正常的特征图。")

if __name__ == "__main__":
    main()