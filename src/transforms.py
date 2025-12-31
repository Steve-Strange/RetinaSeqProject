import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RetinaPreprocess:
    @staticmethod
    def apply_clahe(image):
        """CLAHE 对比度增强"""
        if len(image.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

    @staticmethod
    def harmonize_od_color(image, od_mask):
        """视盘颜色同化：将视盘颜色拉向背景色，消除高亮干扰"""
        if od_mask is None or np.max(od_mask) == 0: 
            return image
        
        image = image.astype(np.float32)
        # 1. 采样背景色
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated_mask = cv2.dilate(od_mask, kernel, iterations=2)
        ring_mask = cv2.subtract(dilated_mask, od_mask)
        
        mean_bg = cv2.mean(image, mask=ring_mask)[:3]
        mean_od = cv2.mean(image, mask=od_mask)[:3]
        
        # 2. 计算增益
        epsilon = 1e-5
        gain_r = np.clip(mean_bg[0] / (mean_od[0] + epsilon), 0.5, 1.2)
        gain_g = np.clip(mean_bg[1] / (mean_od[1] + epsilon), 0.2, 1.1)
        gain_b = np.clip(mean_bg[2] / (mean_od[2] + epsilon), 0.2, 1.1)
        
        # 3. 应用平滑增益
        gain_map = np.ones_like(image, dtype=np.float32)
        mask_float = od_mask.astype(np.float32) / 255.0
        mask_blurred = cv2.GaussianBlur(mask_float, (31, 31), 0)
        
        gain_map[:, :, 0] = 1.0 - mask_blurred * (1.0 - gain_r)
        gain_map[:, :, 1] = 1.0 - mask_blurred * (1.0 - gain_g)
        gain_map[:, :, 2] = 1.0 - mask_blurred * (1.0 - gain_b)
        
        return np.clip(image * gain_map, 0, 255).astype(np.uint8)

    @staticmethod
    def get_green_enhanced_stack(image):
        """生成特征栈: [Green, CLAHE, TopHat]"""
        g = image[:, :, 1]
        
        # Ch1: Raw Green
        
        # Ch2: CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g_clahe = clahe.apply(g)
        
        # Ch3: TopHat (提取微血管)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, kernel)
        tophat = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
        
        return cv2.merge([g, g_clahe, tophat])

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