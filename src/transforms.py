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
        """
        【边缘光晕修复版】视盘颜色同化
        针对问题：边缘出现黑色“甜甜圈”阴影。
        解决方案：
        1. 构建一个专门的“边缘光晕层 (Halo Layer)”。
        2. 引入动态安全底板 (Dynamic Safety Floor)：
           - 中心区域：允许压暗 (Floor = 0.8 * Original)
           - 边缘区域：强制提亮 (Floor = 1.1 * Original)
        3. 扩大保护范围：通过膨胀操作，确保覆盖到视盘外的暗环。
        """
        if od_mask is None or np.max(od_mask) == 0: 
            return image
        
        image = image.astype(np.float32)
        
        # === 1. 定义区域 (Zone Definition) ===
        # 核心区 (Core): 需要压暗的地方
        shrink_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30)) # 腐蚀得更多一点，只搞中心
        core_mask = cv2.erode(od_mask, shrink_kernel, iterations=1)
        
        # 边缘区 (Edge): 需要提亮/保护的地方
        # 逻辑：比原 Mask 再大一圈 (Dilate)，减去 核心区
        expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30)) # 扩大范围
        dilated_mask = cv2.dilate(od_mask, expand_kernel, iterations=1)
        
        # 制作边缘权重的 Mask (环形)
        edge_mask = cv2.subtract(dilated_mask, core_mask)
        
        # 柔化 Mask
        mask_float = core_mask.astype(np.float32) / 255.0
        mask_blurred = cv2.GaussianBlur(mask_float, (51, 51), 0)
        mask_blurred = np.expand_dims(mask_blurred, axis=2)
        
        # 柔化边缘权重图
        edge_float = edge_mask.astype(np.float32) / 255.0
        edge_blurred = cv2.GaussianBlur(edge_float, (41, 41), 0)
        edge_blurred = np.expand_dims(edge_blurred, axis=2)

        # === 2. 采样与计算比率 ===
        # 背景采样区 (在膨胀区之外)
        bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        bg_zone = cv2.bitwise_not(cv2.dilate(dilated_mask, bg_kernel, iterations=2))
        
        mean_bg = cv2.mean(image, mask=bg_zone)[:3]
        mean_od = cv2.mean(image, mask=core_mask)[:3]
        
        corrected = np.copy(image)
        epsilon = 1e-5

        # === 3. 通道处理 (同前，但更温和) ===
        
        # R 通道: 提亮保红
        ratio_r = mean_bg[0] / (mean_od[0] + epsilon)
        ratio_r = np.clip(ratio_r, 1.05, 1.3) # 稍微激进一点提亮 R
        corrected[:, :, 0] *= ratio_r
        
        # G/B 通道: 压暗
        for ch in [1, 2]:
            raw_ratio = mean_bg[ch] / (mean_od[ch] + epsilon)
            target_ratio = np.clip(raw_ratio * 1.4, 0.75, 1.05) # 下限提到 0.75
            
            channel_data = image[:, :, ch]
            norm_data = channel_data / 255.0
            attenuation = 1.0 - (norm_data * (1.0 - target_ratio))
            corrected[:, :, ch] *= attenuation

        # === 4. 融合与动态兜底 (关键修改) ===
        
        # 初步融合
        harmonized = image * (1 - mask_blurred) + corrected * mask_blurred
        
        # 【动态安全底板】
        # 逻辑：如果像素在边缘区(edge_blurred高)，底板就是 image * 1.1 (提亮)
        #       如果像素在其他区，底板就是 image * 0.85 (允许压暗)
        
        # floor_factor 从 0.85 平滑过渡到 1.15
        # 边缘越强，factor 越大
        floor_factor = 0.85 + (edge_blurred * 0.3) 
        
        safety_floor = image * floor_factor
        
        # 取最大值：这就保证了边缘绝对不会黑，反而会被 edge_blurred 撑起来
        final_result = np.maximum(harmonized, safety_floor)
        
        return np.clip(final_result, 0, 255).astype(np.uint8)
    
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