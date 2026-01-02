import numpy as np
import cv2
import torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

class ODGenerator:
    """ 使用 SegFormer 生成视盘 Mask """
    def __init__(self, device='cuda'):
        self.device = device
        print("Loading SegFormer for OD extraction...")
        self.processor = AutoImageProcessor.from_pretrained("pamixsun/segformer_for_optic_disc_cup_segmentation")
        self.model = SegformerForSemanticSegmentation.from_pretrained("pamixsun/segformer_for_optic_disc_cup_segmentation")
        self.model.to(self.device)
        self.model.eval()

    def get_mask(self, image_rgb):
        inputs = self.processor(image_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # 上采样回原图尺寸
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=image_rgb.shape[:2], mode="bilinear", align_corners=False
        )
        pred = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        # 0=背景, 1=视盘, 2=视杯。将 1和2 视为 OD
        od_mask = (pred > 0).astype(np.uint8) * 255
        return od_mask

class RetinaPreprocess:
    @staticmethod
    def harmonize_od_color(image_rgb, od_mask):
        """ 简单的颜色同化 (示例实现) """
        if od_mask is None or np.sum(od_mask) == 0:
            return image_rgb
            
        # 简单的 inpainting 或 亮度调整示例
        # 这里仅做简单平滑处理，实际可替换为你更复杂的逻辑
        mask_bool = od_mask > 0
        img_out = image_rgb.copy()
        # 计算非 OD 区域的平均颜色
        mean_bg = cv2.mean(image_rgb, mask=(1-od_mask//255).astype(np.uint8))[:3]
        # 稍微将 OD 区域向背景色混合
        img_out[mask_bool] = (img_out[mask_bool] * 0.7 + np.array(mean_bg) * 0.3).astype(np.uint8)
        return img_out

    @staticmethod
    def get_green_enhanced_stack(image_rgb):
        """ 生成 [Green, CLAHE, TopHat] 特征栈 """
        # 1. Green Channel
        g = image_rgb[:, :, 1]

        # 2. CLAHE on Green
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g_clahe = clahe.apply(g)

        # 3. TopHat (形态学变换 - 突出血管)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(g_clahe, cv2.MORPH_TOPHAT, kernel)

        # Stack (H, W, 3)
        return np.stack([g, g_clahe, tophat], axis=-1)