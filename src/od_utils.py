import torch
import cv2
import numpy as np
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

class ODSegmenter:
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        model_name = "pamixsun/segformer_for_optic_disc_cup_segmentation"
        print(f"[ODSegmenter] Loading model: {model_name}...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print("[ODSegmenter] Model loaded successfully.")
        except Exception as e:
            print(f"[ODSegmenter] Error loading model: {e}")
            self.model = None

    def process(self, image_bgr):
        """
        输入: OpenCV读取的 BGR 图像 (numpy uint8)
        输出: OD Mask (numpy uint8, 0/255)
        """
        if self.model is None or image_bgr is None:
            return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

        # 转换为 RGB
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 预处理
        inputs = self.processor(img_rgb, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # 上采样回原图大小
        upsampled = torch.nn.functional.interpolate(
            logits, 
            size=image_bgr.shape[:2], 
            mode="bilinear", 
            align_corners=False
        )
        
        # 获取预测结果
        pred = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        
        # 提取视盘 (Label 1) 和 视杯 (Label 2)，只要大于0就是OD区域
        od_mask = np.where(pred > 0, 255, 0).astype(np.uint8)
        
        return od_mask