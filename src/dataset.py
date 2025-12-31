import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from src.transforms import RetinaPreprocess

class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path, od_masks_path, transform=None, use_od_guidance=True):
        self.images_path = images_path
        self.masks_path = masks_path
        self.od_masks_path = od_masks_path
        self.transform = transform
        self.use_od_guidance = use_od_guidance

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        # 1. 读取
        img_path = self.images_path[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask_path = self.masks_path[index]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 读取 OD Mask
        # ... (和之前一样) ...
        
        # ==========================================
        # 颜色通道设计的核心修改点
        # ==========================================
        
        # 1. 先做视盘压暗 (还是在 RGB 空间做比较好)
        if self.use_od_guidance:
             image = RetinaPreprocess.attenuate_od_brightness(image, od_mask)
        
        # 2. 【关键】替换 RGB 为 [G, CLAHE_G, Gamma_G]
        # 这一步把 3通道的RGB 变成了 3通道的绿色增强集
        image_engineered = RetinaPreprocess.get_green_enhanced_stack(image)

        # 3. Albumentations 增强
        if self.transform:
            # 注意：Albumentations 会把 image 当作 3通道图处理，
            # 只要我们 feed 进去的是 (H,W,3)，它就能正常做几何变换，
            # 旋转、缩放对是不是 RGB 颜色没有偏见。
            augmented = self.transform(image=image_engineered, mask=mask, mask0=od_mask)
            
            # 拿到的是 Tensor: [3, H, W]
            image_tensor = augmented['image'] 
            mask = augmented['mask']
            od_mask = augmented['mask0']
        
        # 4. 构造最终输入
        # image_tensor 已经是 [3, H, W] 了 (即 G, CLAHE, Gamma)
        mask = mask.float() / 255.0
        od_mask = od_mask.float() / 255.0
        
        if self.use_od_guidance:
            if not isinstance(od_mask, torch.Tensor):
                od_mask = torch.from_numpy(od_mask)
            
            od_mask = od_mask.unsqueeze(0) # [1, H, W]
            
            # 拼接: [3通道绿色特征] + [1通道OD Mask] = [4通道输入]
            input_tensor = torch.cat([image_tensor, od_mask], dim=0) 
        else:
            input_tensor = image_tensor

        mask = mask.unsqueeze(0)
        return input_tensor, mask