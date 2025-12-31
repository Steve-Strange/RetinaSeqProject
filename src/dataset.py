import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DriveDataset(Dataset):
    def __init__(self, images_path, vessel_masks_path, od_masks_path, transform=None, use_od_guidance=True):
        self.images_path = images_path
        self.vessel_masks_path = vessel_masks_path
        self.od_masks_path = od_masks_path
        self.transform = transform
        self.use_od_guidance = use_od_guidance

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        # 1. 读取原始图片 (RGB)
        image = cv2.imread(self.images_path[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. 读取血管标签 (Gray)
        mask = cv2.imread(self.vessel_masks_path[index], cv2.IMREAD_GRAYSCALE)
        
        # 3. 读取视盘 Mask (Gray) - 创新点
        img_name = os.path.basename(self.images_path[index]).split('.')[0]
        # 假设文件名对应关系一致，根据实际情况修改扩展名
        od_path = os.path.join(self.od_masks_path, img_name + ".png") 
        
        if self.use_od_guidance and os.path.exists(od_path):
            od_mask = cv2.imread(od_path, cv2.IMREAD_GRAYSCALE)
            if od_mask.shape != image.shape[:2]:
                od_mask = cv2.resize(od_mask, (image.shape[1], image.shape[0]))
        else:
            # 如果没有视盘mask，用全黑代替
            od_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # 4. 数据增强 (Albumentations)
        # 关键：Masks 需要传入列表，以便同时增强 血管mask 和 视盘mask
        if self.transform:
            augmented = self.transform(image=image, masks=[mask, od_mask])
            image = augmented['image']
            mask = augmented['masks'][0]
            od_mask = augmented['masks'][1]

        # 归一化已经在Transform里做了，或者手动做
        # 这里为了演示清晰，假设Transform只做几何变换，手动转Tensor
        # 如果 transform 里有 ToTensorV2 则不需要下面步骤
        
        # 5. 融合输入 (Feature Fusion)
        # image: [3, H, W], od_mask: [1, H, W]
        # 归一化 [0, 1]
        image = image.float() / 255.0
        mask = mask.float() / 255.0
        od_mask = od_mask.float() / 255.0

        if self.use_od_guidance:
            # 增加一个维度变为 [1, H, W]
            od_mask = od_mask.unsqueeze(0) 
            # 拼接: 变成 4 通道 [4, H, W]
            input_tensor = torch.cat([image, od_mask], dim=0)
        else:
            input_tensor = image
            
        mask = mask.unsqueeze(0) # [1, H, W]

        return input_tensor, mask

# 定义增强策略
def get_transforms(img_size=512):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2() # 自动把 HWC -> CHW, 0-255 -> Tensor
    ])