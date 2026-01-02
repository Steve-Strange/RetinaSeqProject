import torch
import numpy as np
import cv2
import os
from torch.utils.data import Dataset

class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path, od_masks_path=None, use_od_input=False):
        self.images_path = images_path
        self.masks_path = masks_path
        self.od_masks_path = od_masks_path
        self.use_od_input = use_od_input
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        # 1. 图像 (假设已经是增强后的 Feature Stack 或 RGB，形状 HxWx3)
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR) 
        # cv2 读取默认为 BGR，如果是特征图，顺序由预处理决定，保持原样即可
        # 如果是 RGB 训练，建议转 RGB: cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = image / 255.0
        
        # 2. 视盘 Mask (如果配置要求输入)
        if self.use_od_input and self.od_masks_path:
            od_mask = cv2.imread(self.od_masks_path[index], cv2.IMREAD_GRAYSCALE)
            od_mask = od_mask / 255.0
            od_mask = np.expand_dims(od_mask, axis=-1) # (H, W, 1)
            # Concatenate -> (H, W, 4)
            image = np.concatenate([image, od_mask], axis=-1)
        
        # Transpose (H, W, C) -> (C, H, W)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.from_numpy(image)

        # 3. 标签 Mask
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0).astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples