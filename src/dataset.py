import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    """
    专门读取 'final_augmented_dataset' 的数据集类。
    因为数据已经是特征图（Feature Stack）且做过几何变换，
    这里只负责：读取、归一化、拼接视盘Mask。
    """
    def __init__(self, images_path, masks_path, od_masks_path, use_od_guidance=True):
        self.images_path = images_path
        self.masks_path = masks_path
        self.od_masks_path = od_masks_path
        self.use_od_guidance = use_od_guidance

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        # 1. 读取 Feature Stack (看起来像RGB的特征图)
        # OpenCV 读取顺序是 BGR。只要我们生成和读取都用 OpenCV，通道顺序就是一致的，不需要转 RGB
        img_path = self.images_path[index]
        image = cv2.imread(img_path) # Shape: [512, 512, 3]
        
        # 2. 读取 血管 Mask
        # 根据文件名找对应的 mask
        fname = os.path.basename(img_path)
        mask_path = os.path.join(self.masks_path, fname)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 3. 读取 OD Mask
        od_p = os.path.join(self.od_masks_path, fname)
        if os.path.exists(od_p):
            od_mask = cv2.imread(od_p, cv2.IMREAD_GRAYSCALE)
        else:
            # 兜底：如果没有OD mask，给一个全黑的
            od_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # 4. 归一化 (0-255 -> 0.0-1.0)
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        od_mask = od_mask.astype(np.float32) / 255.0
        
        # 5. 转 Tensor (HWC -> CHW)
        # image: [3, 512, 512]
        image = torch.from_numpy(image).permute(2, 0, 1)
        # mask: [1, 512, 512]
        mask = torch.from_numpy(mask).unsqueeze(0)
        # od_mask: [1, 512, 512]
        od_mask = torch.from_numpy(od_mask).unsqueeze(0)
        
        # 6. 核心创新点：拼接 OD Mask
        if self.use_od_guidance:
            # 拼接后变成 [4, 512, 512]
            input_tensor = torch.cat([image, od_mask], dim=0) 
        else:
            input_tensor = image

        return input_tensor, mask