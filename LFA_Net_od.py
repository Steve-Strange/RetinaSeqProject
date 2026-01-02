# %%
# Data Augmentation
import sys
import os

if os.path.exists("config.py"):
    print("Warning: Found 'config.py' in the current directory. Please rename it to avoid conflicts with torch.")
if os.path.exists("torch.py"):
    print("Warning: Found 'torch.py' in the current directory. Please rename it to avoid conflicts with torch.")

# 优先导入 torch
import torch
print(f"Torch version: {torch.__version__}")

import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate, RandomBrightnessContrast, RandomCrop, RandomRotate90, RandomGridShuffle

""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

'''加载数据：原图+标签'''
def load_data(path):
    train_x = sorted(glob(os.path.join(path, "training", "images", "*.tif")))
    train_y = sorted(glob(os.path.join(path, "training", "1st_manual", "*.gif")))

    test_x = sorted(glob(os.path.join(path, "test", "images", "*.tif")))
    test_y = sorted(glob(os.path.join(path, "test", "1st_manual", "*.gif")))

    return (train_x, train_y), (test_x, test_y)

'''
增强数据
对图像及其对应mask数据增强
'''
def load_data_with_od(path):
    # 加载原图、血管Mask、视盘Mask
    train_x = sorted(glob(os.path.join(path, "training", "images", "*.tif")))
    train_y = sorted(glob(os.path.join(path, "training", "1st_manual", "*.gif")))
    train_od = sorted(glob(os.path.join(path, "training", "od_masks", "*.png"))) # 新增

    test_x = sorted(glob(os.path.join(path, "test", "images", "*.tif")))
    test_y = sorted(glob(os.path.join(path, "test", "1st_manual", "*.gif")))
    test_od = sorted(glob(os.path.join(path, "test", "od_masks", "*.png"))) # 新增

    return (train_x, train_y, train_od), (test_x, test_y, test_od)

def augment_data_triple_triple(images, masks, od_masks, save_path, augment=True):
    size = (560, 560)
    
    # 确保保存路径存在
    create_dir(os.path.join(save_path, "image"))
    create_dir(os.path.join(save_path, "mask"))
    create_dir(os.path.join(save_path, "od_mask")) # 新增

    for idx, (x, y, od) in tqdm(enumerate(zip(images, masks, od_masks)), total=len(images)):
        name = x.split(os.sep)[-1].split(".")[0]

        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]
        od = cv2.imread(od, cv2.IMREAD_GRAYSCALE) # 读取视盘Mask

        if augment:
            transformations = [
                HorizontalFlip(p=1.0),
                VerticalFlip(p=1.0),
                Rotate(limit=45, p=1.0),
                RandomBrightnessContrast(p=0.6),
                RandomCrop(300, 300, p=0.8),
                RandomRotate90(p=1.0),
                RandomGridShuffle(p=0.7)
            ]
            
            # 使用列表保存增强结果
            X, Y, OD = [x], [y], [od]

            for aug in transformations:
                # Albumentations 支持多 Mask: use masks=[mask1, mask2]
                augmented = aug(image=x, masks=[y, od]) 
                X.append(augmented["image"])
                Y.append(augmented["masks"][0])
                OD.append(augmented["masks"][1])
        else:
            X, Y, OD = [x], [y], [od]

        index = 0
        for i, m, o in zip(X, Y, OD):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)
            o = cv2.resize(o, size)

            tmp_name = f"{name}_{index}.png"
            
            cv2.imwrite(os.path.join(save_path, "image", tmp_name), i)
            cv2.imwrite(os.path.join(save_path, "mask", tmp_name), m)
            cv2.imwrite(os.path.join(save_path, "od_mask", tmp_name), o) # 保存视盘Mask
            index += 1

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the data """
    # 修改为相对路径
    data_path = "data/DRIVE/"
    
    if os.path.exists(data_path):
        (train_x, train_y, train_od), (test_x, test_y, test_od) = load_data_with_od(data_path)

        print(f"Train: {len(train_x)} - {len(train_y)} - {len(train_od)}")
        print(f"Test: {len(test_x)} - {len(test_y)} - {len(test_od)}")

        """ Create directories to save the augmented data """
        create_dir("working/new_data/train/image/")
        create_dir("working/new_data/train/mask/")
        create_dir("working/new_data/test/image/")
        create_dir("working/new_data/test/mask/")

        """ Data augmentation """
        # 取消注释以运行数据增强
        augment_data_triple_triple(train_x, train_y, train_od, "working/new_data/train/", augment=True)
        augment_data_triple_triple(test_x, test_y, test_od, "working/new_data/test/", augment=False)
    else:
        print(f"Data path not found: {data_path}")

# %%
# Model (LFA-Net)
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalModulation(nn.Module):
    def __init__(self, in_channels, gamma=2.0, alpha=0.25):
        super(FocalModulation, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mean = self.gap(x)
        max_val = self.gmp(x)
        modulation = (max_val - mean) * self.alpha
        modulation = self.conv(modulation)
        modulation = self.sigmoid(modulation)
        scaled_inputs = x * modulation
        outputs = torch.pow(scaled_inputs, self.gamma)
        return outputs

class FocalModulationContextAggregation(nn.Module):
    def __init__(self, in_channels, filters):
        super(FocalModulationContextAggregation, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, filters, kernel_size=1)
        self.relu2 = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv_ctx = nn.Conv2d(filters, filters, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.focal_mod = FocalModulation(filters)

    def forward(self, x):
        c1 = self.relu1(self.conv1(x))
        c2 = self.relu2(self.conv2(x))
        
        global_context = self.gap(c2)
        global_context = self.sigmoid(self.conv_ctx(global_context))
        global_context = c1 * global_context
        
        fm = self.focal_mod(global_context)
        return torch.cat([c1, fm], dim=1)

class VisionMambaInspired(nn.Module):
    def __init__(self, dim, dropout_rate=0.1):
        super(VisionMambaInspired, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mixer = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mixer = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        
        shortcut = x
        x_perm = x.permute(0, 2, 3, 1)
        x_norm = self.norm1(x_perm).permute(0, 3, 1, 2)
        x_tm = self.token_mixer(x_norm) + shortcut
        
        shortcut = x_tm
        x_perm = x_tm.permute(0, 2, 3, 1)
        x_norm = self.norm2(x_perm)
        x_cm = self.channel_mixer(x_norm)
        x_cm = x_cm.permute(0, 3, 1, 2)
        
        return x_cm + shortcut

class LiteFusionAttention(nn.Module):
    def __init__(self, in_channels, filters):
        super(LiteFusionAttention, self).__init__()
        self.proj1 = nn.Conv2d(in_channels, filters, kernel_size=1)
        self.norm = nn.LayerNorm(filters)
        self.conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.fmca = FocalModulationContextAggregation(filters, filters)
        self.proj2 = nn.Conv2d(2 * filters, filters, kernel_size=1)
        self.vm = VisionMambaInspired(filters)
        
        self.res_proj = nn.Conv2d(in_channels, filters, kernel_size=1) if in_channels != filters else nn.Identity()

    def forward(self, x):
        input_tensor = self.proj1(x)
        
        x_perm = input_tensor.permute(0, 2, 3, 1)
        x_norm = self.norm(x_perm).permute(0, 3, 1, 2)
        
        x_conv = self.conv(x_norm)
        x_fmca = self.fmca(x_conv)
        x_proj = self.proj2(x_fmca)
        
        res = self.res_proj(x) if isinstance(self.res_proj, nn.Conv2d) else input_tensor
        out = x_proj + res
        
        out = self.vm(out)
        return out

class RA_AttentionBlock(nn.Module):
    def __init__(self, in_channels, n_classes, k):
        super(RA_AttentionBlock, self).__init__()
        self.k = k
        self.n_classes = n_classes
        self.conv = nn.Conv2d(in_channels, k * n_classes, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(k * n_classes)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        b, c, h, w = x.shape
        f = self.relu(self.bn(self.conv(x)))
        
        x1 = self.gmp(f)
        x2 = self.gap(f)
        x_mul = x1 * x2
        
        x_reshape = x_mul.view(b, self.n_classes, self.k)
        s = torch.mean(x_reshape, dim=-1, keepdim=False)
        
        f_perm = f.permute(0, 2, 3, 1)
        f_reshape = f_perm.view(b, h, w, self.n_classes, self.k)
        f_mean = torch.mean(f_reshape, dim=-1, keepdim=False)
        
        s_expanded = s.view(b, 1, 1, self.n_classes)
        x_weighted = f_mean * s_expanded
        
        m = torch.mean(x_weighted, dim=-1, keepdim=True)
        m = m.permute(0, 3, 1, 2)
        
        semantic = x * m
        return semantic

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super(ConvBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv3x3_dilated = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        c1 = self.conv1x1(x)
        c3 = self.conv3x3(x)
        c3d = self.conv3x3_dilated(x)
        out = c1 + c3 + c3d
        out = self.leaky_relu(out)
        return out

class build_unet(nn.Module):
    def __init__(self, input_channels=4, num_classes=1, feature_scale=2, dropout=0.5):
        super(build_unet, self).__init__()
        filters = [int(x / feature_scale) for x in [16, 32, 64]]
        
        self.conv1 = ConvBlock(input_channels, filters[0], dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(filters[0])
        
        self.conv2 = ConvBlock(filters[0], filters[1], dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(filters[1])
        
        self.conv3 = ConvBlock(filters[1], filters[2], dropout)
        self.pool3 = nn.MaxPool2d(2)
        self.bn3 = nn.BatchNorm2d(filters[2])
        
        self.lfa = LiteFusionAttention(filters[2], filters=32)
        
        lfa_out_channels = 32 
        self.att1 = RA_AttentionBlock(lfa_out_channels, 1, 16)
        
        self.up1 = nn.ConvTranspose2d(lfa_out_channels * 2, filters[2], kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.att2 = RA_AttentionBlock(filters[1], 1, 16)
        self.dec_conv1 = nn.Conv2d(filters[1] + filters[2], filters[2], kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        self.up2 = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.att3 = RA_AttentionBlock(filters[0], 1, 16)
        self.dec_conv2 = nn.Conv2d(filters[0] + filters[2], filters[2], kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        self.up3 = nn.ConvTranspose2d(filters[2], filters[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv3 = nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        
        self.final = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.bn1(self.pool1(c1))
        
        c2 = self.conv2(p1)
        p2 = self.bn2(self.pool2(c2))
        
        c3 = self.conv3(p2)
        p3 = self.bn3(self.pool3(c3))
        
        lfa = self.lfa(p3)
        
        att1 = self.att1(lfa)
        fused = torch.cat([att1, lfa], dim=1)
        
        d1 = self.up1(fused)
        if d1.size() != c2.size():
             d1 = F.interpolate(d1, size=c2.shape[2:], mode='bilinear', align_corners=True)
             
        att2 = self.att2(c2)
        d1 = torch.cat([att2, d1], dim=1)
        d1 = self.relu1(self.dec_conv1(d1))
        
        d2 = self.up2(d1)
        if d2.size() != c1.size():
             d2 = F.interpolate(d2, size=c1.shape[2:], mode='bilinear', align_corners=True)

        att3 = self.att3(c1)
        d2 = torch.cat([att3, d2], dim=1)
        d2 = self.relu2(self.dec_conv2(d2))
        
        d3 = self.up3(d2)
        if d3.size() != x.size():
             d3 = F.interpolate(d3, size=x.shape[2:], mode='bilinear', align_corners=True)
             
        d3 = self.relu3(self.dec_conv3(d3))
        
        out = self.final(d3)
        return out

if __name__ == "__main__":
    x = torch.randn((2, 4, 560, 560))
    f = build_unet()
    y = f(x)
    print(y.shape)

# %%
# Data
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path, od_masks_path):
        self.images_path = images_path
        self.masks_path = masks_path
        self.od_masks_path = od_masks_path
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = image / 255.0 
        # (H, W, 3)
        
        """ Reading OD mask """
        od_mask = cv2.imread(self.od_masks_path[index], cv2.IMREAD_GRAYSCALE)
        od_mask = od_mask / 255.0
        od_mask = np.expand_dims(od_mask, axis=-1) # (H, W, 1)
        
        """ Concatenate: 融合 Image 和 OD Mask """
        # 结果形状: (H, W, 4) ->  transpose -> (4, H, W)
        input_tensor = np.concatenate([image, od_mask], axis=-1)
        
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = input_tensor.astype(np.float32)
        input_tensor = torch.from_numpy(input_tensor)

        """ Reading label mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return input_tensor, mask

    def __len__(self):
        return self.n_samples

# %%
# Loss
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

# %%
# Utils
import os
import time
import random
import numpy as np
import cv2
import torch

""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

""" Create a directory. """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Calculate the time taken """
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# %%

# Train
import os
import time
from glob import glob
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

# from data import DriveDataset
# from model import build_unet
# from loss import DiceLoss, DiceBCELoss
# from utils import seeding, create_dir, epoch_time

'''训练深度学习模型'''
def train(model, loader, optimizer, loss_fn, device, show_images=False):
    epoch_loss = 0.0

    model.train()
    for i, (x, y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        # if i == 1 and show_images:
        #    # 显示第一批图像和掩码
        #     img = x[0].cpu().numpy()  # 假设图像是CHW格式
        #     img = np.transpose(img, (1, 2, 0))  # 转换为HWC格式
        #     img = img[..., ::-1]  # 将BGR转换为RGB
        #     mask = y[0].cpu().numpy()  # 假设掩码是CHW格式
        #     mask = np.transpose(mask, (1, 2, 0))  # 转换为HWC格式

        #     plt.figure(figsize=(12, 6))

        #     plt.subplot(1, 2, 1)
        #     plt.imshow(img)
        #     plt.title("Sample Image")

        #     plt.subplot(1, 2, 2)
        #     plt.imshow(mask, cmap='gray')
        #     plt.title("Corresponding Mask")

        #     plt.show()

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    #计算整个epoch的平均损失
    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
    return epoch_loss

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files")

    """ Load dataset """
    train_x = sorted(glob("working/new_data/train/image/*"))
    train_y = sorted(glob("working/new_data/train/mask/*"))
    train_od = sorted(glob("working/new_data/train/od_mask/*")) # 加载增强后的 OD Mask

    valid_x = sorted(glob("working/new_data/test/image/*"))
    valid_y = sorted(glob("working/new_data/test/mask/*"))
    valid_od = sorted(glob("working/new_data/test/od_mask/*"))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ Hyperparameters """
    H = 560
    W = 560
    size = (H, W)
    batch_size = 64
    num_epochs = 100   
    lr = 1e-3
    checkpoint_path = "working/files/drive_checkpoint.pth"

    """ Dataset and loader """
    train_dataset = DriveDataset(train_x, train_y, train_od)
    valid_dataset = DriveDataset(valid_x, valid_y, valid_od)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   ## GTX 1060 6GB
    model = build_unet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    loss_fn = DiceBCELoss()

    """ Training the model """
    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()

        #该轮训练的平均损失值 train_loss
        train_loss = train(model, train_loader, optimizer, loss_fn, device, show_images=True)
        #返回验证损失 valid_loss
        valid_loss = evaluate(model, valid_loader, loss_fn, device)

        """ Saving the model """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)

# %%


# %%
# Test
import os, time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix

# 假设 build_unet, create_dir, seeding 等函数已经在上下文中定义
# 如果是在 notebook 中，确保上面的 cell 已经运行

def calculate_metrics(y_true, y_pred):
    """ 计算指标 (保持不变) """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    score_f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
    score_jaccard = tp / (tp + fp + fn + 1e-6)
    score_recall = tp / (tp + fn + 1e-6)
    score_specificity = tn / (tn + fp + 1e-6)
    score_precision = tp / (tp + fp + 1e-6)
    score_acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_specificity]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Folders """
    create_dir("working/results")

    """ Load dataset """
    # 1. 加载三个路径列表：图像、金标准Mask、视盘Mask
    test_x = sorted(glob("working/new_data/test/image/*"))
    test_y = sorted(glob("working/new_data/test/mask/*"))
    test_od = sorted(glob("working/new_data/test/od_mask/*")) # 【新增】加载视盘Mask路径

    # 检查文件数量是否匹配
    if len(test_x) != len(test_od):
        print(f"警告: 图像数量 ({len(test_x)}) 与视盘Mask数量 ({len(test_od)}) 不匹配！")

    """ Hyperparameters """
    H = 560
    W = 560
    size = (W, H)
    checkpoint_path = "working/files/drive_checkpoint_od.pth"

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型时，确保 input_channels=4
    model = build_unet(input_channels=4) 
    model = model.to(device)
    
    # 如果训练时用了 DataParallel，这里加载权重可能需要处理 'module.' 前缀
    # 这里假设直接加载
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []
    
    # 遍历 Image, Label, OD_Mask
    for i, (x_path, y_path, od_path) in tqdm(enumerate(zip(test_x, test_y, test_od)), total=len(test_x)):
        """ Extract the name """
        name = x_path.split("/")[-1].split(".")[0]

        """ 1. 处理原图 (RGB) """
        image = cv2.imread(x_path, cv2.IMREAD_COLOR) 
        # image 已经是 560x560 (数据增强阶段处理过)
        x = np.transpose(image, (2, 0, 1))      ## (3, 560, 560)
        x = x / 255.0
        
        """ 2. 处理视盘 Mask (OD) """
        od_mask = cv2.imread(od_path, cv2.IMREAD_GRAYSCALE) # 读取单通道
        # od_mask 已经是 560x560
        od_mask = od_mask / 255.0
        od_mask = np.expand_dims(od_mask, axis=0) ## (1, 560, 560)

        """ 3. 合并通道 (Concatenate) """
        # 将 RGB (3, H, W) 和 OD (1, H, W) 在通道维度合并 -> (4, H, W)
        x_concat = np.concatenate([x, od_mask], axis=0)
        
        # 转为 Tensor 并增加 Batch 维度 -> (1, 4, 560, 560)
        x_tensor = np.expand_dims(x_concat, axis=0) 
        x_tensor = x_tensor.astype(np.float32)
        x_tensor = torch.from_numpy(x_tensor).to(device)

        """ Reading label mask """
        mask = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
        y = np.expand_dims(mask, axis=0)            
        y = y / 255.0
        y = np.expand_dims(y, axis=0)               
        y = y.astype(np.float32)
        y = torch.from_numpy(y).to(device)

        with torch.no_grad():
            """ Prediction """
            start_time = time.time()
            
            # 模型输入现在是 4 通道，不会报错了
            pred_y = model(x_tensor) 
            pred_y = torch.sigmoid(pred_y)
            
            total_time = time.time() - start_time
            time_taken.append(total_time)

            score = calculate_metrics(y, pred_y)
            # print(f":-- jaccard:{score[0]:1.4f} ...") 
            
            metrics_score = list(map(add, metrics_score, score))
            
            # 后处理用于保存图片
            pred_y = pred_y[0].cpu().numpy()        
            pred_y = np.squeeze(pred_y, axis=0)     
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        """ Saving masks """
        ori_mask = mask_parse(mask)
        pred_y_viz = mask_parse(pred_y)
        line = np.ones((size[1], 10, 3)) * 128

        # 拼接图片用于展示结果
        cat_images = np.concatenate(
            [image, line, ori_mask, line, pred_y_viz * 255], axis=1
        )
        cv2.imwrite(f"working/results/{name}.png", cat_images)

    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    acc = metrics_score[4]/len(test_x)
    spec = metrics_score[5]/len(test_x)
    print(f"\nOverall---Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Sensitivity: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - Specificity:{spec:1.4f}")

    fps = 1/np.mean(time_taken)
    print("FPS: ", fps)

# %%



