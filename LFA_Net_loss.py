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
def augment_data(images, masks, save_path, augment=True):
    size = (560, 560)

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting the name """
        name = x.split(os.sep)[-1].split(".")[0]
        
        """ Reading image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]

        if x is None or y is None:
            print(f"Error reading image or mask for {name}")
            continue

        if augment == True:
            transformations = [
                HorizontalFlip(p=1.0),
                VerticalFlip(p=1.0),
                Rotate(limit=45, p=1.0),
                RandomBrightnessContrast(p=0.6),
                RandomCrop(300, 300, p=0.8),
                RandomRotate90(p=1.0),
                RandomGridShuffle(p=0.7)
            ]

            augmented_images = [x]
            augmented_masks = [y]

            for aug in transformations:
                augmented = aug(image=x, mask=y)
                augmented_images.append(augmented["image"])
                augmented_masks.append(augmented["mask"])
            
            X = augmented_images
            Y = augmented_masks

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            #将图像和掩码调整到目标大小
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the data """
    # 修改为相对路径
    data_path = "data/DRIVE/"
    
    if os.path.exists(data_path):
        (train_x, train_y), (test_x, test_y) = load_data(data_path)

        print(f"Train: {len(train_x)} - {len(train_y)}")
        print(f"Test: {len(test_x)} - {len(test_y)}")

        """ Create directories to save the augmented data """
        create_dir("working/new_data/train/image/")
        create_dir("working/new_data/train/mask/")
        create_dir("working/new_data/test/image/")
        create_dir("working/new_data/test/mask/")

        """ Data augmentation """
        # 取消注释以运行数据增强
        augment_data(train_x, train_y, "working/new_data/train/", augment=True)
        augment_data(test_x, test_y, "working/new_data/test/", augment=False)
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
    def __init__(self, input_channels=3, num_classes=1, feature_scale=2, dropout=0.5):
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
    x = torch.randn((2, 3, 560, 560))
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
    def __init__(self, images_path, masks_path):

        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = image/255.0 ## (512, 512, 3)
        image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = mask/255.0   ## (512, 512)
        mask = np.expand_dims(mask, axis=0) ## (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        return image, mask

    def __len__(self):
        return self.n_samples

# %%
# Loss
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss
    特别适合类别极不平衡 + 微结构（血管）
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )

        loss = torch.pow((1 - tversky), self.gamma)
        return loss

class BoundaryLoss(nn.Module):
    """
    Boundary Loss via distance transform
    强制预测边界贴近 GT 边界
    """
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)

        # 计算边界（简单 Sobel）
        def get_boundary(x):
            sobel_x = torch.tensor([[1, 0, -1],
                                    [2, 0, -2],
                                    [1, 0, -1]], device=x.device).float()
            sobel_y = sobel_x.t()

            sobel_x = sobel_x.view(1, 1, 3, 3)
            sobel_y = sobel_y.view(1, 1, 3, 3)

            grad_x = F.conv2d(x, sobel_x, padding=1)
            grad_y = F.conv2d(x, sobel_y, padding=1)

            return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        pred_boundary = get_boundary(inputs)
        gt_boundary = get_boundary(targets)

        return F.l1_loss(pred_boundary, gt_boundary)

class VesselSegmentationLoss(nn.Module):
    """
    最终联合损失：
    Focal Tversky + Boundary
    """
    def __init__(self, w_tversky=1.0, w_boundary=0.1):
        super(VesselSegmentationLoss, self).__init__()
        self.tversky = FocalTverskyLoss()
        self.boundary = BoundaryLoss()
        self.w_tversky = w_tversky
        self.w_boundary = w_boundary

    def forward(self, inputs, targets):
        loss_t = self.tversky(inputs, targets)
        loss_b = self.boundary(inputs, targets)
        return self.w_tversky * loss_t + self.w_boundary * loss_b


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

    valid_x = sorted(glob("working/new_data/test/image/*"))
    valid_y = sorted(glob("working/new_data/test/mask/*"))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ Hyperparameters """
    H = 560
    W = 560
    size = (H, W)
    batch_size = 64
    num_epochs = 100   
    lr = 1e-3
    checkpoint_path = "working/files/drive_checkpoint_loss.pth"

    """ Dataset and loader """
    train_dataset = DriveDataset(train_x, train_y)
    valid_dataset = DriveDataset(valid_x, valid_y)

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
    # loss_fn = DiceBCELoss()
    
    loss_fn = VesselSegmentationLoss(
        w_tversky=1.0,
        w_boundary=0.1
    )


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
import imageio
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, confusion_matrix

# from model import build_unet
# from utils import create_dir, seeding

def calculate_metrics(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    # 使用混淆矩阵计算 TP, FP, FN, TN
    # labels=[0, 1] 确保即使数据中缺少某一类也能返回 2x2 矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Dice Coefficient (F1 Score) = 2 * TP / (2 * TP + FP + FN)
    score_f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)

    # Jaccard (IoU) = TP / (TP + FP + FN)
    score_jaccard = tp / (tp + fp + fn + 1e-6)

    # Sensitivity (Recall) = TP / (TP + FN)
    score_recall = tp / (tp + fn + 1e-6)

    # Specificity = TN / (TN + FP)
    score_specificity = tn / (tn + fp + 1e-6)

    # Precision = TP / (TP + FP)
    score_precision = tp / (tp + fp + 1e-6)

    # Accuracy = (TP + TN) / (TP + TN + FP + FN)
    score_acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_specificity]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Folders """
    create_dir("working/results")

    """ Load dataset """
    test_x = sorted(glob("working/new_data/test/image/*"))
    test_y = sorted(glob("working/new_data/test/mask/*"))

    """ Hyperparameters """
    H = 560
    W = 560
    size = (W, H)
    checkpoint_path = "working/files/drive_checkpoint_loss.pth"

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []
    
    # Lists to store metrics for each image for plotting
    image_indices = []
    jaccard_scores = []
    f1_scores = []
    recall_scores = []
    specificity_scores = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading image """
        image = cv2.imread(x, cv2.IMREAD_COLOR) ## (512, 512, 3)
        ## image = cv2.resize(image, size)
        x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512)
        x = x/255.0
        x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)

        """ Reading mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
        ## mask = cv2.resize(mask, size)
        y = np.expand_dims(mask, axis=0)            ## (1, 512, 512)
        y = y/255.0
        y = np.expand_dims(y, axis=0)               ## (1, 1, 512, 512)
        y = y.astype(np.float32)
        y = torch.from_numpy(y)
        y = y.to(device)

        with torch.no_grad():
            """ Prediction and Calculating FPS """
            start_time = time.time()
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            total_time = time.time() - start_time
            time_taken.append(total_time)


            score = calculate_metrics(y, pred_y)
            print(f":-- jaccard:{score[0]:1.4f}, f1:{score[1]:1.4f},recall:{score[2]:1.4f},precision:{score[3]:1.3f},acc:{score[4]:1.4f},specificity:{score[5]:1.3f}")
            
            metrics_score = list(map(add, metrics_score, score))
            
            # Store for plotting
            image_indices.append(i)
            jaccard_scores.append(score[0])
            f1_scores.append(score[1])
            recall_scores.append(score[2])
            specificity_scores.append(score[5])
            
            pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
            pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        """ Saving masks """
        ori_mask = mask_parse(mask)
        pred_y = mask_parse(pred_y)
        line = np.ones((size[1], 10, 3)) * 128

        cat_images = np.concatenate(
            [image, line, ori_mask, line, pred_y * 255], axis=1
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



