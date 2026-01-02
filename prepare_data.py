import os
import argparse
import glob
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from src.utils import create_dir
from src.transforms import ODGenerator, RetinaPreprocess
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/DRIVE", help="原始数据路径")
    parser.add_argument("--output_path", type=str, default="output/processed_data", help="输出路径")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--augment_times", type=int, default=5, help="每张图增强多少倍")
    return parser.parse_args()

def get_transforms(size):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.95, 1.05), rotate=(-10, 10), shear=(-2, 2), p=0.8),
        A.Perspective(scale=(0.01, 0.02), keep_size=True, p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
        ], p=0.3),
    ], additional_targets={'mask0': 'mask'}) # mask0 用于 OD Mask

def process_set(subset, args, od_gen):
    print(f"Processing {subset} set...")
    img_dir = os.path.join(args.data_path, subset, "images")
    mask_dir = os.path.join(args.data_path, subset, "1st_manual")
    
    save_img_dir = os.path.join(args.output_path, subset, "image")
    save_mask_dir = os.path.join(args.output_path, subset, "mask")
    save_od_dir = os.path.join(args.output_path, subset, "od_mask")
    
    for d in [save_img_dir, save_mask_dir, save_od_dir]:
        create_dir(d)

    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")) + glob.glob(os.path.join(img_dir, "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.gif")) + glob.glob(os.path.join(mask_dir, "*.png")))

    transform = get_transforms(args.img_size)

    for img_p, mask_p in tqdm(zip(img_paths, mask_paths), total=len(img_paths)):
        name = os.path.basename(img_p).split(".")[0]
        
        # 1. 读图
        image = cv2.imread(img_p)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. 读/生成 OD Mask
        # 这里实时生成，也可以先生成好保存
        od_mask = od_gen.get_mask(image)
        
        # 3. 读 Label Mask
        import imageio
        mask = imageio.mimread(mask_p)[0] if mask_p.endswith('.gif') else cv2.imread(mask_p, 0)
        
        # 4. 特征工程 (Green+CLAHE+TopHat)
        image_harmonized = RetinaPreprocess.harmonize_od_color(image, od_mask)
        image_engineered = RetinaPreprocess.get_green_enhanced_stack(image_harmonized) # (H, W, 3)

        # 5. 增强循环
        # 测试集通常只 Resize 不增强，这里简单处理：如果是 test 且 augment_times > 0，也可以增强
        # 或者在代码里控制：训练集增强，测试集仅 Resize
        times = args.augment_times if subset == "training" else 1
        
        for i in range(times):
            # 如果是测试集且不需要增强，需要一个新的 transform 只做 Resize
            if subset == "test":
                aug = A.Compose([A.Resize(args.img_size, args.img_size)], additional_targets={'mask0': 'mask'})(image=image_engineered, mask=mask, mask0=od_mask)
            else:
                aug = transform(image=image_engineered, mask=mask, mask0=od_mask)
            
            # 保存 (image_engineered 是 3通道，可以直接 imwrite，无需转 BGR，因为它是特征图)
            cv2.imwrite(os.path.join(save_img_dir, f"{name}_{i}.png"), aug['image'])
            cv2.imwrite(os.path.join(save_mask_dir, f"{name}_{i}.png"), aug['mask'])
            cv2.imwrite(os.path.join(save_od_dir, f"{name}_{i}.png"), aug['mask0'])

if __name__ == "__main__":
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    od_gen = ODGenerator(device)
    
    process_set("training", args, od_gen)
    process_set("test", args, od_gen)