import os
import cv2
import imageio
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from albumentations import HorizontalFlip, VerticalFlip, Rotate, RandomBrightnessContrast, RandomCrop, RandomRotate90, RandomGridShuffle

from src.utils import create_dir
from src.config import DATASET_CONFIG, PROCESSED_DATA_ROOT, IMG_SIZE

def load_raw_data(dataset_name):
    cfg = DATASET_CONFIG[dataset_name]
    path = cfg["root_path"]
    
    if dataset_name == "DRIVE":
        train_x = sorted(glob(os.path.join(path, "training", "images", f"*{cfg['img_ext']}")))
        train_y = sorted(glob(os.path.join(path, "training", "1st_manual", f"*{cfg['mask_ext']}")))
        test_x = sorted(glob(os.path.join(path, "test", "images", f"*{cfg['img_ext']}")))
        test_y = sorted(glob(os.path.join(path, "test", "1st_manual", f"*{cfg['mask_ext']}")))
        return (train_x, train_y), (test_x, test_y)
    else:
        # CHASE_DB 和 STARE 需要读取并随机划分
        images = sorted(glob(os.path.join(path, cfg["img_dir"], f"*{cfg['img_ext']}")))
        masks = sorted(glob(os.path.join(path, cfg["mask_dir"], f"*{cfg['mask_ext']}")))
        
        if len(images) == 0:
            print(f"Error: No images found in {path}")
            return ([], []), ([], [])
            
        train_x, test_x, train_y, test_y = train_test_split(images, masks, test_size=0.2, random_state=42)
        return (train_x, train_y), (test_x, test_y)

def perform_augmentation(images, masks, save_path, augment=True):
    size = IMG_SIZE

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images), leave=False):
        name = os.path.basename(x).split(".")[0]

        # 读取图像
        x_img = cv2.imread(x, cv2.IMREAD_COLOR)
        
        # 读取掩码 (处理 gif 与其他格式)
        if y.endswith(".gif"):
            y_mask = imageio.mimread(y)[0]
        else:
            y_mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

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
            
            X_list = [x_img]
            Y_list = [y_mask]

            for aug in transformations:
                augmented = aug(image=x_img, mask=y_mask)
                X_list.append(augmented["image"])
                Y_list.append(augmented["mask"])
        else:
            X_list = [x_img]
            Y_list = [y_mask]

        index = 0
        for i, m in zip(X_list, Y_list):
            try:
                i = cv2.resize(i, size)
                m = cv2.resize(m, size)
            except Exception as e:
                print(f"Resize error on {name}: {e}")
                continue

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)
            index += 1

def run_augmentation(dataset_name):
    print(f"[Augment] Processing {dataset_name}...")
    
    (train_x, train_y), (test_x, test_y) = load_raw_data(dataset_name)
    
    # 构建保存路径: working/new_data/DRIVE/train/image ...
    save_root = os.path.join(PROCESSED_DATA_ROOT, dataset_name)
    create_dir(os.path.join(save_root, "train", "image"))
    create_dir(os.path.join(save_root, "train", "mask"))
    create_dir(os.path.join(save_root, "test", "image"))
    create_dir(os.path.join(save_root, "test", "mask"))

    perform_augmentation(train_x, train_y, os.path.join(save_root, "train"), augment=True)
    perform_augmentation(test_x, test_y, os.path.join(save_root, "test"), augment=False)
    print(f"[Augment] {dataset_name} Finished.")