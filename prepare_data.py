import os
import cv2
import glob
import gzip
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image

# ================= 配置区域 =================
# 目标保存路径
OUTPUT_DIR = "./data/unified_data"
IMG_SIZE = 512  # 统一调整为 512x512

# 源数据路径 (请确保与你本地路径一致)
PATH_DRIVE_TRAIN = "./data/DRIVE/training"
PATH_DRIVE_TEST  = "./data/DRIVE/test"
PATH_STARE       = "./data/STARE"
PATH_CHASE       = "./data/CHASE_DB"
# ===========================================

def make_dirs():
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "masks"), exist_ok=True)

def save_data(image, mask, prefix, filename):
    """
    统一保存函数：Resize -> Save as PNG
    """
    if image is None or mask is None:
        print(f"出错: 图片或Mask为空 - {filename}")
        return

    # Resize
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    # 确保 Mask 是单通道
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # 生成新文件名
    # 去掉所有后缀，只留纯文件名
    base_name = filename.split('.')[0] 
    new_name = f"{prefix}_{base_name}.png"

    # 保存路径
    img_save_path = os.path.join(OUTPUT_DIR, "images", new_name)
    mask_save_path = os.path.join(OUTPUT_DIR, "masks", new_name)

    cv2.imwrite(img_save_path, image)
    cv2.imwrite(mask_save_path, mask)

def load_gz_image(path, is_mask=False):
    """
    专门读取 .gz 压缩图片
    """
    try:
        with gzip.open(path, 'rb') as f:
            # 使用 PIL 读取文件流
            pil_img = Image.open(f)
            pil_img.load() # 强制读取到内存
            
            if is_mask:
                # Mask 转灰度
                return np.array(pil_img.convert('L'))
            else:
                # 原图转 RGB numpy 数组
                img_rgb = np.array(pil_img.convert('RGB'))
                # OpenCV 需要 BGR 格式
                return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"读取压缩文件失败 {path}: {e}")
        return None

def process_drive(root_path, subset_name):
    print(f"--- 处理 DRIVE {subset_name} ---")
    img_dir = os.path.join(root_path, "images")
    mask_dir = os.path.join(root_path, "1st_manual") 
    
    img_list = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    
    for img_path in tqdm(img_list):
        image = cv2.imread(img_path)
        
        filename = os.path.basename(img_path)
        img_id = filename.split('_')[0] 
        
        mask_name = f"{img_id}_manual1.gif"
        mask_path = os.path.join(mask_dir, mask_name)
        
        if not os.path.exists(mask_path):
            # 尝试找 .png (如果你之前转过)
            mask_path = mask_path.replace(".gif", ".png")
            if not os.path.exists(mask_path):
                continue
            
        # 读取 Mask (兼容 gif/png)
        mask_pil = Image.open(mask_path).convert('L')
        mask = np.array(mask_pil)

        save_data(image, mask, "drive", filename)

def process_stare(root_path):
    print(f"--- 处理 STARE (支持 .gz) ---")
    
    # 路径匹配你的截图
    img_dir = os.path.join(root_path, "stare-images")
    mask_dir = os.path.join(root_path, "labels-ah")
    
    # 获取所有 .gz 文件
    gz_files = sorted(glob.glob(os.path.join(img_dir, "*.gz")))
    
    for img_path in tqdm(gz_files):
        # 1. 读取原图 (im0001.ppm.gz)
        image = load_gz_image(img_path, is_mask=False)
        if image is None: continue
        
        filename = os.path.basename(img_path) # im0001.ppm.gz
        
        # 2. 解析 ID
        # 分割文件名: im0001.ppm.gz -> im0001
        base_id = filename.split('.')[0] 
        
        # 3. 寻找 Mask
        # STARE 标签通常命名为 im0001.ah.ppm.gz
        mask_name = f"{base_id}.ah.ppm.gz"
        mask_path = os.path.join(mask_dir, mask_name)
        
        if not os.path.exists(mask_path):
            print(f"跳过: 找不到标签 {mask_name}")
            continue
            
        # 4. 读取 Mask
        mask = load_gz_image(mask_path, is_mask=True)
        
        # 5. 保存
        save_data(image, mask, "stare", base_id)

def process_chase(root_path):
    print(f"--- 处理 CHASE_DB ---")
    img_dir = os.path.join(root_path, "img")
    mask_dir = os.path.join(root_path, "Masks")
    
    img_list = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not img_list:
        # 尝试找 .png
        img_list = sorted(glob.glob(os.path.join(img_dir, "*.png")))

    for img_path in tqdm(img_list):
        image = cv2.imread(img_path)
        filename = os.path.basename(img_path)
        
        base_name = os.path.splitext(filename)[0] # Image_01L
        
        # 尝试匹配多种 Mask 后缀
        candidates = [
            f"{base_name}_1stHO.png",
            f"{base_name}_1stHO.jpg",
            f"{base_name}.png" # 有些版本可能直接同名
        ]
        
        mask = None
        for c in candidates:
            mp = os.path.join(mask_dir, c)
            if os.path.exists(mp):
                mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                break
        
        if mask is None:
            print(f"跳过: CHASE mask找不到 {filename}")
            continue
            
        save_data(image, mask, "chase", filename)

if __name__ == "__main__":
    make_dirs()
    
    # 1. 处理 STARE (本次修复的重点)
    if os.path.exists(PATH_STARE):
        process_stare(PATH_STARE)
    else:
        print("未找到 STARE 路径")

    # 2. 处理 DRIVE
    if os.path.exists(PATH_DRIVE_TRAIN):
        process_drive(PATH_DRIVE_TRAIN, "Training")
    if os.path.exists(PATH_DRIVE_TEST):
        process_drive(PATH_DRIVE_TEST, "Test")
        
    # 3. 处理 CHASE_DB
    if os.path.exists(PATH_CHASE):
        process_chase(PATH_CHASE)
        
    print("\n处理完成！请检查 ./data/unified_data 文件夹")