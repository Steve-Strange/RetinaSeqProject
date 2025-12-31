import os
import cv2
import glob
import gzip
import numpy as np
from tqdm import tqdm
from PIL import Image

# === 配置 ===
OUTPUT_DIR = "./data/unified_data"
IMG_SIZE = 512

# 请修改为你本地的实际路径
PATH_DRIVE_TRAIN = "./data/DRIVE/training"
PATH_DRIVE_TEST  = "./data/DRIVE/test"
PATH_STARE       = "./data/STARE"
PATH_CHASE       = "./data/CHASE_DB"
# ============

def make_dirs():
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "masks"), exist_ok=True)

def save_data(image, mask, prefix, filename):
    if image is None or mask is None: return
    
    # Resize
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    if len(mask.shape) == 3: mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # 二值化 Mask (确保只有0和255)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    base_name = filename.split('.')[0]
    new_name = f"{prefix}_{base_name}.png"

    cv2.imwrite(os.path.join(OUTPUT_DIR, "images", new_name), image)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "masks", new_name), mask)

def load_gz_image(path, is_mask=False):
    try:
        with gzip.open(path, 'rb') as f:
            pil_img = Image.open(f)
            pil_img.load()
            if is_mask: return np.array(pil_img.convert('L'))
            else: return cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
    except: return None

def process_all():
    make_dirs()
    
    # 1. DRIVE
    for subset in [PATH_DRIVE_TRAIN, PATH_DRIVE_TEST]:
        if not os.path.exists(subset): continue
        print(f"处理 DRIVE: {subset}")
        img_list = glob.glob(os.path.join(subset, "images", "*.tif"))
        for p in tqdm(img_list):
            img = cv2.imread(p)
            fname = os.path.basename(p)
            img_id = fname.split('_')[0]
            mask_path = os.path.join(subset, "1st_manual", f"{img_id}_manual1.gif")
            if not os.path.exists(mask_path): # 尝试找 .png
                 mask_path = mask_path.replace(".gif", ".png")
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path).convert('L'))
                save_data(img, mask, "drive", fname)

    # 2. STARE (支持 .gz)
    if os.path.exists(PATH_STARE):
        print("处理 STARE...")
        gz_files = glob.glob(os.path.join(PATH_STARE, "stare-images", "*.gz"))
        for p in tqdm(gz_files):
            img = load_gz_image(p)
            fname = os.path.basename(p)
            base_id = fname.split('.')[0]
            mask_path = os.path.join(PATH_STARE, "labels-ah", f"{base_id}.ah.ppm.gz")
            if os.path.exists(mask_path):
                mask = load_gz_image(mask_path, is_mask=True)
                save_data(img, mask, "stare", base_id)

    # 3. CHASE_DB
    if os.path.exists(PATH_CHASE):
        print("处理 CHASE_DB...")
        img_list = glob.glob(os.path.join(PATH_CHASE, "img", "*.jpg")) + glob.glob(os.path.join(PATH_CHASE, "images", "*.png"))
        for p in tqdm(img_list):
            img = cv2.imread(p)
            fname = os.path.basename(p)
            base = os.path.splitext(fname)[0]
            # 尝试匹配 mask
            candidates = [f"{base}_1stHO.png", f"{base}_1stHO.jpg", f"{base}.png"]
            mask = None
            for c in candidates:
                mp = os.path.join(PATH_CHASE, "Masks", c)
                if os.path.exists(mp):
                    mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                    break
            save_data(img, mask, "chase", fname)

if __name__ == "__main__":
    process_all()
    print("数据统一完成！")