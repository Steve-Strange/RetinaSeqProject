import os

# === 硬件设置 ===
# 注意：这里删除了 os.environ["CUDA_VISIBLE_DEVICES"]
# 显卡分配将在 main.py 中通过多进程动态指定

# === 路径配置 ===
DATA_ROOT = "data"
PROCESSED_DATA_ROOT = "working/new_data"
CHECKPOINT_DIR = "working/checkpoints"
RESULT_DIR = "working/results"

# === 数据集配置 ===
DATASET_CONFIG = {
    "DRIVE": {
        "root_path": os.path.join(DATA_ROOT, "DRIVE"),
        "img_ext": ".tif",
        "mask_ext": ".gif",
        "has_split": True
    },
    "CHASE_DB": {
        "root_path": os.path.join(DATA_ROOT, "CHASE_DB"),
        "img_dir": "img",
        "mask_dir": "Masks",
        "img_ext": ".jpg", 
        "mask_ext": ".png",
        "has_split": False
    },
    "STARE": {
        "root_path": os.path.join(DATA_ROOT, "STARE"),
        "img_dir": "stare-images",
        "mask_dir": "labels-ah",
        "img_ext": ".ppm",
        "mask_ext": ".ppm",
        "has_split": False
    }
}

# 这里定义好你想跑的任务
TARGET_DATASETS = ["DRIVE", "CHASE_DB", "STARE"]

# === 训练超参数 (单卡 A6000 版) ===
IMG_SIZE = (560, 560)


BATCH_SIZE = 64      

# 因为同时跑3个任务，CPU压力大，Workers不要设太大，建议 8
NUM_WORKERS = 16

NUM_EPOCHS = 200
LR = 1e-3