"""
RetinaSegProject - 视网膜血管分割项目

核心模块:
    - config: 配置文件
    - model: LFA-Net网络模型
    - loss: 损失函数 (Focal Tversky + Boundary)
    - dataset: PyTorch数据集类
    - augment: 数据增强
    - trainer: 训练模块
    - tester: 测试模块
    - od_utils: 视盘分割工具
    - transforms: 图像预处理
    - utils: 工具函数
"""
from .config import (
    DATA_ROOT, PROCESSED_DATA_ROOT, CHECKPOINT_DIR, RESULT_DIR, LOG_DIR,
    DATASET_CONFIG, TARGET_DATASETS, IMG_SIZE, BATCH_SIZE, NUM_EPOCHS, LR
)
from .model import build_unet
from .loss import VesselSegmentationLoss
from .dataset import DriveDataset
from .augment import run_augmentation
from .trainer import run_training
from .tester import run_testing, run_comparative_test
from .od_utils import ODSegmenter
from .utils import seeding, create_dir, calculate_metrics

__version__ = "1.0.0"
__author__ = "RetinaSegProject Team"
