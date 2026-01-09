# RetinaSegProject

视网膜血管分割项目 - 基于 LFA-Net (Lite Fusion Attention Network) 的医学图像分割系统

## 项目简介

本项目实现了一个端到端的视网膜血管分割系统，结合了：
- **LFA-Net**: 轻量级融合注意力网络，包含 Focal Modulation 和 Vision Mamba 启发的模块
- **视盘检测**: 使用 SegFormer 进行视盘(Optic Disc)区域分割
- **边缘优化**: 基于距离场的视盘边缘增强处理

支持三个公开数据集：**DRIVE**, **CHASE_DB**, **STARE**

## 项目结构

```
RetinaSegProject/
├── main.py                 # 统一CLI入口 (唯一的运行脚本)
├── src/                    # 核心模块
│   ├── __init__.py         # 模块导出
│   ├── config.py           # 配置文件 (路径、超参数)
│   ├── model.py            # LFA-Net 网络模型
│   ├── loss.py             # 损失函数 (Focal Tversky + Boundary)
│   ├── dataset.py          # PyTorch 数据集类
│   ├── augment.py          # 数据增强
│   ├── trainer.py          # 训练模块
│   ├── tester.py           # 测试模块
│   ├── od_utils.py         # 视盘分割工具 (SegFormer)
│   ├── transforms.py       # 图像预处理
│   └── utils.py            # 工具函数
├── data/                   # 数据目录
│   ├── DRIVE/              # DRIVE 数据集
│   ├── CHASE_DB/           # CHASE_DB 数据集
│   └── STARE/              # STARE 数据集
├── working/                # 工作目录
│   ├── new_data/           # 处理后的数据
│   ├── checkpoints/        # 模型权重
│   ├── logs/               # 训练日志
│   └── results/            # 测试结果
└── requirements.txt        # 依赖包
```

## 安装

### 环境要求

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.x (推荐)

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- torch, torchvision
- opencv-python
- albumentations
- transformers (用于 SegFormer)
- scikit-learn
- matplotlib
- tqdm

## 使用方法

### 命令行参数

```bash
python main.py --help
```

```
usage: main.py [-h] [--prepare] [--train] [--test] [--compare] [--all]
               [--dataset DATASET] [--model MODEL] [--loss LOSS]
               [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--lr LR]
               [--no-od] [--od-intensity OD_INTENSITY]
               [--parallel] [--gpus GPUS] [--seed SEED]

Commands:
  --prepare       数据预处理和增强
  --train         训练模型
  --test          测试模型
  --compare       对比测试 (Baseline vs OD优化)
  --all           完整流程 (prepare + train + test)

Dataset Options:
  --dataset       数据集名称，多个用逗号分隔，或使用 'all'

Model & Loss Options:
  --model         模型类型: baseline, dcn_sp, aspp (默认: baseline)
  --loss          损失函数: vessel, dice, dice_bce, bce, focal_tversky (默认: vessel)

Training Options:
  --epochs        训练轮数 (默认: 200)
  --batch-size    批次大小 (默认: 64)
  --lr            学习率 (默认: 1e-3)

Testing Options:
  --no-od         测试时禁用视盘分割
  --od-intensity  OD边缘增强强度 (默认: 60)

Parallel Options:
  --parallel      多GPU并行处理不同数据集
  --gpus          GPU ID列表，用逗号分隔
```

### 使用示例

#### 1. 单数据集完整流程

```bash
# 数据准备 + 训练 + 测试
python main.py --all --dataset DRIVE
```

#### 2. 分步执行

```bash
# 仅数据预处理
python main.py --prepare --dataset DRIVE

# 仅训练
python main.py --train --dataset DRIVE --epochs 100 --batch-size 32

# 仅测试
python main.py --test --dataset DRIVE

# 对比测试
python main.py --compare --dataset DRIVE --od-intensity 60
```

#### 3. 使用不同模型变体

```bash
# 使用 DCN + Strip Pooling 模型
python main.py --train --dataset DRIVE --model dcn_sp

# 使用 ASPP 模型
python main.py --train --dataset DRIVE --model aspp

# 使用不同损失函数
python main.py --train --dataset DRIVE --loss dice_bce

# 组合使用
python main.py --all --dataset DRIVE --model aspp --loss dice
```

#### 4. 多数据集处理

```bash
# 串行处理多个数据集
python main.py --all --dataset DRIVE,CHASE_DB,STARE

# 或使用 all 关键字
python main.py --all --dataset all
```

#### 5. 多GPU并行

```bash
# 使用多GPU并行处理不同数据集
python main.py --all --dataset all --parallel --gpus 0,1,2
```

## 模型架构

### LFA-Net (Lite Fusion Attention Network)

```
输入图像 (3, H, W)
    │
    ▼
┌─────────────────┐
│   Encoder       │
│  ConvBlock x3   │  ← 1x1 + 3x3 + 3x3(dilation) + LeakyReLU
│  + MaxPool      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LiteFusionAttn  │  ← Focal Modulation + Vision Mamba Inspired
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Decoder       │
│  RA_Attention   │  ← Resource-Aware Attention
│  + Upsample x3  │
└────────┬────────┘
         │
         ▼
输出掩码 (1, H, W)
```

### 核心模块

1. **FocalModulation**: 自适应焦点调制注意力
2. **FocalModulationContextAggregation (FMCA)**: 上下文聚合模块
3. **VisionMambaInspired**: Vision Mamba 启发的 Token/Channel Mixer
4. **LiteFusionAttention**: 轻量融合注意力
5. **RA_AttentionBlock**: 资源感知注意力

### 模型变体

| 模型 | 参数 | 描述 |
|------|------|------|
| `baseline` | `--model baseline` | 标准 LFA-Net |
| `dcn_sp` | `--model dcn_sp` | DCN (可变形卷积) + Strip Pooling，适合弯曲血管 |
| `aspp` | `--model aspp` | 添加 ASPP 模块，多尺度特征提取 |

### 损失函数

| 损失函数 | 参数 | 描述 |
|----------|------|------|
| `vessel` | `--loss vessel` | Focal Tversky + Boundary (默认，推荐) |
| `dice` | `--loss dice` | 纯 Dice Loss |
| `dice_bce` | `--loss dice_bce` | Dice + BCE 联合损失 |
| `bce` | `--loss bce` | 纯 BCE Loss |
| `focal_tversky` | `--loss focal_tversky` | 纯 Focal Tversky Loss |

**Vessel Loss 公式:**
```
L = w_tversky * FocalTverskyLoss + w_boundary * BoundaryLoss
```

- **Focal Tversky Loss**: 处理血管细长、类别不平衡的特性 (α=0.7, β=0.3, γ=0.75)
- **Boundary Loss**: Sobel 边界提取 + L1 损失，强化血管边界

## 数据集

### DRIVE
- 40张视网膜图像 (20训练 + 20测试)
- 图像格式: .tif
- 标注格式: .gif

### CHASE_DB
- 28张视网膜图像 (自动8:2划分)
- 图像格式: .jpg
- 标注格式: .png

### STARE
- 20张视网膜图像 (自动8:2划分)
- 图像格式: .ppm
- 标注格式: .ppm

## 配置参数

编辑 `src/config.py` 修改默认参数：

```python
# 图像尺寸
IMG_SIZE = (560, 560)

# 训练参数
BATCH_SIZE = 64
NUM_EPOCHS = 200
LR = 1e-3
NUM_WORKERS = 16

# 路径配置
DATA_ROOT = "data"
PROCESSED_DATA_ROOT = "working/new_data"
CHECKPOINT_DIR = "working/checkpoints"
RESULT_DIR = "working/results"
LOG_DIR = "working/logs"
```

## 输出结果

### 训练输出

- `working/checkpoints/{dataset}_checkpoint.pth`: 最佳模型权重
- `working/logs/{dataset}_log.csv`: 训练日志
- `working/logs/{dataset}_loss.png`: 损失曲线图

### 测试输出

- `working/results/{dataset}/`: 分割结果可视化
  - 每张图包含：原图 | 真值 | OD掩码 | 误差图

### 评估指标

- **IoU (Jaccard)**: 交并比
- **F1 (Dice)**: Dice 系数
- **Sensitivity (Recall)**: 召回率
- **Precision**: 精确率
- **Accuracy**: 准确率
- **Specificity**: 特异性

## API 使用

也可以在 Python 代码中直接使用：

```python
from src import (
    build_unet, run_training, run_testing,
    run_comparative_test, ODSegmenter
)

# 构建模型
model = build_unet()

# 训练
run_training("DRIVE", epochs=100)

# 测试
run_testing("DRIVE")

# 对比测试
run_comparative_test("DRIVE")

# 使用视盘分割
od_seg = ODSegmenter()
od_mask = od_seg.process(image_bgr)
```

## 技术特点

1. **统一CLI入口**: 所有功能通过 `main.py` 一个脚本调用
2. **模块化设计**: 清晰的模块划分，易于扩展
3. **多GPU支持**: 支持多GPU并行处理不同数据集
4. **视盘处理**: 集成 SegFormer 视盘分割，提供边缘优化
5. **完整流程**: 从数据预处理到测试的端到端流程

## 参考文献

- LFA-Net 相关论文
- SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
- DRIVE, CHASE_DB, STARE 数据集

## License

MIT License
