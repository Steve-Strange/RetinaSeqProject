"""
测试模块
包含：标准测试、对比测试(含OD优化)、可视化结果生成
"""
import os
import numpy as np
import cv2
import torch
from glob import glob
from tqdm import tqdm
from operator import add
from collections import OrderedDict

from .config import PROCESSED_DATA_ROOT, CHECKPOINT_DIR, RESULT_DIR
from .model import get_model
from .od_utils import ODSegmenter
from .utils import (
    create_dir, calculate_metrics,
    mask_to_bgr, draw_error_map, add_label, harmonize_od_edge
)


def run_testing(dataset_name, use_od=True, model_type='baseline'):
    """
    执行标准测试

    Args:
        dataset_name: 数据集名称 (DRIVE/CHASE_DB/STARE)
        use_od: 是否启用视盘分割显示
        model_type: 模型类型 ('baseline', 'dcn_sp', 'aspp')
    """
    print(f"\n[Test] Starting testing for {dataset_name}...")
    print(f"  Model: {model_type}")

    suffix = f"_{model_type}" if model_type != 'baseline' else ""
    data_path = os.path.join(PROCESSED_DATA_ROOT, dataset_name)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{dataset_name.lower()}{suffix}_checkpoint.pth")
    save_path = os.path.join(RESULT_DIR, f"{dataset_name}{suffix}")
    create_dir(save_path)

    test_x = sorted(glob(os.path.join(data_path, "test", "image", "*")))
    test_y = sorted(glob(os.path.join(data_path, "test", "mask", "*")))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载血管分割模型
    model = get_model(model_type).to(device)
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found for {dataset_name}")
        return
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # 初始化视盘分割模型
    od_segmenter = ODSegmenter(device=device) if use_od else None

    metrics_score = [0.0] * 6

    for i, (x_p, y_p) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = os.path.basename(x_p).split(".")[0]

        # 读取数据
        image_bgr = cv2.imread(x_p, cv2.IMREAD_COLOR)
        x = np.transpose(image_bgr, (2, 0, 1)) / 255.0
        x = np.expand_dims(x, axis=0).astype(np.float32)
        x = torch.from_numpy(x).to(device)

        mask_gray = cv2.imread(y_p, cv2.IMREAD_GRAYSCALE)
        y = np.expand_dims(mask_gray, axis=0) / 255.0
        y = np.expand_dims(y, axis=0).astype(np.float32)
        y = torch.from_numpy(y).to(device)

        with torch.no_grad():
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)

            score = calculate_metrics(y, pred_y)
            metrics_score = list(map(add, metrics_score, score))

            pred_y_np = pred_y[0].cpu().numpy().squeeze(0)
            pred_binary = (pred_y_np > 0.5).astype(np.uint8)

            # 视盘预测
            od_mask = od_segmenter.process(image_bgr) if use_od else np.zeros_like(mask_gray)

        # 绘图
        gt_show = mask_to_bgr(mask_gray)
        od_show = mask_to_bgr(od_mask)
        gt_binary = (mask_gray > 127).astype(np.uint8)
        error_map = draw_error_map(gt_binary, pred_binary)

        # 拼接: 原图 | 真值 | OD | 误差图
        h, w, _ = image_bgr.shape
        sep = np.ones((h, 10, 3), dtype=np.uint8) * 255
        final_img = np.concatenate([image_bgr, sep, gt_show, sep, od_show, sep, error_map], axis=1)

        cv2.imwrite(os.path.join(save_path, f"{name}.png"), final_img)

    # 输出结果
    count = len(test_x)
    jaccard = metrics_score[0] / count
    f1 = metrics_score[1] / count
    sensitivity = metrics_score[2] / count
    precision = metrics_score[3] / count
    acc = metrics_score[4] / count
    specificity = metrics_score[5] / count

    print(f"\n[{dataset_name} Result]")
    print(f"  IoU (Jaccard) : {jaccard:.4f}")
    print(f"  F1 (Dice)     : {f1:.4f}")
    print(f"  Sensitivity   : {sensitivity:.4f}")
    print(f"  Precision     : {precision:.4f}")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  Specificity   : {specificity:.4f}")
    print(f"  Images saved to: {save_path}")

    del model
    if od_segmenter:
        del od_segmenter
    torch.cuda.empty_cache()

    return {
        'jaccard': jaccard, 'f1': f1, 'sensitivity': sensitivity,
        'precision': precision, 'accuracy': acc, 'specificity': specificity
    }


def run_comparative_test(dataset_name, od_intensity=60, dist_out=30, dist_in=10, model_type='baseline'):
    """
    执行对比测试 (Baseline vs OD优化)

    Args:
        dataset_name: 数据集名称
        od_intensity: OD边缘增强强度
        dist_out: 向外距离参数
        dist_in: 向内距离参数
        model_type: 模型类型 ('baseline', 'dcn_sp', 'aspp')
    """
    print(f"\n{'=' * 60}")
    print(f"Running COMPARATIVE Test: {dataset_name}")
    print(f"  Model: {model_type}")
    print(f"{'=' * 60}")

    suffix = f"_{model_type}" if model_type != 'baseline' else ""
    data_path = os.path.join(PROCESSED_DATA_ROOT, dataset_name)
    save_path = os.path.join(RESULT_DIR, f"{dataset_name}{suffix}_Comparative")
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{dataset_name.lower()}{suffix}_checkpoint.pth")
    create_dir(save_path)

    test_x = sorted(glob(os.path.join(data_path, "test", "image", "*")))
    test_y = sorted(glob(os.path.join(data_path, "test", "mask", "*")))

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = get_model(model_type).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    od_segmenter = ODSegmenter(device=device)

    metrics_base = [0.0] * 6
    metrics_opt = [0.0] * 6

    for x_path, y_path in tqdm(zip(test_x, test_y), total=len(test_x)):
        name = os.path.basename(x_path).split(".")[0]

        img_orig = cv2.imread(x_path)
        mask_gt = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
        y_tensor = torch.from_numpy(np.expand_dims(np.expand_dims(mask_gt, 0), 0) / 255.0).float().to(device)

        # Path A: Baseline
        x_base = np.transpose(img_orig, (2, 0, 1)) / 255.0
        x_base_t = torch.from_numpy(np.expand_dims(x_base, 0).astype(np.float32)).to(device)

        with torch.no_grad():
            pred_base = torch.sigmoid(model(x_base_t))
            metrics_base = list(map(add, metrics_base, calculate_metrics(y_tensor, pred_base)))
            pred_base_np = (pred_base[0].cpu().numpy().squeeze(0) > 0.5).astype(np.uint8)

        # Path B: OD Optimized
        od_mask = od_segmenter.process(img_orig)
        img_opt = harmonize_od_edge(
            img_orig, od_mask,
            dist_out_px=dist_out, gamma_out=1.0,
            dist_in_px=dist_in, gamma_in=4.0,
            intensity=od_intensity
        )

        x_opt = np.transpose(img_opt, (2, 0, 1)) / 255.0
        x_opt_t = torch.from_numpy(np.expand_dims(x_opt, 0).astype(np.float32)).to(device)

        with torch.no_grad():
            pred_opt = torch.sigmoid(model(x_opt_t))
            metrics_opt = list(map(add, metrics_opt, calculate_metrics(y_tensor, pred_opt)))
            pred_opt_np = (pred_opt[0].cpu().numpy().squeeze(0) > 0.5).astype(np.uint8)

        # 可视化 (6列对比)
        col1 = add_label(img_orig, "Original")
        col2 = add_label(mask_to_bgr(od_mask), "OD Mask")
        col3 = add_label(draw_error_map((mask_gt > 127), pred_base_np), "Base Error")
        col4 = add_label(img_opt, "Processed")
        col5 = add_label(draw_error_map((mask_gt > 127), pred_opt_np), "Opt Error")
        col6 = add_label(mask_to_bgr(mask_gt), "GT")

        h = col1.shape[0]
        sep = np.ones((h, 5, 3), dtype=np.uint8) * 100
        final_row = np.concatenate([col1, sep, col2, sep, col3, sep, col4, sep, col5, sep, col6], axis=1)

        cv2.imwrite(os.path.join(save_path, f"{name}.png"), final_row)

    # 打印结果
    count = len(test_x)

    def print_metrics(name, m):
        print(f"--- {name} ---")
        print(f"  IoU (Jaccard): {m[0] / count:.4f}")
        print(f"  F1 (Dice)    : {m[1] / count:.4f}")
        print(f"  Sensitivity  : {m[2] / count:.4f}")
        print(f"  Precision    : {m[3] / count:.4f}")
        print(f"  Accuracy     : {m[4] / count:.4f}")
        print(f"  Specificity  : {m[5] / count:.4f}")

    print(f"\n>>> [{dataset_name}] RESULTS <<<")
    print_metrics("Baseline", metrics_base)
    print_metrics("Optimized", metrics_opt)

    diff_iou = (metrics_opt[0] - metrics_base[0]) / count
    diff_sens = (metrics_opt[2] - metrics_base[2]) / count
    diff_prec = (metrics_opt[3] - metrics_base[3]) / count

    print(f"\n[Improvement]")
    print(f"  IoU Change        : {diff_iou:+.4f}")
    print(f"  Sensitivity Change: {diff_sens:+.4f}")
    print(f"  Precision Change  : {diff_prec:+.4f}")
    print(f"  Images saved to: {save_path}\n")

    del model, od_segmenter
    torch.cuda.empty_cache()
