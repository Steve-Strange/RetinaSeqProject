import os
import cv2
import torch
import numpy as np
import glob
import pandas as pd
from tqdm import tqdm
import segmentation_models_pytorch as smp
import traceback

# 引入你的自定义模块
from src.model import UNet as CustomUNet
from src.transforms import RetinaPreprocess
from src.utils import calculate_metrics

# === 配置区域 ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "./checkpoints"
DATA_DIR = "./data/unified_data"
RESULTS_DIR = "./evaluation_results"

# 映射文件名中的关键词到模型架构
ARCH_MAP = {
    "custom": "custom",
    "resnet34": "resnet34",
    "resnet50": "resnet50",
    "efficientnet": "efficientnet-b4",
    "transunet": "transunet"
}

def save_comparison_row(save_path, original_rgb, gt_mask, pred_binary, dice_score):
    """
    生成横向对比图：[原图 | 真值 | 预测 | 差异图]
    """
    # 1. 准备各个分量 (统一转为 BGR 3通道，方便 OpenCV 拼接)
    
    # A. 原图 (RGB -> BGR)
    vis_orig = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    
    # B. 真值 (Gray -> BGR)
    vis_gt = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
    
    # C. 预测 (Gray -> BGR)
    vis_pred = cv2.cvtColor(pred_binary, cv2.COLOR_GRAY2BGR)
    
    # D. 差异图 (Diff Map)
    vis_diff = np.zeros_like(vis_orig)
    
    # TP (白色): 预测对的
    vis_diff[(pred_binary==255) & (gt_mask==255)] = [255, 255, 255]
    # FP (蓝色): 误检 (OpenCV BGR: 255, 0, 0)
    vis_diff[(pred_binary==255) & (gt_mask==0)] = [255, 0, 0]
    # FN (红色): 漏检 (OpenCV BGR: 0, 0, 255)
    vis_diff[(pred_binary==0) & (gt_mask==255)] = [0, 0, 255]

    # --- 添加文字标题 ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thick = 2
    color = (200, 255, 200) # 淡绿色文字，对比度高
    pos = (20, 40)

    cv2.putText(vis_orig, "Original", pos, font, scale, color, thick)
    cv2.putText(vis_gt,   "Ground Truth", pos, font, scale, color, thick)
    cv2.putText(vis_pred, f"Pred (Dice:{dice_score:.2f})", pos, font, scale, color, thick)
    # 差异图文字做小一点
    cv2.putText(vis_diff, "Diff (B:FP, R:FN)", (10, 40), font, 0.7, color, thick)

    # 2. 横向拼接
    # 在图片之间加一条细白线，方便区分
    separator = np.ones((vis_orig.shape[0], 5, 3), dtype=np.uint8) * 128
    
    combined_row = np.hstack([
        vis_orig, separator, 
        vis_gt, separator, 
        vis_pred, separator, 
        vis_diff
    ])
    
    # 3. 保存
    cv2.imwrite(save_path, combined_row)

def load_model(model_path):
    """智能加载模型"""
    fname = os.path.basename(model_path)
    
    # 1. 推断架构
    arch = None
    for key, value in ARCH_MAP.items():
        if key in fname:
            arch = value
            break
    
    if arch is None:
        if "best_model.pth" in fname:
            print(f"⚠️ 识别到旧版默认文件名 {fname}，强制指定架构为 'custom'")
            arch = "custom"
        else:
            print(f"⚠️ 警告: 无法从文件名 {fname} 识别架构，尝试默认使用 resnet34")
            arch = "resnet34"
        
    # 2. 推断通道数 (OD Guidance)
    use_od = True 
    if "od0" in fname or "no_od" in fname:
        use_od = False
    if fname == "best_model.pth":
        use_od = True

    in_channels = 4 if use_od else 3
    print(f"Loading {fname} -> Arch: {arch}, Input: {in_channels}ch")
    
    # 3. 构建模型
    if arch == "custom":
        model = CustomUNet(in_channels=in_channels, num_classes=1)
    elif arch == "transunet":
        from src.model_transunet import TransUNet
        model = TransUNet(img_size=512, in_channels=in_channels, out_channels=1)
    else:
        model = smp.Unet(
            encoder_name=arch,        
            encoder_weights=None, 
            in_channels=in_channels,                  
            classes=1,
            activation=None
        )
        
    # 4. 加载权重
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
        
    try:
        model.load_state_dict(new_state_dict)
    except RuntimeError as e:
        print(f"❌ 权重加载失败！架构不匹配。")
        raise e
    
    model.to(DEVICE)
    model.eval()
    
    return model, use_od, fname

def preprocess_sample(img_path, od_path, use_od):
    """预处理单张图片"""
    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    if os.path.exists(od_path):
        od_mask = cv2.imread(od_path, cv2.IMREAD_GRAYSCALE)
    else:
        od_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        
    harmonized = RetinaPreprocess.harmonize_od_color(rgb, od_mask)
    feature_stack = RetinaPreprocess.get_green_enhanced_stack(harmonized)
    
    input_img = feature_stack.astype(np.float32) / 255.0
    od_input = od_mask.astype(np.float32) / 255.0
    
    img_tensor = torch.from_numpy(input_img).permute(2, 0, 1)
    od_tensor = torch.from_numpy(od_input).unsqueeze(0)
    
    if use_od:
        input_tensor = torch.cat([img_tensor, od_tensor], dim=0)
    else:
        input_tensor = img_tensor
        
    return input_tensor.unsqueeze(0).to(DEVICE), rgb, od_mask

def evaluate_single_model(model_path):
    """评估单个模型"""
    model, use_od, model_name = load_model(model_path)
    
    save_vis_dir = os.path.join(RESULTS_DIR, os.path.splitext(model_name)[0])
    os.makedirs(save_vis_dir, exist_ok=True)
    
    all_imgs = sorted(glob.glob(os.path.join(DATA_DIR, "images", "*.png")))
    test_imgs = [p for p in all_imgs if "test" in os.path.basename(p)]
    if len(test_imgs) == 0: test_imgs = all_imgs[:20] 
    
    metrics = []
    
    print(f"--> 正在评估 {model_name} (共 {len(test_imgs)} 张)...")
    
    with torch.no_grad():
        for img_p in tqdm(test_imgs):
            fname = os.path.basename(img_p)
            mask_p = os.path.join(DATA_DIR, "masks", fname)
            od_p = os.path.join(DATA_DIR, "od_masks", fname)
            
            input_tensor, rgb, od_mask = preprocess_sample(img_p, od_p, use_od)
            
            gt = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
            gt_tensor = torch.from_numpy(gt / 255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            logits = model(input_tensor)
            
            # 计算指标
            scores = calculate_metrics(logits, gt_tensor)
            metrics.append(scores) 
            dice_score = scores[1]
            
            # 可视化准备
            pred_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_binary = (pred_prob > 0.5).astype(np.uint8) * 255
            
            # 调用新的横向可视化函数
            save_path = os.path.join(save_vis_dir, f"vis_{fname}")
            save_comparison_row(save_path, rgb, gt, pred_binary, dice_score)
            
    avg_scores = np.mean(metrics, axis=0)
    return {
        "Model": model_name,
        "IoU": avg_scores[0],
        "Dice": avg_scores[1],
        "Acc": avg_scores[2],
        "Sen": avg_scores[3],
        "Spe": avg_scores[4]
    }

def main():
    model_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth")))
    if not model_files:
        print("未找到模型文件")
        return
    
    print(f"找到 {len(model_files)} 个模型: {[os.path.basename(f) for f in model_files]}")
    
    all_results = []
    
    for pth in model_files:
        try:
            res = evaluate_single_model(pth)
            all_results.append(res)
        except Exception as e:
            print(f"\n❌ 模型 {os.path.basename(pth)} 评估失败，跳过。")
            print(f"错误信息: {str(e)}")
            continue
        
    if not all_results:
        return

    df = pd.DataFrame(all_results)
    print("\n========= 最终对比排行榜 =========")
    df = df.sort_values(by="Dice", ascending=False)
    print(df.to_string(index=False))
    
    df.to_csv(os.path.join(RESULTS_DIR, "leaderboard.csv"), index=False)
    print(f"\n评估完成！结果已保存至 {RESULTS_DIR}")

if __name__ == "__main__":
    main()