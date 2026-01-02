import os
import argparse
import time
from glob import glob
import cv2
import torch
import numpy as np
from tqdm import tqdm
from operator import add
from src.model import LFANet
from src.utils import seeding, create_dir, calculate_metrics

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="output/checkpoints/best_model.pth")
    parser.add_argument("--data_path", type=str, default="output/processed_data/test")
    parser.add_argument("--save_path", type=str, default="output/test_results")
    parser.add_argument("--use_od_input", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    seeding(42)
    create_dir(args.save_path)
    
    test_x = sorted(glob(os.path.join(args.data_path, "image/*")))
    test_y = sorted(glob(os.path.join(args.data_path, "mask/*")))
    test_od = sorted(glob(os.path.join(args.data_path, "od_mask/*")))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_channels = 4 if args.use_od_input else 3
    
    model = LFANet(input_channels=input_channels)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()
    
    metrics_score = [0.0] * 6
    time_taken = []
    
    for i, (x_p, y_p, od_p) in tqdm(enumerate(zip(test_x, test_y, test_od)), total=len(test_x)):
        name = os.path.basename(x_p).split(".")[0]
        
        # Load Image (3ch feature stack)
        image = cv2.imread(x_p)
        image = image / 255.0
        
        # Load OD if needed
        if args.use_od_input:
            od = cv2.imread(od_p, 0) / 255.0
            od = np.expand_dims(od, -1)
            image = np.concatenate([image, od], axis=-1)
            
        x = np.transpose(image, (2, 0, 1)).astype(np.float32)
        x = torch.from_numpy(x).unsqueeze(0).to(device)
        
        # Load GT
        mask = cv2.imread(y_p, 0) / 255.0
        y = np.expand_dims(mask, 0).astype(np.float32)
        y = torch.from_numpy(y).unsqueeze(0).to(device)
        
        with torch.no_grad():
            start = time.time()
            pred = torch.sigmoid(model(x))
            time_taken.append(time.time() - start)
            
            score = calculate_metrics(y, pred)
            metrics_score = list(map(add, metrics_score, score))
            
            # Save
            pred_img = pred[0].cpu().numpy().squeeze() > 0.5
            pred_img = (pred_img * 255).astype(np.uint8)
            
            # Visualization: Original (Ch0 for visual) + GT + Pred
            # Note: image is feature stack, showing Ch0 (Green) is okay for visualization
            vis_img = (cv2.imread(x_p)[:,:,0]).astype(np.uint8) 
            vis_gt = (mask * 255).astype(np.uint8)
            
            cat_img = np.concatenate([vis_img, vis_gt, pred_img], axis=1)
            cv2.imwrite(os.path.join(args.save_path, f"{name}.png"), cat_img)

    fps = 1 / np.mean(time_taken)
    jaccard = metrics_score[0] / len(test_x)
    f1 = metrics_score[1] / len(test_x)
    print(f"Jaccard: {jaccard:.4f}, F1: {f1:.4f}, FPS: {fps:.2f}")