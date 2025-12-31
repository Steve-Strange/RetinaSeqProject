import torch
import cv2
import os
import matplotlib.pyplot as plt
from src.model import UNet
from src.transforms import get_val_transforms

# 配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model.pth"
TEST_IMG = "./data/unified_data/images/drive_01_test.png" # 替换你想测试的图片
OD_PATH = "./data/unified_data/od_masks/drive_01_test.png"
USE_OD = True

def predict():
    # 1. 加载模型
    in_c = 4 if USE_OD else 3
    model = UNet(in_channels=in_c).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 2. 读取数据
    original_img = cv2.imread(TEST_IMG)
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # 预处理流程 (需与 Dataset 一致)
    if USE_OD and os.path.exists(OD_PATH):
        od_mask = cv2.imread(OD_PATH, cv2.IMREAD_GRAYSCALE)
        # 这里为了演示简单，直接在 Albumentations 里转
    else:
        od_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Transform
    transforms = get_val_transforms()
    aug = transforms(image=img, mask0=od_mask)
    img_tensor = aug['image'].to(DEVICE)
    od_tensor = aug['mask0'].float().to(DEVICE).unsqueeze(0)
    
    # 构造输入
    if USE_OD:
        input_tensor = torch.cat([img_tensor, od_tensor], dim=0).unsqueeze(0)
    else:
        input_tensor = img_tensor.unsqueeze(0)
        
    # 3. 预测
    with torch.no_grad():
        pred = model(input_tensor)
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()
        
    pred_binary = (pred > 0.5).astype(np.uint8) * 255
    
    # 4. 可视化
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Input")
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.title("OD Mask (Guidance)")
    plt.imshow(od_mask, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title("Prediction")
    plt.imshow(pred_binary, cmap='gray')
    plt.savefig("prediction_result.png")
    print("结果已保存到 prediction_result.png")

if __name__ == "__main__":
    predict()