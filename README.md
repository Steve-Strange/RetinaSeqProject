### 数据统一、视盘分割

python prepare_data.py

python src/preprocess_od.py


### 数据增强、增强效果可视化

python generate_final_dataset.py

python visualize_augmentations.py

### 模型训练

python train.py --model_arch custom --no_od --batch_size 64

python train.py --model_arch resnet34 --batch_size 64

python train.py --model_arch efficientnet-b4 --batch_size 48

pkill -9 python