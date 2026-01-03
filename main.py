import os
import torch.multiprocessing as mp
from src.utils import seeding

# 定义数据集与 GPU ID 的映射关系
# 格式: "数据集名称": GPU_ID
GPU_MAPPING = {
    "DRIVE": 1,
    "CHASE_DB": 6,
    "STARE": 7
}

def process_wrapper(dataset_name, gpu_id):
    """
    这是独立的子进程函数。
    它会设置自己的环境变量，然后导入训练代码并执行。
    """
    try:
        # === 关键步骤：在进程启动最开始，指定该进程可见的显卡 ===
        # 这样对 PyTorch 来说，这张卡就是 'cuda:0'，代码不需要做任何修改
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        print(f"--- [Process Start] Dataset: {dataset_name} on GPU: {gpu_id} ---")
        
        # === 延迟导入 ===
        # 必须在设置完环境变量后导入，防止 PyTorch 提前初始化 CUDA
        from src.augment import run_augmentation
        from train import run_training
        from test import run_testing
        import gc
        import torch
        
        # 设定随机种子
        seeding(42)

        # 1. 数据增强
        run_augmentation(dataset_name)
        
        # 2. 训练
        run_training(dataset_name)
        
        # 3. 测试
        run_testing(dataset_name)
        
        # 清理
        gc.collect()
        torch.cuda.empty_cache()
        print(f"--- [Process Finish] Dataset: {dataset_name} Done. ---")
        
    except Exception as e:
        print(f"!!! [Error] Dataset {dataset_name} failed on GPU {gpu_id}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 设置启动方法为 spawn，这对 CUDA 多进程是必须的
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print(f"Starting Multi-GPU Training Tasks...")
    print(f"Mapping: {GPU_MAPPING}")

    processes = []

    # 为每个数据集创建一个进程
    for dataset_name, gpu_id in GPU_MAPPING.items():
        p = mp.Process(target=process_wrapper, args=(dataset_name, gpu_id))
        p.start()
        processes.append(p)

    # 等待所有进程结束
    for p in processes:
        p.join()

    print("\nAll datasets processed successfully.")