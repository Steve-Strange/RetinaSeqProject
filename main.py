#!/usr/bin/env python3
"""
RetinaSegProject - 视网膜血管分割项目

统一命令行入口，支持以下功能：
  --prepare    数据预处理和增强
  --train      模型训练
  --test       模型测试
  --compare    对比测试 (Baseline vs OD优化)
  --all        完整流程 (prepare + train + test)

使用示例:
  python main.py --prepare --dataset DRIVE
  python main.py --train --dataset DRIVE --epochs 100
  python main.py --test --dataset DRIVE
  python main.py --compare --dataset DRIVE
  python main.py --all --dataset DRIVE,CHASE_DB,STARE
  python main.py --all --dataset all --parallel
"""
import os
import argparse
import torch.multiprocessing as mp

from src.utils import seeding, create_dir
from src.config import TARGET_DATASETS


def parse_args():
    parser = argparse.ArgumentParser(
        description="RetinaSegProject - 视网膜血管分割",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --prepare --dataset DRIVE
  python main.py --train --dataset DRIVE --epochs 100 --batch-size 32
  python main.py --test --dataset DRIVE
  python main.py --compare --dataset DRIVE --od-intensity 60
  python main.py --all --dataset all
  python main.py --all --dataset all --parallel --gpus 1,6,7
        """
    )

    # 主要命令
    cmd_group = parser.add_argument_group("Commands (至少选择一个)")
    cmd_group.add_argument("--prepare", action="store_true", help="数据预处理和增强")
    cmd_group.add_argument("--train", action="store_true", help="训练模型")
    cmd_group.add_argument("--test", action="store_true", help="测试模型")
    cmd_group.add_argument("--compare", action="store_true", help="对比测试 (Baseline vs OD优化)")
    cmd_group.add_argument("--all", action="store_true", help="完整流程 (prepare + train + test)")

    # 数据集选择
    data_group = parser.add_argument_group("Dataset Options")
    data_group.add_argument("--dataset", type=str, default="DRIVE",
                            help="数据集名称，多个用逗号分隔，或使用 'all' (默认: DRIVE)")

    # 模型和损失函数选择
    model_group = parser.add_argument_group("Model & Loss Options")
    model_group.add_argument("--model", type=str, default="baseline",
                             choices=["baseline", "dcn_sp", "aspp"],
                             help="模型类型: baseline, dcn_sp (DCN+StripPooling), aspp (默认: baseline)")
    model_group.add_argument("--loss", type=str, default="vessel",
                             choices=["vessel", "dice", "dice_bce", "bce", "focal_tversky"],
                             help="损失函数: vessel, dice, dice_bce, bce, focal_tversky (默认: vessel)")

    # 训练参数
    train_group = parser.add_argument_group("Training Options")
    train_group.add_argument("--epochs", type=int, default=None, help="训练轮数 (默认: 200)")
    train_group.add_argument("--batch-size", type=int, default=None, help="批次大小 (默认: 64)")
    train_group.add_argument("--lr", type=float, default=None, help="学习率 (默认: 1e-3)")

    # 测试参数
    test_group = parser.add_argument_group("Testing Options")
    test_group.add_argument("--no-od", action="store_true", help="测试时禁用视盘分割")
    test_group.add_argument("--od-intensity", type=int, default=60, help="OD边缘增强强度 (默认: 60)")

    # 并行参数
    parallel_group = parser.add_argument_group("Parallel Options")
    parallel_group.add_argument("--parallel", action="store_true", help="多GPU并行处理不同数据集")
    parallel_group.add_argument("--gpus", type=str, default="0,1,2",
                                help="GPU ID列表，用逗号分隔 (默认: 0,1,2)")

    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")

    return parser.parse_args()


def get_datasets(dataset_arg):
    """解析数据集参数"""
    if dataset_arg.lower() == "all":
        return TARGET_DATASETS
    return [d.strip().upper() for d in dataset_arg.split(",")]


def process_single(dataset_name, args, gpu_id=None):
    """处理单个数据集"""
    # 设置GPU
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"\n[Process] Dataset: {dataset_name} on GPU: {gpu_id}")

    # 延迟导入
    from src.augment import run_augmentation
    from src.trainer import run_training
    from src.tester import run_testing, run_comparative_test
    import gc
    import torch

    seeding(args.seed)

    try:
        # 数据预处理
        if args.prepare or args.all:
            print(f"\n{'=' * 50}")
            print(f"[Step 1] Preparing data for {dataset_name}")
            print(f"{'=' * 50}")
            run_augmentation(dataset_name)

        # 训练
        if args.train or args.all:
            print(f"\n{'=' * 50}")
            print(f"[Step 2] Training {dataset_name}")
            print(f"{'=' * 50}")
            run_training(
                dataset_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                model_type=args.model,
                loss_type=args.loss
            )

        # 测试
        if args.test or args.all:
            print(f"\n{'=' * 50}")
            print(f"[Step 3] Testing {dataset_name}")
            print(f"{'=' * 50}")
            run_testing(dataset_name, use_od=not args.no_od, model_type=args.model)

        # 对比测试
        if args.compare:
            print(f"\n{'=' * 50}")
            print(f"[Compare] Running comparative test for {dataset_name}")
            print(f"{'=' * 50}")
            run_comparative_test(dataset_name, od_intensity=args.od_intensity, model_type=args.model)

        gc.collect()
        torch.cuda.empty_cache()
        print(f"\n[Done] {dataset_name} completed successfully!")

    except Exception as e:
        print(f"[Error] {dataset_name} failed: {e}")
        import traceback
        traceback.print_exc()


def process_wrapper(dataset_name, args, gpu_id):
    """多进程包装器"""
    process_single(dataset_name, args, gpu_id)


def main():
    args = parse_args()

    # 检查是否有命令
    if not any([args.prepare, args.train, args.test, args.compare, args.all]):
        print("Error: 请至少指定一个命令 (--prepare, --train, --test, --compare, --all)")
        print("使用 --help 查看帮助")
        return

    datasets = get_datasets(args.dataset)
    gpu_ids = [int(g) for g in args.gpus.split(",")]

    print("=" * 60)
    print("RetinaSegProject - 视网膜血管分割")
    print("=" * 60)
    print(f"Datasets: {datasets}")
    print(f"Model: {args.model} | Loss: {args.loss}")
    print(f"Commands: ", end="")
    cmds = []
    if args.prepare or args.all:
        cmds.append("prepare")
    if args.train or args.all:
        cmds.append("train")
    if args.test or args.all:
        cmds.append("test")
    if args.compare:
        cmds.append("compare")
    print(", ".join(cmds))
    print(f"Parallel: {args.parallel}")
    if args.parallel:
        print(f"GPUs: {gpu_ids}")
    print("=" * 60)

    if args.parallel and len(datasets) > 1:
        # 多进程并行
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        processes = []
        for i, dataset in enumerate(datasets):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            p = mp.Process(target=process_wrapper, args=(dataset, args, gpu_id))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # 串行处理
        for dataset in datasets:
            process_single(dataset, args)

    print("\n" + "=" * 60)
    print("All tasks completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
