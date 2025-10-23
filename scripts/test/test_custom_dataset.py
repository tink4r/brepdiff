#!/usr/bin/env python
"""
测试脚本：验证 custom_step 数据集能否正确加载
"""
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import torch
from brepdiff.config import load_config
from brepdiff.datasets import DATASETS

def test_custom_dataset():
    print("=" * 60)
    print("测试 Custom Step 数据集加载")
    print("=" * 60)
    
    # 加载配置
    config = load_config("configs/custom_overfit.yaml", "")
    print(f"\n✓ 成功加载配置文件: configs/custom_overfit.yaml")
    print(f"  - dataset: {config.dataset}")
    print(f"  - h5_path: {config.h5_path}")
    print(f"  - overfit: {config.overfit}")
    print(f"  - overfit_data_size: {config.overfit_data_size}")
    
    # 获取数据集类
    dataset_cls = DATASETS.get(config.dataset)
    if dataset_cls is None:
        raise ValueError(f"找不到数据集: {config.dataset}")
    print(f"\n✓ 找到数据集类: {dataset_cls.__name__}")
    
    # 创建训练集
    print("\n创建训练集...")
    train_dataset = dataset_cls(config, split="train")
    print(f"✓ 训练集创建成功")
    print(f"  - 数据集大小: {len(train_dataset)}")
    
    # 测试加载第一个样本
    print("\n加载第一个样本...")
    sample = train_dataset[0]
    
    print(f"✓ 样本加载成功！")
    print(f"\n样本信息:")
    print(f"  - uvgrid.coord: shape={sample.uvgrid.coord.shape}")
    print(f"  - uvgrid.normal: shape={sample.uvgrid.normal.shape}")
    print(f"  - uvgrid.grid_mask: shape={sample.uvgrid.grid_mask.shape}")
    
    # 测试 DataLoader
    print("\n测试 DataLoader...")
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # 测试时使用 0
        drop_last=False,
    )
    
    batch = next(iter(train_loader))
    print(f"✓ DataLoader 工作正常")
    print(f"\nBatch 信息:")
    print(f"  - batch size: {len(batch)}")
    print(f"  - uvgrid.coord: shape={batch[0].uvgrid.coord.shape}")
    print(f"  - uvgrid.grid_mask: shape={batch[0].uvgrid.grid_mask.shape}")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！数据集配置正确，可以开始训练。")
    print("=" * 60)

if __name__ == "__main__":
    test_custom_dataset()
