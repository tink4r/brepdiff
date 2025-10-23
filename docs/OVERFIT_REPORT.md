# 🎯 Custom Overfit 训练配置检查报告

## ✅ 检查结果总结

所有配置已通过验证，可以开始训练！

---

## 📋 已完成的配置检查

### 1. ✅ 数据文件
- **HDF5 文件**: `data/custom_uvgrid/custom.h5` ✓
- **训练列表**: `data/custom_uvgrid/custom_train.txt` ✓  
- **Invalid 列表**: `data/custom_uvgrid/custom_invalid.txt` ✓
- **数据样本**: `00031214_9643da4778ab413381cc7038_step_006_step`
- **面数统计**: 116 个面

### 2. ✅ 数据集类
- **类名**: `CustomStepDataset` 
- **继承**: `ABCDataset`
- **已注册**: 在 `DATASETS` 字典中 ✓

### 3. ✅ 训练配置 (custom_overfit.yaml)

#### 关键修改项：
```yaml
dataset: custom_step                    # ✓ 使用自定义数据集
h5_path: "./data/custom_uvgrid/custom.h5"  # ✓ 正确路径
max_n_prims: 150                       # ✓ 已调整（原50 → 150）支持116个面
overfit: True                           # ✓ 启用 overfit 模式
overfit_data_size: 1                    # ✓ 只使用1个样本
overfit_data_repetition: 2048           # ✓ 重复2048次
batch_size: 16                          # ✓ 适合单样本训练
num_epochs: 200                         # ✓ 足够 overfit
```

#### 其他重要配置：
- **学习率**: 0.0005 (适中)
- **优化器**: AdamW with weight decay
- **梯度裁剪**: max_norm=0.5
- **验证频率**: val_epoch=50
- **可视化**: 每1000步

### 4. ✅ 测试验证
运行 `test_custom_dataset.py` 结果：
```
✓ 成功加载配置文件
✓ 找到数据集类
✓ 训练集创建成功 (2048 样本，因为重复了2048次)
✓ 样本加载成功
  - uvgrid.coord: shape=(150, 8, 8, 3)
  - uvgrid.normal: shape=(150, 8, 8, 3)
  - uvgrid.grid_mask: shape=(150, 8, 8)
```

---


## 🚀 如何启动训练

### **推荐的工作流**

# 1. 数据处理
```bash
python scripts/visualization/vis_step.py my_raw_data/*.step --output-dir data/custom_uvgrid
```

# 2. 转换为 HDF5
```bash
python -m scripts.postprocessing.npz_to_h5 \
    data/custom_uvgrid/npz_for_vis \
    data/custom_uvgrid/custom.h5 \
    --list-path data/custom_uvgrid/custom_train.txt
```

# 3. 创建 invalid 文件（如果不存在）
```bash
touch data/custom_uvgrid/custom_invalid.txt
```


# 4. 启动训练
### 方法 1: 使用提供的脚本（推荐）
```bash
cd /data5/pengyunqiu/proj/brepdiff
./scripts/train_custom_overfit.sh
```

### 方法 2: 直接使用 Python 命令
```bash
cd /data5/pengyunqiu/proj/brepdiff

python -m scripts.run \
    ./logs/custom_overfit/run_001 \
    --config-path ./configs/custom_overfit.yaml \
    --wandb-offline
```

### 方法 3: 使用自定义参数覆盖
```bash
./scripts/train_custom_overfit.sh \
    --override "batch_size=8|num_epochs=500"
```



---

## 📊 训练监控

### 检查点位置
- **日志目录**: `logs/custom_overfit/<timestamp>/`
- **模型检查点**: `logs/custom_overfit/<timestamp>/ckpts/`
- **配置备份**: `logs/custom_overfit/<timestamp>/config.yaml`

### 可视化
- **训练可视化**: `logs/custom_overfit/<timestamp>/vis/`
- **验证可视化**: 每50个epoch生成一次
- **Tensorboard**: 可以使用 `tensorboard --logdir logs/custom_overfit/`

### 训练日志
训练过程中会输出：
- Loss 值（coord loss + grid_mask loss）
- 验证 loss
- 学习率
- 训练速度 (it/s)

---

## 🎯 Overfit 成功的判断标准

### 预期行为
1. **训练 loss 持续下降**
   - 应该降到非常低的值 (< 0.01)
   - 如果不降可能是学习率或模型容量问题

2. **验证 loss = 训练 loss**
   - 因为只有一个样本，两者应该相同

3. **生成结果应该与输入非常相似**
   - 在 `vis/` 目录查看生成的可视化
   - 生成的 UV-grid 应该能重建原始模具形状

### 如果 Overfit 失败
- **Loss 不下降**: 
  - 检查学习率是否太小
  - 增加模型容量（增大 hidden_size, depth）
- **Loss 震荡**:
  - 减小学习率
  - 减小 batch_size
- **显存不足**:
  - 减小 batch_size
  - 减小 max_n_prims

---

## 🔧 常见问题排查

### 1. 显存不足 (OOM)
```bash
# 修改配置降低显存占用
batch_size: 8  # 减小
max_n_prims: 120  # 如果可能，稍微减小
```

### 2. 训练太慢
```bash
# 使用更少的 worker
num_workers: 4  # 从8减到4
```

### 3. Wandb 错误
```bash
# 确保使用 offline 模式
--wandb-offline
```

### 4. CUDA 错误
```bash
# 确认 GPU 可用
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📝 配置文件摘要

### 已修复的问题
1. ✅ `max_n_prims: 50 → 150` (支持116个面的模具)
2. ✅ 创建了 `custom_invalid.txt` 文件
3. ✅ 数据集正确注册和配置
4. ✅ Overfit 参数正确设置

### 配置亮点
- **快速验证**: 200 epochs 应该足够看到 overfit
- **合理的 batch size**: 16 个样本每批次
- **适中的学习率**: 0.0005 with AdamW
- **完整的可视化**: 训练和验证都会生成可视化

---

## 🎉 下一步

1. **启动训练**: `./scripts/train_custom_overfit.sh`
2. **监控进度**: 观察终端输出的 loss 值
3. **检查可视化**: 查看 `logs/custom_overfit/<timestamp>/vis/` 目录
4. **等待 overfit**: 训练约1-2小时（取决于 GPU）

---

## 💡 提示

- 这是一个 **overfit 测试**，目的是验证模型能否记住单个样本
- 如果 overfit 成功，说明：
  ✅ 数据格式正确
  ✅ 模型架构合理
  ✅ 训练流程正常
  ✅ 可以开始在完整数据集上训练

祝训练顺利！🚀
