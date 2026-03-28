# 验证集Dice不变动诊断向导

## 问题
验证Dice全程保持 0.744 不变，不随模型参数更新。

## 诊断步骤

### Step 1: 检查数据划分（快速诊断）
```bash
python debug_dataset_split.py
```

**预期输出（20个样本，20%验证）：**
```
Total files found: 20
split_num = 5, val_split = 0.2

Train set (split_idx=0, step=5):
  Indices: [0, 5, 10, 15]
  Files: ['la_xxx.nii.gz', ...]
  Count: 4

Validation set (split_idx=1, step=5):
  Indices: [1, 6, 11, 16]
  Files: ['la_yyy.nii.gz', ...]
  Count: 4

✓ No overlap between train and val
```

### Step 2: 运行带调试的训练（5个epoch快速测试）
```bash
python train.py \
 --batch_size 1 \
 --num_workers 2 \
 --task_name "ft_heart_debug" \
 --checkpoint "sam_med3d_turbo.pth" \
 --lr 8e-5 \
 --num_epochs 5 \
 --gpu_ids 0 \
 --img_size 128 \
 --val_split 0.2 \
 --val_interval 1 \
 --eval_num_clicks 2
```

**关键参数说明：**
- `--batch_size 1`：单个样本验证，避免平均效应
- `--val_interval 1`：每个epoch验证
- `--eval_num_clicks 2`：快速验证（只2次迭代）

### Step 3: 观察输出

**看这些关键行：**
```
[DEBUG] Data split info:
  - Train dataset size: 16          ← 16个训练样本
  - Val dataset size: 4             ← 4个验证样本

[DEBUG] val_dataloaders info:
  - Dataloader length: 4            ← 4个批次（batch_size=1）
  - Dataset length: 4

[DEBUG] Batch 0: Dice = 0.xxx       ← 每个验证样本的Dice
[DEBUG] Batch 1: Dice = 0.yyy
[DEBUG] Batch 2: Dice = 0.zzz
[DEBUG] Batch 3: Dice = 0.www
[DEBUG] Validation complete: processed 4 batches, avg Dice = 0.744
```

## 诊断结果判定

### 情况A：验证集为空（Val dataset size: 0）
```
❌ 问题：数据划分失败
✅ 解决方案：
   1. 检查 datasets/Task02_Heart/labelsTr 里是否有文件
   2. 确保 img_datas 配置正确指向 Task02_Heart
```

### 情况B：验证集有数据但Dice全是 0.744
```
⚠️  可能原因：
   1. 验证集太小（只有4个样本）→ 每个样本的Dice更新得慢
   2. 模型初始化后，加载到验证集时表现就是0.744（巧合）
   3. 验证循环中的梯度计算有问题

✅ 解决方案：
   1. 增加验证集比例：--val_split 0.5（50%）
   2. 检查训练Dice是否在变化（应该在变）
   3. 如果训练Dice变化，验证Dice不变 → 过拟合问题
```

### 情况C：验证集Dice逐批次变化，但平均值固定
```
这可能意味着验证循环每次重新初始化为相同的起点
```

## 如果问题依然存在

运行这个更详细的诊断：
```bash
python train.py \
 --batch_size 1 \
 --num_workers 0 \
 --task_name "ft_heart_full_debug" \
 --checkpoint "sam_med3d_turbo.pth" \
 --lr 8e-5 \
 --num_epochs 10 \
 --gpu_ids 0 \
 --img_size 128 \
 --val_split 0.5 \
 --val_interval 1 \
 --eval_num_clicks 3
```

并将输出上传，特别是标有 `[DEBUG]` 的部分。

## 预期正常行为

**训练Dice应该逐步增加：**
```
EPOCH: 0, Train Dice: 0.860
EPOCH: 1, Train Dice: 0.865
EPOCH: 2, Train Dice: 0.870
```

**验证Dice应该也逐步增加（可能比训练更平缓）：**
```
EPOCH: 0, Val Dice: 0.744  ← 初始值
EPOCH: 1, Val Dice: 0.756  ← 逐步上升
EPOCH: 2, Val Dice: 0.768
```

如果验证Dice不变但训练Dice变化 → **过拟合** (这是正常现象，实际可能是数据集太小)
