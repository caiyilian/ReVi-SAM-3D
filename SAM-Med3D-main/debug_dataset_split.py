"""Debug script to check dataset splitting"""
import os
from glob import glob

# 直接计算会分配哪些文件
dataset_root = "datasets/Task02_Heart"
label_dir = os.path.join(dataset_root, "labelsTr")

# 获取所有标签文件
files = sorted([f for f in os.listdir(label_dir) if f.endswith('.nii.gz')])
print(f"Total files found: {len(files)}")
print(f"Files: {files}")

# 模拟数据划分
val_split = 0.2
split_num = max(1, int(1.0 / val_split)) if val_split > 0 else 1
print(f"\nsplit_num = {split_num}, val_split = {val_split}")

# 训练集（split_idx=0）
train_indices = list(range(0, len(files), split_num))
train_files = [files[i] for i in train_indices if i < len(files)]
print(f"\nTrain set (split_idx=0, step={split_num}):")
print(f"  Indices: {train_indices}")
print(f"  Files: {train_files}")
print(f"  Count: {len(train_files)}")

# 验证集（split_idx=1）  
val_indices = list(range(1, len(files), split_num))
val_files = [files[i] for i in val_indices if i < len(files)]
print(f"\nValidation set (split_idx=1, step={split_num}):")
print(f"  Indices: {val_indices}")
print(f"  Files: {val_files}")
print(f"  Count: {len(val_files)}")

# 检查是否有重叠
overlap = set(train_files) & set(val_files)
if overlap:
    print(f"\n⚠️  WARNING: Overlap found! {overlap}")
else:
    print(f"\n✓ No overlap between train and val")
