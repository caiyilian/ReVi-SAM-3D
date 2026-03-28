
import os
import nibabel as nib
import numpy as np
from PIL import Image

def normalize_intensity(x, clip_window=None):
    """
    归一化医学图像亮度到 [0, 1]（适配SAM2的ToTensor预期）。
    可以传入 clip_window=[min, max] 进行硬截断，
    如果不传，则使用 1% 和 99% 分位数进行截断。
    """
    if clip_window is not None:
        x = np.clip(x, clip_window[0], clip_window[1])
    else:
        # 使用分位数截断来处理异常值
        b = np.percentile(x, 99.0)
        t = np.percentile(x, 1.0)
        x = np.clip(x, t, b)
        
    # 防止除以 0
    if np.max(x) == np.min(x):
        return np.zeros_like(x, dtype=np.float32)
    
    # 线性缩放到 [0, 1]（而不是[-1, 1]）
    # 这样便于转为uint8 [0, 255]并输入到SAM2
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x.astype(np.float32)


img_path = r"datasets\Task02_Heart\imagesTr\la_003.nii.gz"
filename = os.path.basename(img_path)
        
# 1. 读取 Image

img_nii = nib.load(img_path)
img_array = img_nii.get_fdata()

# 确保通道顺序为 (D, H, W)，nibabel 默认可能是 (W, H, D)
# 这取决于具体的 nii 文件，通常需要转置
if img_array.shape[2] < img_array.shape[0] and img_array.shape[2] < img_array.shape[1]:
    img_array = np.transpose(img_array, (2, 1, 0)) # 将 (W, H, D) 转为 (D, H, W)
print(f"{img_array.shape=}") # img_array.shape=(D, 320, 320)

# 2. 读取 Label
label_path = r"datasets\Task02_Heart\labelsTr\la_003.nii.gz"
label_nii = nib.load(label_path)
label_array = label_nii.get_fdata()
if label_array.shape[2] < label_array.shape[0] and label_array.shape[2] < label_array.shape[1]:
    label_array = np.transpose(label_array, (2, 1, 0))
label_array = label_array.astype(np.uint8)
print(f"{label_array.shape=}") # label_array.shape=(D, 320, 320)
print(f"{np.unique(label_array)=}") # np.unique(label_array)=array([0, 1], dtype=uint8)