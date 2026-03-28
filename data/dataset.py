import os
import json
import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
from PIL import Image

# 导入SAM2的标准预处理Transform
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../models/sam2_modified/sam2-main'))
from sam2.utils.transforms import SAM2Transforms

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

class Medical3DDataset(Dataset):
    """
    将 3D 的 .nii.gz 文件加载并沿 Z 轴切片为 2D 序列的数据集。
    返回的 Tensor 形状为:
    image: [D, C, H, W] -> 通常为 [D, 3, H, W] (RGB, SAM2标准格式)
    label: [D, 1, H, W]
    
    预处理流程：与SAM2原本训练时一致
    1. 医学图像 float → [0, 1]（normalize_intensity）
    2. [0, 1] → [0, 255] uint8
    3. uint8 → PIL Image
    4. PIL Image → SAM2Transforms（ToTensor [0,1] → Resize → ImageNet标准化）
    """
    def __init__(self, data_dir, split="imagesTr", label_dir="labelsTr", 
                 resolution=1024, convert_to_rgb=True, vlm_json_path=None):
        """
        :param data_dir: 根目录，例如 "ReVi-SAM-3D/data/Task02_Heart"
        :param split: 子文件夹名称，"imagesTr" 或 "imagesTs"
        :param label_dir: 标签文件夹名称
        :param resolution: SAM2输入分辨率（避免硬编码），例如512/1024。默认1024
        :param convert_to_rgb: 是否将单通道医学图像转为RGB（SAM2需要RGB输入）
        :param vlm_json_path: 离线生成的真实 VLM 文本描述 JSON 文件路径
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.label_dir = label_dir
        self.resolution = resolution  # SAM2目标分辨率（动态配置，避免硬编码）
        self.convert_to_rgb = convert_to_rgb
        
        # 初始化SAM2标准预处理Transform（包含Resize和ImageNet标准化）
        self.sam_transforms = SAM2Transforms(
            resolution=self.resolution,
            mask_threshold=0.0  # 掩码阈值（用于后处理，这里不用）
        )
        
        # 兼容读取真实的 VLM 文本描述
        self.vlm_texts = {}
        self.use_real_vlm = False
        if vlm_json_path is not None and os.path.exists(vlm_json_path):
            with open(vlm_json_path, 'r', encoding='utf-8') as f:
                self.vlm_texts = json.load(f)
            self.use_real_vlm = True
            print(f"Loaded real VLM texts from {vlm_json_path}")
        else:
            print("No real VLM texts found or path not provided, will use Mock VLM texts.")
        
        # 获取所有图像路径
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, split, "*.nii.gz")))
        
        # 验证是否有标签 (测试集可能没有标签)
        self.has_labels = False
        if len(self.image_paths) > 0:
            sample_name = os.path.basename(self.image_paths[0])
            potential_label_path = os.path.join(data_dir, label_dir, sample_name)
            if os.path.exists(potential_label_path):
                self.has_labels = True

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        filename = os.path.basename(img_path)
        
        # 1. 读取 Image
        import nibabel as nib
        img_nii = nib.load(img_path)
        img_array = img_nii.get_fdata()
        
        # 确保通道顺序为 (D, H, W)，nibabel 默认可能是 (W, H, D)
        # 这取决于具体的 nii 文件，通常需要转置
        if img_array.shape[2] < img_array.shape[0] and img_array.shape[2] < img_array.shape[1]:
            img_array = np.transpose(img_array, (2, 1, 0)) # 将 (W, H, D) 转为 (D, H, W)
        
        # 归一化到[0, 1]（适配SAM2的ToTensor预期）
        img_array = normalize_intensity(img_array)
        img_array = img_array.astype(np.float32)
        
        # 2. 读取 Label
        if self.has_labels:
            label_path = os.path.join(self.data_dir, self.label_dir, filename)
            label_nii = nib.load(label_path)
            label_array = label_nii.get_fdata()
            if label_array.shape[2] < label_array.shape[0] and label_array.shape[2] < label_array.shape[1]:
                label_array = np.transpose(label_array, (2, 1, 0))
            label_array = label_array.astype(np.uint8)
        else:
            label_array = np.zeros_like(img_array, dtype=np.uint8)
        
        # 3. 使用SAM2标准预处理Transform
        # SAM2Transforms期望输入是 PIL Image 或 numpy uint8 [0, 255]
        # 逐切片处理：转为uint8 → PIL Image → SAM2Transforms
        img_list = []
        D = img_array.shape[0]
        
        for d in range(D):
            # 取单个切片 [H, W]，float [0, 1]
            slice_2d = img_array[d]
            
            # 转为uint8 [0, 255]
            slice_uint8 = (slice_2d * 255).astype(np.uint8)
            
            # 转为RGB（复制灰度为3通道，使其符合SAM2Transforms期望的RGB格式）
            # SAM2的ImageNet标准化是为RGB设计的，需要[3, H, W]格式
            rgb_array = np.stack([slice_uint8] * 3, axis=-1)  # [H, W, 3]
            pil_image = Image.fromarray(rgb_array, mode='RGB')
            
            # 使用SAM2 Transforms（包含ToTensor、Resize到resolution、ImageNet标准化）
            # 输出为 [3, resolution, resolution]
            transformed = self.sam_transforms(pil_image)
            img_list.append(transformed)
        
        # 堆叠所有切片 [D, 3, resolution, resolution]
        img_tensor = torch.stack(img_list, dim=0)
        
        # 4. Label预处理（需要与image_tensor的空间尺寸对齐）
        # Label进行Resize到相同分辨率（使用nearest，保留二值性）
        label_tensor = torch.from_numpy(label_array).unsqueeze(1).float()  # [D, 1, H, W]
        label_tensor = torch.nn.functional.interpolate(
            label_tensor, 
            size=(self.resolution, self.resolution), 
            mode='nearest'
        ).byte()  # [D, 1, resolution, resolution]
        
        # 5. 获取 VLM 文本（优先真实，兼容 Mock）
        if self.use_real_vlm and filename in self.vlm_texts:
            vlm_text = self.vlm_texts[filename]
        else:
            vlm_text = get_mock_vlm_text(filename)
        
        return {
            "image": img_tensor,     # [D, 3, resolution, resolution] - SAM2标准格式
            "label": label_tensor,   # [D, 1, resolution, resolution]
            "filename": filename,
            "original_shape": img_array.shape, # (D, H, W)
            "vlm_text": vlm_text
        }

def get_mock_vlm_text(filename):
    """
    针对创新点 1 的 Mock 机制。
    在 LLaVA-Med 就绪前，返回一段占位的解剖学描述文本。
    """
    # 这里可以根据 filename 做一些简单的区分，现在统一返回固定文本
    return f"A medical 3D scan ({filename}) showing specific anatomical structures and potential lesions."

# 简单测试代码
if __name__ == "__main__":
    test_dir = r"E:\projects\大模型分割\方案\ReVi-SAM-3D\data\Task02_Heart"
    # 测试时，可以传入 vlm_json_path 进行真实验证。这里使用当前目录下的 vlm_texts_3d.json 作为测试
    test_json_path = os.path.join(os.path.dirname(__file__), "vlm_texts_3d.json")
    dataset = Medical3DDataset(data_dir=test_dir, split="imagesTr", label_dir="labelsTr", resolution=512, vlm_json_path=test_json_path)
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Loaded sample: {sample['filename']}")
        print(f"Image tensor shape: {sample['image'].shape}, dtype: {sample['image'].dtype}")
        print(f"Label tensor shape: {sample['label'].shape}, dtype: {sample['label'].dtype}")
        print(f"VLM text: {sample['vlm_text']}")
    else:
        print("Dataset is empty. Please check the path.")
