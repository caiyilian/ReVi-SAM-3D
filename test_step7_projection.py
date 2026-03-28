import sys
import os
import torch

# 将项目根目录加入 path，以便导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.sam2_modified.projection import DeformableProjectionModule

def test_projection_module():
    print("Initializing DeformableProjectionModule...")
    
    # 假设 SAM 2 某层的视觉特征维度为 256
    visual_dim = 256
    # CLIP 提取的文本特征维度为 512
    text_dim = 512
    
    # 初始化模块
    # 注意：如果在没有安装 DCNv4 的机器上运行，会 fallback 到普通 Conv2d 且打印警告
    # 在您的服务器上运行时，它会自动使用 DCNv4
    projection_module = DeformableProjectionModule(visual_dim=visual_dim, text_dim=text_dim)
    
    # 将模型移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    projection_module = projection_module.to(device)
    print(f"Module initialized and moved to {device}")

    # 1. 模拟视觉特征 (Batch_size=2, Channel=256, H=64, W=64)
    # 这通常是 SAM 2 Image Encoder 某一个 block 输出的特征图
    B, C, H, W = 2, visual_dim, 64, 64
    visual_feat = torch.randn(B, C, H, W).to(device)
    print(f"\nSimulated visual feature shape: {visual_feat.shape}")

    # 2. 模拟文本特征 (Batch_size=2, Seq_len=29, Feature_dim=512)
    # 这正是我们刚才在 test_step6_text_encoder.py 中获得的输出形状
    N, C_text = 29, text_dim
    text_feat = torch.randn(B, N, C_text).to(device)
    print(f"Simulated text feature shape: {text_feat.shape}")

    print("\nForward pass through DeformableProjectionModule...")
    
    # 3. 前向传播
    output_feat = projection_module(visual_feat, text_feat)
    
    print("\nForward Pass Successful!")
    print("-" * 50)
    print(f"Output visual feature shape: {output_feat.shape}")
    print("-" * 50)
    
    # 验证输入输出形状是否完全一致 (残差连接的基本要求)
    assert visual_feat.shape == output_feat.shape, "Output shape must match input visual feature shape!"
    print("Shape verification passed. The module is ready to be injected into SAM 2.")

if __name__ == "__main__":
    test_projection_module()
