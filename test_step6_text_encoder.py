import sys
import os
import torch

# 将项目根目录加入 path，以便导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.sam2_modified.text_encoder import TextPromptEncoder

def test_text_encoder():
    print("Initializing TextPromptEncoder...")
    # 初始化模型，它会自动下载 CLIP 的权重（如果本地没有的话）
    encoder = TextPromptEncoder()
    
    # 构造两个测试 Mock 文本 (模拟 batch_size = 2)
    mock_texts = [
        "A medical 3D scan (la_003.nii.gz) showing specific anatomical structures and potential lesions.",
        "A normal healthy heart scan without any visible tumor."
    ]
    
    print(f"\nInput texts:")
    for i, t in enumerate(mock_texts):
        print(f"[{i}]: {t}")
        
    print("\nExtracting features...")
    # 前向传播提取特征
    last_hidden_state, pooler_output = encoder(mock_texts)
    
    print("\nExtraction Successful!")
    print("-" * 30)
    print(f"last_hidden_state shape: {last_hidden_state.shape}") 
    # 期望输出类似: torch.Size([2, 21, 512]) (其中21是token长度，由padding决定，512是特征维度)
    
    print(f"pooler_output shape:     {pooler_output.shape}") 
    # 期望输出类似: torch.Size([2, 512])
    print("-" * 30)
    
    # 验证参数是否被冻结
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params} (Expected: 0)")

if __name__ == "__main__":
    test_text_encoder()
