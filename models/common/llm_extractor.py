import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

def generate_2d_sin_cos_positional_encoding(height, width):
    """
    生成 2D 正余弦位置编码，供 LLaMA/DeepSeek 的 Rotary Position Embedding (RoPE) 使用。
    参考自 LLM4Seg 的实现。
    """
    token = height * width
    # 2D coordinates and Calculate sin and cos encodings
    # Shape: (H*W, 1)
    sin_y = torch.sin(torch.arange(token).view(1, token, 1)).float()
    cos_x = torch.cos(torch.arange(token).view(1, token, 1)).float()
    
    # 动态获取设备，方便在 CPU/GPU 上运行
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return sin_y.to(device), cos_x.to(device)


class FrozenLLMLayerExtractor(nn.Module):
    """
    基于预训练大语言模型 (LLM) 中间层的特征提取器。
    用于创新点 2 & 3：提取全局语义和时空依赖。
    完全借鉴《LLM4Seg》的“三明治”结构：Adapter1 -> Frozen LLM Layer -> Adapter2
    """
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        model_name="/public/cyl/fourth_works/pretrained_weights/DeepSeek-R1-Distill-Qwen-1.5B", 
        layer_idx=27, 
        llm_hidden_dim=1536,
        h=32, 
        w=32,
        freeze=True
    ):
        """
        :param in_channels: 输入视觉特征的通道数 (例如 SAM 2 的 256)
        :param out_channels: 输出特征的通道数 (通常等于 in_channels)
        :param model_name: Hugging Face 上的 LLM 模型名称
        :param layer_idx: 要提取的 LLM Transformer Block 索引
        :param llm_hidden_dim: LLM 内部的隐藏层维度 (DeepSeek-1.5B 为 1536)
        :param h: 视觉特征图的高度
        :param w: 视觉特征图的宽度
        :param freeze: 是否冻结 LLM 层参数
        """
        super().__init__()
        self.h = h
        self.w = w
        self.layer_idx = layer_idx
        
        # 1. 前向投影层 (升维到 LLM 的维度)
        self.adapter1 = nn.Linear(in_channels, llm_hidden_dim)
        
        # 2. 获取并冻结 LLM 单层
        print(f"Loading {model_name}...")
        # 加载完整模型
        full_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # 提取指定层
        self.llm_layer = full_model.model.layers[layer_idx]
        print(f"Successfully extracted layer {layer_idx} from {model_name}.")
        
        # 冻结参数
        if freeze:
            for param in self.llm_layer.parameters():
                param.requires_grad = False
            print("LLM layer parameters frozen.")
        else:
            print("LLM layer parameters UNfrozen (Trainable).")
            
        # 释放完整模型以节省内存 (这一步很重要，特别是对于大模型)
        del full_model
        torch.cuda.empty_cache()
        

        self._is_dummy = False
            
        # 3. 生成位置编码
        # 注意：这里作为缓存注册，避免每次 forward 都重新生成
        sin, cos = generate_2d_sin_cos_positional_encoding(h, w)
        self.register_buffer('sin', sin)
        self.register_buffer('cos', cos)
        
        # 4. 后向投影层 (降维回视觉特征的维度)
        self.adapter2 = nn.Linear(llm_hidden_dim, out_channels)

    def forward(self, x):
        """
        :param x: 视觉特征，形状为 (B, C, H, W) 或已经是 (B, Seq, C) 的序列
        :return: 增强后的特征，形状与输入相同
        """
        is_2d = len(x.shape) == 4
        if is_2d:
            B, C, H, W = x.shape
            # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
            x_seq = x.flatten(2).transpose(1, 2)
        else:
            x_seq = x
            B, seq_len, C = x_seq.shape

        # 1. 升维投影
        # (B, Seq, C) -> (B, Seq, LLM_Dim)
        hidden_states = self.adapter1(x_seq)
        
        # 动态处理位置编码的长度匹配问题
        # 如果输入的序列长度大于预设的 h*w (比如拼了多帧)，我们需要扩展 sin/cos
        seq_len = hidden_states.shape[1]
        if seq_len != self.sin.shape[0]:
            # 为了简单起见，如果长度不匹配，我们动态生成对应长度的 1D RoPE
            # 这是一个稳妥的 fallback，确保时空追踪时长序列不会报错
            sin_1d = torch.sin(torch.arange(seq_len).view(1, seq_len, 1)).float().to(hidden_states.device)
            cos_1d = torch.cos(torch.arange(seq_len).view(1, seq_len, 1)).float().to(hidden_states.device)
            pos_emb = [sin_1d, cos_1d]
        else:
            pos_emb = [self.sin, self.cos]
        
        # 2. 通过 LLM 层
        if not self._is_dummy:
            # LLaMA/DeepSeek 层通常接受 position_embeddings 作为参数 (用于 RoPE)
            llm_outputs = self.llm_layer(
                hidden_states=hidden_states, 
                position_embeddings=pos_emb
            )
            # llm_layer 返回的是一个 tuple，第一个元素是 hidden_states
            hidden_states = llm_outputs[0]
        else:
            # Dummy layer testing
            hidden_states = self.llm_layer(hidden_states)
            
        # 3. 降维投影
        # (B, Seq, LLM_Dim) -> (B, Seq, Out_C)
        out_seq = self.adapter2(hidden_states)
        
        if is_2d:
            # (B, H*W, Out_C) -> (B, Out_C, H*W) -> (B, Out_C, H, W)
            out = out_seq.transpose(1, 2).reshape(B, -1, self.h, self.w)
            return out
        
        return out_seq
