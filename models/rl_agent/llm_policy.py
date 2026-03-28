import torch
import torch.nn as nn
import sys
import os

# Ensure the project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)


from models.common.llm_extractor import FrozenLLMLayerExtractor
HAS_LLM_EXTRACTOR = True


class LLMDrivenPromptAgentDDQN(nn.Module):
    """
    基于大模型单层的策略网络 (LLM-Layer-Driven Policy Network)
    状态: SAM提取的图像特征 (Image Features) + 当前边界框坐标 (Bbox)
    输出: 9个离散动作的 Q 值 (Q-values)
    """
    def __init__(self, visual_dim=256, llm_dim=1536, action_size=9, h=32, w=32):
        super().__init__()
        self.visual_dim = visual_dim
        self.llm_dim = llm_dim
        self.action_size = action_size
        self.h = h
        self.w = w
        
        # 1. 提取冻结的 LLM 单层 (复用步骤 9 的 DeepSeek 提取器)
        if HAS_LLM_EXTRACTOR:
            self.llm_layer_extractor = FrozenLLMLayerExtractor(
                in_channels=llm_dim, # We will do the projection outside the extractor for fine-grained control
                out_channels=llm_dim,
                model_name="/public/cyl/fourth_works/pretrained_weights/DeepSeek-R1-Distill-Qwen-1.5B",
                layer_idx=27,
                llm_hidden_dim=llm_dim,
                h=h,
                w=w,
                freeze=True
            )
        else:
            # Fallback for local testing without the extractor file
            self.llm_layer_extractor = nn.Linear(llm_dim, llm_dim)
            exit()
            
        # 2. 状态投影层 (将异构输入映射到 LLM 的高维空间)
        self.vis_proj = nn.Linear(visual_dim, llm_dim)
        # Bbox format is [x1, y1, x2, y2], we project it to llm_dim
        self.bbox_proj = nn.Linear(4, llm_dim) 
        
        # 3. 轻量级 Q-value 预测头 (MLP)
        # 仅这部分和投影层是可训练的，LLM层完全冻结
        self.q_head = nn.Sequential(
            nn.Linear(llm_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, image_features, current_bbox):
        """
        :param image_features: SAM2 提取的视觉特征, 形状 [B, C, H, W] 或 [B, H*W, C]
        :param current_bbox: 当前的边界框坐标, 形状 [B, 4]
        :return: 动作的 Q 值, 形状 [B, action_size]
        """
        B = image_features.size(0)
        # 1. 处理视觉特征
        if len(image_features.shape) == 4:
            # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
            vis_seq = image_features.flatten(2).transpose(1, 2)
        else:
            vis_seq = image_features
            
        # [B, N, C] -> [B, N, llm_dim]
        vis_tokens = self.vis_proj(vis_seq)
        
        # 2. 处理 Bbox 特征
        # [B, 4] -> [B, llm_dim] -> [B, 1, llm_dim]
        bbox_token = self.bbox_proj(current_bbox).unsqueeze(1)
        
        # 3. 序列拼接
        # 将 Bbox 作为 Prompt Token 放在视觉序列的最前面
        # [B, 1+N, llm_dim]
        seq_tokens = torch.cat([bbox_token, vis_tokens], dim=1)
        
        # 4. 送入冻结的 LLM 层进行深层语义交互
        if HAS_LLM_EXTRACTOR and hasattr(self.llm_layer_extractor, 'llm_layer'):
            # Directly use the underlying LLM layer to bypass the Extractor's adapter 
            # since we already projected it here.
            # Get the position embeddings if available
            pos_emb = None
            if hasattr(self.llm_layer_extractor, 'sin'):
                seq_len = seq_tokens.shape[1]
                sin = self.llm_layer_extractor.sin
                cos = self.llm_layer_extractor.cos
                if seq_len != sin.shape[0]:
                    sin_1d = torch.sin(torch.arange(seq_len).view(1, seq_len, 1)).float().to(seq_tokens.device)
                    cos_1d = torch.cos(torch.arange(seq_len).view(1, seq_len, 1)).float().to(seq_tokens.device)
                    pos_emb = [sin_1d, cos_1d]
                else:
                    pos_emb = [sin, cos]
                    
            if not self.llm_layer_extractor._is_dummy:
                llm_out = self.llm_layer_extractor.llm_layer(
                    hidden_states=seq_tokens,
                    position_embeddings=pos_emb
                )[0]
            else:
                llm_out = self.llm_layer_extractor.llm_layer(seq_tokens)
        else:
            # Fallback
            llm_out = self.llm_layer_extractor(seq_tokens)
            
        # 5. 提取蕴含了视觉全局上下文的 Bbox Token
        # 因为 bbox_token 放在了序列的第 0 个位置
        contextual_bbox_feat = llm_out[:, 0, :] # [B, llm_dim]
        
        # 6. 预测 9 个离散动作的 Q 值
        q_values = self.q_head(contextual_bbox_feat) # [B, 9]
        
        return q_values
