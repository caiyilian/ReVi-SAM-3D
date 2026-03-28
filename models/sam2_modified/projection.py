import torch
import torch.nn as nn

# 尝试导入 DCNv4，由于在本地可能没有编译环境，这里做个 fallback，方便代码查看

from DCNv4 import DCNv4
HAS_DCNV4 = True


class DeformableProjectionModule(nn.Module):
    """
    跨维度注意力引导的形变投影模块 (Cross-Dimensional Deformable Projection Module)
    创新点 1 的核心：将 1D 文本语义特征通过 Cross-Attention 和 DCNv4 注入到 2D 视觉特征中。
    """
    def __init__(self, visual_dim, text_dim, num_heads=8):
        """
        :param visual_dim: 视觉特征的通道数 (C)，通常对应 SAM 2 某一层输出的维度
        :param text_dim: 文本特征的维度 (通常为 CLIP 的 512)
        :param num_heads: 多头注意力的头数
        """
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        
        # 1. 文本特征对齐层：将文本特征投影到视觉特征相同的维度
        self.text_proj = nn.Linear(text_dim, visual_dim)
        
        # 2. 跨模态多头注意力 (Cross-Attention)
        # Query: 视觉特征, Key & Value: 文本特征
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=visual_dim,
            num_heads=num_heads,
            batch_first=True # PyTorch 1.9+ 支持 batch_first=True，输入形状为 (B, Seq, Feature)
        )
        
        # 3. 形变对齐层 (DCNv4)
        # 注意：DCNv4 要求 C // group 必须是 16 的倍数。如果 visual_dim 是 256，group 可以设为 4 或 8 等。
        group = 4
        if visual_dim % (16 * group) != 0:
            # 自动调整 group，确保满足 CUDA 算子要求
            for g in [1, 2, 4, 8, 16]:
                if visual_dim % (16 * g) == 0:
                    group = g
                    break
        
        if HAS_DCNV4:
            self.deform_conv = DCNv4(
                channels=visual_dim,
                kernel_size=3,
                stride=1,
                pad=1,
                dilation=1,
                group=group,
                offset_scale=1.0
            )
        else:
            # Fallback 仅用于在没有编译 DCNv4 的机器上通过语法检查
            self.deform_conv = nn.Conv2d(visual_dim, visual_dim, kernel_size=3, padding=1)
            
        # 4. 融合与归一化
        self.norm1 = nn.LayerNorm(visual_dim)
        self.norm2 = nn.LayerNorm(visual_dim)
        # 使用 1x1 卷积融合 DCNv4 的输出
        self.fuse_conv = nn.Conv2d(visual_dim, visual_dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, visual_feat, text_feat):
        """
        :param visual_feat: 2D 视觉特征，形状为 (B, C, H, W)
        :param text_feat: 1D 文本特征序列，形状为 (B, N, C_text) -> (B, 29, 512)
        :return: 增强后的 2D 视觉特征，形状为 (B, C, H, W)
        """
        B, C, H, W = visual_feat.shape
        
        # 1. 维度对齐
        # 文本特征: (B, N, C_text) -> (B, N, C)
        text_feat_proj = self.text_proj(text_feat)
        
        # 视觉特征平铺: (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        visual_seq = visual_feat.view(B, C, -1).transpose(1, 2)
        
        # 2. 跨模态交互 (Cross-Attention)
        # Query = visual_seq, Key/Value = text_feat_proj
        # 输出 attn_out: (B, H*W, C)
        attn_out, _ = self.cross_attn(
            query=self.norm1(visual_seq), 
            key=text_feat_proj, 
            value=text_feat_proj
        )
        
        # 残差连接并恢复 2D 形状
        visual_seq = visual_seq + attn_out
        visual_seq = self.norm2(visual_seq)
        
        # (B, H*W, C) -> (B, C, H*W) -> (B, C, H, W)
        attn_feat_2d = visual_seq.transpose(1, 2).view(B, C, H, W)
        
        # 3. 形变特征对齐 (DCNv4)
        if HAS_DCNV4:
            # DCNv4 输入要求形状为 (N, L, C) 即 (B, H*W, C)
            # visual_seq 已经是 (B, H*W, C)
            dcn_out = self.deform_conv(visual_seq, shape=(H, W)) # 输出 (B, H*W, C)
            # 转换回 (B, C, H, W)
            dcn_out_2d = dcn_out.transpose(1, 2).view(B, C, H, W)
        else:
            # Fallback (B, C, H, W) -> Conv2d -> (B, C, H, W)
            dcn_out_2d = self.deform_conv(attn_feat_2d)
            
        # 4. 最终融合 (带有残差连接)
        fused_feat = self.fuse_conv(self.act(dcn_out_2d))
        out_feat = visual_feat + fused_feat
        
        return out_feat
