# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import nn, Tensor

from sam2.modeling.sam.transformer import RoPEAttention

from sam2.modeling.sam2_utils import get_activation_fn, get_clones

# --- INJECT Frozen LLM Layer & DCNv4 (Step 10) ---
import sys
import os
# Ensure the project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../../")) # e:\projects\大模型分割\方案\ReVi-SAM-3D
if project_root not in sys.path:
    sys.path.append(project_root)


from models.common.llm_extractor import FrozenLLMLayerExtractor
HAS_LLM_EXTRACTOR = True



from DCNv4 import DCNv4
HAS_DCNV4 = True

# --------------------------------------------------

class MemoryAttentionLayer(nn.Module):

    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: nn.Module,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)

        # Where to add pos enc
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:

        # Self-Attn, Cross-Attn
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class MemoryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,  # Do layers expect batch first input?
        use_llm_tracker: bool = True, # --- INJECT (Step 10) ---
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first
        
        # --- INJECT LLM Tracker & DCNv4 Local Branch (Step 10) ---
        self.use_llm_tracker = use_llm_tracker
        if self.use_llm_tracker and HAS_LLM_EXTRACTOR:
            # Global Branch: Frozen LLM Layer
            # We assume sequence length will be around 64x64 + text tokens = 4096+
            self.llm_tracker = FrozenLLMLayerExtractor(
                in_channels=d_model,
                out_channels=d_model,
                model_name="/public/cyl/fourth_works/pretrained_weights/DeepSeek-R1-Distill-Qwen-1.5B",
                layer_idx=27,
                llm_hidden_dim=1536,
                h=64, # SAM2 memory features are usually 64x64
                w=64,
                freeze=True
            ) # 实测有运行到这里
            
            # Local Branch: Deformable Conv to align memory to current frame
            # SAM2 features are d_model (256)
            group = 4
            if d_model % (16 * group) != 0:
                for g in [1, 2, 4, 8, 16]:
                    if d_model % (16 * g) == 0:
                        group = g
                        break
            
            if HAS_DCNV4:
                self.local_align_conv = DCNv4(
                    channels=d_model,
                    kernel_size=3,
                    stride=1,
                    pad=1,
                    dilation=1,
                    group=group,
                    offset_scale=1.0
                ) # 实测有运行到这里
            else:
                # Fallback to standard conv for testing
                self.local_align_conv = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
                exit()
                
            self.fuse_proj = nn.Linear(d_model * 2, d_model)
        
        # --------------------------------------------------

    def forward(
        self,
        curr: torch.Tensor,  # self-attention inputs
        memory: torch.Tensor,  # cross-attention inputs
        curr_pos: Optional[Tensor] = None,  # pos_enc for self-attention inputs
        memory_pos: Optional[Tensor] = None,  # pos_enc for cross-attention inputs
        num_obj_ptr_tokens: int = 0,  # number of object pointer *tokens*
    ):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = (
                curr[0],
                curr_pos[0],
            )
        # SAM 2 originally expects (Seq, Batch, Dim) if not batch_first
        # Or (Batch, Seq, Dim) if batch_first
        # The assertion below checks batch size match.
        batch_dim_idx = 0 if self.batch_first else 1
        assert (
            curr.shape[batch_dim_idx] == memory.shape[batch_dim_idx]
        ), f"Batch size must be the same for curr ({curr.shape}) and memory ({memory.shape})"

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        # --- INJECT Local-Global Collaborative Spatiotemporal Tracking (Step 10) ---
        if self.use_llm_tracker and HAS_LLM_EXTRACTOR and hasattr(self, 'llm_tracker'):
            # Current inputs are likely (Seq, B, C) if not batch_first, or (B, Seq, C) if batch_first
            # Let's ensure batch first for our custom logic
            if not self.batch_first:
                output = output.transpose(0, 1) # [B, N, C]
                memory = memory.transpose(0, 1) # [B, M, C]
            
            B, N, C = output.shape
            
            # 1. Local Branch: Deformable Alignment
            # Reshape sequences back to 2D for convolution
            # SAM2 memory attention usually operates on 64x64 grids
            H = W = int(N**0.5)
            if H * W == N:
                out_2d = output.transpose(1, 2).view(B, C, H, W)
                
                # If HAS_DCNV4, it expects (B, H*W, C). If fallback Conv2d, it expects (B, C, H, W)
                if HAS_DCNV4:
                    local_feat = self.local_align_conv(output, shape=(H, W)) # [B, N, C]
                else:
                    local_feat_2d = self.local_align_conv(out_2d) # [B, C, H, W]
                    local_feat = local_feat_2d.flatten(2).transpose(1, 2) # [B, N, C]
            else:
                # Fallback if not a perfect square (e.g. object pointer tokens included)
                local_feat = output 
                
            # 2. Global Branch: Frozen LLM Spatial-Temporal Matching
            # Concatenate memory and current frame tokens to let LLM find long-range dependencies
            # We don't have explicit text embedding here, but memory can act as prompt
            seq_tokens = torch.cat([memory, output], dim=1) # [B, M+N, C]
            
            # Note: llm_tracker expects (B, Seq, C) or (B, C, H, W). We pass (B, Seq, C).
            # The internal positional encoding of LLM extractor might only cover N tokens,
            # but we pass the whole sequence. In a rigorous implementation, we might need 1D RoPE here.
            # We use the generic sequence forward path of our extractor.
            llm_out_seq = self.llm_tracker(seq_tokens) # [B, M+N, C]
            
            # Extract the refined current frame tokens
            global_feat = llm_out_seq[:, -N:, :] # [B, N, C]
            
            # 3. Fusion
            fused = torch.cat([local_feat, global_feat], dim=-1) # [B, N, 2C]
            output = self.fuse_proj(fused) # [B, N, C]
            
            if not self.batch_first:
                # Convert back to seq first if needed
                output = output.transpose(0, 1)
                
            # Skip the original Transformer blocks since we replaced them
            normed_output = self.norm(output)
            return normed_output
        # --------------------------------------------------------------------------

        if self.batch_first:
            # Convert to batch first
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)

        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                **kwds,
            )
        normed_output = self.norm(output)

        if self.batch_first:
            # Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)

        return normed_output
