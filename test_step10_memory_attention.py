import sys
import os
import torch
import torch.nn as nn

# Ensure the project root and sam2-main are in sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
sam2_main_path = os.path.join(project_root, "models", "sam2_modified", "sam2-main")
sys.path.append(project_root)
sys.path.append(sam2_main_path)

from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from sam2.modeling.sam.transformer import RoPEAttention

def test_step10_memory_attention():
    print("Testing Step 10: LLM-Guided Memory Attention")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # SAM 2 Memory Attention typical dimensions
    B = 2
    d_model = 256
    H, W = 64, 64 # Spatial size of memory features
    N = H * W     # Sequence length (4096)
    
    # We simulate memory tokens from previous frames.
    # Suppose we have 2 memory frames: 2 * 4096 = 8192 tokens
    M = 2 * N

    print(f"\nInitializing MemoryAttention with use_llm_tracker=True...")
    
    # Create a dummy layer just to satisfy the original __init__ signature
    cross_attention = RoPEAttention(rope_theta=10000.0, feat_sizes=[H, W], embedding_dim=d_model, num_heads=1, downsample_rate=1, dropout=0.1)
    self_attention = RoPEAttention(rope_theta=10000.0, feat_sizes=[H, W], embedding_dim=d_model, num_heads=1, downsample_rate=1, dropout=0.1)
    layer = MemoryAttentionLayer(activation="relu", dim_feedforward=2048, dropout=0.1, pos_enc_at_attn=False, self_attention=self_attention, d_model=d_model, pos_enc_at_cross_attn_keys=True, pos_enc_at_cross_attn_queries=False, cross_attention=cross_attention)
    
    try:
        mem_attn = MemoryAttention(
            d_model=d_model, 
            pos_enc_at_input=True, 
            layer=layer, 
            num_layers=1, # Original used multiple layers, but our LLM bypasses them
            batch_first=True,
            use_llm_tracker=True
        ).to(device)
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # Check if injected components are present
    if not hasattr(mem_attn, 'llm_tracker'):
        print("ERROR: llm_tracker not injected into MemoryAttention.")
        return
    else:
        print("SUCCESS: llm_tracker is successfully injected into MemoryAttention.")
        
    if not hasattr(mem_attn, 'local_align_conv'):
        print("ERROR: local_align_conv (DCNv4) not injected into MemoryAttention.")
        return
    else:
        print("SUCCESS: local_align_conv is successfully injected into MemoryAttention.")

    # Prepare dummy inputs
    # In batch_first=True mode, shapes are [B, SeqLen, C]
    curr = torch.randn(B, N, d_model).to(device)
    memory = torch.randn(B, M, d_model).to(device)
    
    curr_pos = torch.randn(B, N, d_model).to(device)
    # memory_pos = torch.randn(B, M, d_model).to(device) # Not strictly required for our custom forward path yet

    print(f"\nInput 'curr' shape (Current Frame): {curr.shape}")
    print(f"Input 'memory' shape (Previous Frames): {memory.shape}")
    print(f"Concatenated sequence length going to LLM: {M + N} tokens")

    # Forward pass
    print("\nRunning forward pass through Local-Global Collaborative Spatiotemporal Tracker...")
    try:
        output = mem_attn(
            curr=curr,
            memory=memory,
            curr_pos=curr_pos,
            memory_pos=None
        )
        
        print("\nForward pass successful!")
        print(f"Output shape: {output.shape}")
        
        # Verify the output shapes
        assert output.shape == (B, N, d_model), f"Output feature shape mismatch! Expected {(B, N, d_model)}, got {output.shape}"
        print("Shape verification passed. The 'MemoryAttention' rewrite is correct.")
        
    except Exception as e:
        print(f"Error during forward pass: {e}")

if __name__ == "__main__":
    test_step10_memory_attention()
