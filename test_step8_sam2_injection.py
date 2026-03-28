import sys
import os
import torch

# Ensure the project root and sam2-main are in sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
sam2_main_path = os.path.join(project_root, "models", "sam2_modified", "sam2-main")
sys.path.append(project_root)
sys.path.append(sam2_main_path)

from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from sam2.modeling.sam.transformer import RoPEAttention

def build_dummy_sam2_base():
    """
    Builds a minimal dummy SAM2Base model just for testing the image_encoder forward pass.
    We don't need the full pre-trained weights, just the architecture.
    """
    print("Building dummy SAM2 architecture...")
    
    # 1. Build Trunk (Hiera)
    trunk = Hiera(embed_dim=96, num_heads=1, stages=[1, 2, 7, 2], global_att_blocks=[5, 7, 9], window_pos_embed_bkg_spatial_size=[7, 7])
    
    # 2. Build Neck
    position_encoding = PositionEmbeddingSine(num_pos_feats=256, normalize=True, scale=None, temperature=10000)
    neck = FpnNeck(
        position_encoding=position_encoding,
        d_model=256,
        backbone_channel_list=[768, 384, 192, 96],
        fpn_top_down_levels=[2, 3],
        fpn_interp_model="nearest"
    )
    
    # 3. Build Image Encoder
    image_encoder = ImageEncoder(trunk=trunk, neck=neck, scalp=1)
    
    # 4. Build Memory Attention (Dummy)
    cross_attention = RoPEAttention(rope_theta=10000.0, feat_sizes=[64, 64], embedding_dim=256, num_heads=1, downsample_rate=1, dropout=0.1)
    self_attention = RoPEAttention(rope_theta=10000.0, feat_sizes=[64, 64], embedding_dim=256, num_heads=1, downsample_rate=1, dropout=0.1)
    layer = MemoryAttentionLayer(activation="relu", dim_feedforward=2048, dropout=0.1, pos_enc_at_attn=False, self_attention=self_attention, d_model=256, pos_enc_at_cross_attn_keys=True, pos_enc_at_cross_attn_queries=False, cross_attention=cross_attention)
    memory_attention = MemoryAttention(d_model=256, pos_enc_at_input=True, layer=layer, num_layers=1)
    
    # 5. Build Memory Encoder (Dummy) - SAM2Base accepts any nn.Module here, but we use a simple Sequential to pass init
    memory_encoder = torch.nn.Sequential(
        torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 256, kernel_size=3, padding=1)
    )
    
    # 6. Build SAM2 Base
    model = SAM2Base(
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
    )
    return model

def test_step8_sam2_injection():
    print("Testing Step 8: SAM2 Image Encoder Injection")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Build the model
    try:
        model = build_dummy_sam2_base().to(device)
        print("SAM2Base model instantiated successfully.")
    except Exception as e:
        print(f"Failed to build SAM2Base: {e}")
        return

    # Check if deformable_projection was successfully injected
    if not hasattr(model, 'deformable_projection'):
        print("ERROR: deformable_projection is NOT injected into SAM2Base.")
        return
    else:
        print("SUCCESS: deformable_projection is successfully injected into SAM2Base.")

    # 2. Prepare dummy inputs
    B = 2
    C_in, H_in, W_in = 3, 512, 512
    img_batch = torch.randn(B, C_in, H_in, W_in).to(device)
    print(f"\nInput image batch shape: {img_batch.shape}")

    # Simulated Mock Text features from Step 6 (e.g., LLaVA-Med mock)
    seq_len = 29
    text_dim = 512
    text_feat = torch.randn(B, seq_len, text_dim).to(device)
    print(f"Input text feature shape: {text_feat.shape}")

    # 3. Forward pass through modified forward_image
    print("\nRunning forward_image with text features...")
    try:
        # Note: We modified forward_image to accept text_feat
        backbone_out = model.forward_image(img_batch, text_feat=text_feat)
        
        # Verify the output shapes
        # The highest level FPN feature (lowest resolution) should be at index -1
        # SAM2 backbone has a stride of 16 for the lowest resolution features.
        # Its shape should be [B, 256, H_in//16, W_in//16]
        visual_feat = backbone_out["backbone_fpn"][-1]
        print("\nForward pass successful!")
        print(f"Output visual feature (lowest res) shape: {visual_feat.shape}")
        
        assert visual_feat.shape == (B, 256, H_in//16, W_in//16), f"Output feature shape mismatch! Expected {(B, 256, H_in//16, W_in//16)}, got {visual_feat.shape}"
        print("Shape verification passed. Integration is correct.")
        
    except Exception as e:
        print(f"Error during forward_image: {e}")

if __name__ == "__main__":
    test_step8_sam2_injection()
