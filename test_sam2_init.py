#!/usr/bin/env python3
"""
Quick test script to verify SAM2 initialization works correctly.
This bypasses the full training pipeline to isolate and test just the SAM2 loading.
"""
import os
import sys
import torch

# Add project root to sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from sam2.build_sam import build_sam2_video_predictor

def test_sam2_init():
    """Test SAM2 initialization with corrected config path."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing SAM2 initialization on device: {device}")
    
    # Use the corrected config path (relative to sam2 package)
    config_file = "configs/sam2/sam2_hiera_l.yaml"
    ckpt_path = "./models/sam2_modified/sam2-main/checkpoints/sam2_hiera_large.pt"
    
    print(f"Config file: {config_file}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Checkpoint exists: {os.path.exists(ckpt_path)}")
    
    try:
        print("\n[Step 1] Loading SAM2 Video Predictor...")
        sam_tracker = build_sam2_video_predictor(
            config_file=config_file,
            ckpt_path=ckpt_path,
            device=device
        )
        print("✓ SAM2 model loaded successfully!")
        
        print(f"\n[Step 2] Model info:")
        print(f"  - Type: {type(sam_tracker)}")
        print(f"  - Device: {next(sam_tracker.parameters()).device}")
        print(f"  - Trainable params: {sum(p.numel() for p in sam_tracker.parameters() if p.requires_grad)}")
        print(f"  - Total params: {sum(p.numel() for p in sam_tracker.parameters())}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Failed to load SAM2 model:")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sam2_init()
    sys.exit(0 if success else 1)
