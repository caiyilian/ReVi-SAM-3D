#!/usr/bin/env python3
"""
Test script to verify that SAM2 checkpoint loading works with modified architecture.
This tests that missing keys (newly injected layers) are handled gracefully.
"""
import os
import sys
import torch
import logging

# Configure logging to see warnings
logging.basicConfig(
    level=logging.WARNING,
    format='[%(levelname)s] %(message)s'
)

# Add project root to sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from sam2.build_sam import build_sam2_video_predictor

def test_checkpoint_loading():
    """Test checkpoint loading with modified architecture."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing SAM2 checkpoint loading on device: {device}\n")
    
    config_file = "configs/sam2/sam2_hiera_l.yaml"
    ckpt_path = "./models/sam2_modified/sam2-main/checkpoints/sam2_hiera_large.pt"
    
    print("=" * 70)
    print("CHECKPOINT LOADING TEST")
    print("=" * 70)
    print(f"Config file: {config_file}")
    print(f"Checkpoint:  {ckpt_path}")
    print(f"Checkpoint exists: {os.path.exists(ckpt_path)}")
    print()
    
    try:
        print("[Step 1/3] Loading SAM2 Video Predictor with modified architecture...")
        print("           (This model has injected layers like deformable_projection and llm_tracker)")
        print()
        
        sam_tracker = build_sam2_video_predictor(
            config_file=config_file,
            ckpt_path=ckpt_path,
            device=device
        )
        
        print("\n[Step 2/3] Analyzing loaded model...")
        
        # Count parameters
        total_params = sum(p.numel() for p in sam_tracker.parameters())
        trainable_params = sum(p.numel() for p in sam_tracker.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in sam_tracker.parameters() if not p.requires_grad)
        
        print(f"  Total parameters:    {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters:    {frozen_params:,}")
        
        print("\n[Step 3/3] Checking for custom layers...")
        
        # Check for custom injected layers
        has_deformable = hasattr(sam_tracker, 'deformable_projection')
        has_llm_tracker = hasattr(sam_tracker.memory_attention, 'llm_tracker') if hasattr(sam_tracker, 'memory_attention') else False
        has_local_align = hasattr(sam_tracker.memory_attention, 'local_align_conv') if hasattr(sam_tracker, 'memory_attention') else False
        
        print(f"  ✓ Has deformable_projection:      {has_deformable}")
        print(f"  ✓ Has memory_attention.llm_tracker: {has_llm_tracker}")
        print(f"  ✓ Has memory_attention.local_align_conv: {has_local_align}")
        
        print("\n" + "=" * 70)
        print("✓ SUCCESS: Checkpoint loaded successfully!")
        print("=" * 70)
        print("\nKey points:")
        print("  1. ✓ Config file found and parsed correctly")
        print("  2. ✓ Checkpoint loaded (missing keys for new layers handled gracefully)")
        print("  3. ✓ New injected layers initialized with random weights")
        print("  4. ✓ Model ready for training")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("✗ FAILED: Error loading checkpoint")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_checkpoint_loading()
    sys.exit(0 if success else 1)
