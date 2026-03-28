import sys
import os
import torch
import numpy as np

# Ensure the project root and sam2-main are in sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
sam2_main_path = os.path.join(project_root, "models", "sam2_modified", "sam2-main")
sys.path.append(project_root)
sys.path.append(sam2_main_path)

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.modeling.sam2_base import SAM2Base

# Create a mock/dummy SAM2 predictor since we just want to test the wrapper logic
# without loading the full 5GB model weights
class DummySAM2VideoPredictor(SAM2VideoPredictor):
    def __init__(self):
        # Skip the heavy initialization
        pass
        
    def propagate_in_video(self, inference_state, start_frame_idx=None, reverse=False, **kwargs):
        num_frames = inference_state["num_frames"]
        
        if reverse:
            end_frame_idx = 0
            processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
        else:
            end_frame_idx = num_frames - 1
            processing_order = range(start_frame_idx, end_frame_idx + 1)
            
        for frame_idx in processing_order:
            # Simulate returning a mask for this frame
            dummy_mask = torch.ones((1, 1, 256, 256)) * frame_idx
            yield frame_idx, [1], dummy_mask

def test_step11_bidirectional_propagation():
    print("Testing Step 11: Bidirectional Propagation Wrapper")
    
    # 1. Initialize predictor
    predictor = DummySAM2VideoPredictor()
    
    # 2. Mock an inference state for a 10-slice 3D volume
    num_frames = 10
    inference_state = {
        "num_frames": num_frames,
    }
    
    # 3. Assume the RL Agent gave a Bbox on the 5th slice (index 4)
    start_frame_idx = 4
    print(f"\nTotal slices (Z-axis): {num_frames}")
    print(f"RL Agent Prompt given at slice: {start_frame_idx}")
    
    # 4. Run Bidirectional Propagation
    all_masks = predictor.bidirectional_propagation(inference_state, start_frame_idx)
    
    print("\n--- Propagation Results ---")
    assert len(all_masks) == num_frames, "Output mask list length should equal num_frames"
    
    missing_frames = []
    for i, mask in enumerate(all_masks):
        if mask is None:
            missing_frames.append(i)
        else:
            print(f"Slice {i}: Mask generated (Mock value = {mask[0, 0, 0, 0].item()})")
            
    if len(missing_frames) == 0:
        print("\nSUCCESS: All slices have been successfully segmented via bidirectional tracking!")
    else:
        print(f"\nERROR: Missing masks for slices: {missing_frames}")

if __name__ == "__main__":
    test_step11_bidirectional_propagation()
