import sys
import os
import torch
import numpy as np

# Ensure the project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from env.sam_closed_loop import SAMClosedLoopEnv

# Mock SAM Model for testing the Environment
class MockSAM:
    def __init__(self, gt_mask, device):
        self.gt_mask = gt_mask
        self.device = device
        
    def predict(self, image_features, bbox):
        """
        Simulate SAM prediction.
        To make it react to the RL agent, we'll create a dummy mask that is just 
        the intersection of the provided bbox and the ground truth mask.
        If the bbox perfectly matches the GT bounding box, Dice will be 1.0.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # Create an empty mask
        pred_mask = torch.zeros_like(self.gt_mask).to(self.device)
        
        # Fill the bbox region with 1s
        pred_mask[0, 0, y1:y2, x1:x2] = 1.0
        
        # Simulate "segmentation" by intersecting with a slightly larger GT area
        # This gives the RL agent a reason to shrink/expand
        simulated_prediction = pred_mask * self.gt_mask
        
        return simulated_prediction

def test_step15_env():
    print("Testing Step 15: RL Closed-Loop Environment")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Create a Dummy Ground Truth Mask (e.g. object at [40:80, 40:80])
    H, W = 128, 128
    gt_mask = torch.zeros((1, 1, H, W)).to(device)
    gt_mask[0, 0, 40:80, 40:80] = 1.0 
    
    # 2. Instantiate Mock SAM and Dummy Features
    mock_sam = MockSAM(gt_mask, device)
    dummy_image_features = torch.randn((1, 256, 64, 64)).to(device) # Shape doesn't matter for mock
    
    # 3. Initial bad Bbox (too small, needs to expand)
    init_bbox = [50, 50, 70, 70] # GT is [40, 40, 80, 80]
    
    # 4. Initialize Environment
    env = SAMClosedLoopEnv(
        sam_model=mock_sam, 
        image_features=dummy_image_features, 
        gt_mask=gt_mask, 
        init_bbox=init_bbox,
        image_size=(H, W),
        step_size=10, # Large step size for quick testing
        max_steps=5
    )
    
    print("\n--- Episode Start ---")
    obs = env.reset()
    print(f"Initial Bbox: {obs}")
    print(f"Initial Dice: {env.current_dsc:.4f}")
    
    # Let's perform a sequence of actions to try and expand the bbox to match GT
    
    # Action 0: Expand Top (y1 -= 10 -> 40)
    print("\nTaking Action: 0 (Expand Top)")
    obs, reward, done, info = env.step(0)
    print(f"New Bbox: {obs}, Reward: {reward:.4f}, Done: {done}, Info: {info}")
    
    # Action 1: Expand Bottom (y2 += 10 -> 80)
    print("\nTaking Action: 1 (Expand Bottom)")
    obs, reward, done, info = env.step(1)
    print(f"New Bbox: {obs}, Reward: {reward:.4f}, Done: {done}, Info: {info}")
    
    # Action 2: Expand Left (x1 -= 10 -> 40)
    print("\nTaking Action: 2 (Expand Left)")
    obs, reward, done, info = env.step(2)
    print(f"New Bbox: {obs}, Reward: {reward:.4f}, Done: {done}, Info: {info}")
    
    # Action 3: Expand Right (x2 += 10 -> 80) -> Now it perfectly matches GT!
    print("\nTaking Action: 3 (Expand Right)")
    obs, reward, done, info = env.step(3)
    print(f"New Bbox: {obs}, Reward: {reward:.4f}, Done: {done}, Info: {info}")
    
    assert info['current_dsc'] == 1.0, "Dice should be 1.0 when bbox perfectly matches GT"
    print("\nAwesome! Agent successfully adjusted the Bbox to achieve perfect segmentation.")
    
    # Action 8: Terminate
    print("\nTaking Action: 8 (Terminate)")
    obs, reward, done, info = env.step(8)
    print(f"New Bbox: {obs}, Reward: {reward:.4f}, Done: {done}, Info: {info}")
    assert done == True, "Environment should be done after Action 8"
    
    print("\nSUCCESS: Step 15 Closed-Loop Environment passed all verification checks!")

if __name__ == "__main__":
    test_step15_env()