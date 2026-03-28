import sys
import os
import torch

# Ensure the project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.metrics import compute_dice_score, compute_iou_score

def test_step14_metrics():
    print("Testing Step 14: Fast GPU-based Metrics for RL Reward")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dummy Ground Truth mask (e.g., a simple square in the middle)
    H, W = 128, 128
    gt_mask = torch.zeros((1, 1, H, W)).to(device)
    gt_mask[0, 0, 40:80, 40:80] = 1.0  # Ground truth box area = 40x40 = 1600

    # 1. Test Perfect Match
    pred_perfect = gt_mask.clone()
    dice_perfect = compute_dice_score(pred_perfect, gt_mask)
    iou_perfect = compute_iou_score(pred_perfect, gt_mask)
    print(f"\n[Test 1] Perfect Match -> Dice: {dice_perfect:.4f}, IoU: {iou_perfect:.4f}")
    assert abs(dice_perfect - 1.0) < 1e-4, "Perfect match should have Dice = 1.0"

    # 2. Test Empty Prediction (e.g., SAM failed to find anything)
    pred_empty = torch.zeros((1, 1, H, W)).to(device)
    dice_empty = compute_dice_score(pred_empty, gt_mask)
    print(f"[Test 2] Empty Prediction -> Dice: {dice_empty:.4f}")
    assert dice_empty < 1e-4, "Empty prediction should have Dice ~ 0.0"

    # 3. Test Partial Match (Shifted box)
    pred_partial = torch.zeros((1, 1, H, W)).to(device)
    # Shifted down and right by 20 pixels, intersection area is 20x20 = 400
    # Union area is 1600 + 1600 - 400 = 2800
    # Expected Dice = 2 * 400 / (1600 + 1600) = 800 / 3200 = 0.25
    pred_partial[0, 0, 60:100, 60:100] = 1.0 
    
    dice_partial = compute_dice_score(pred_partial, gt_mask)
    print(f"[Test 3] Partial Match -> Dice: {dice_partial:.4f}")
    assert abs(dice_partial - 0.25) < 1e-3, f"Expected 0.25, got {dice_partial}"

    # 4. Test probabilities (logits from SAM before hard thresholding)
    # Simulating SAM outputting 0.9 inside the box and 0.1 outside
    pred_probs = torch.ones((1, 1, H, W)).to(device) * 0.1
    pred_probs[0, 0, 40:80, 40:80] = 0.9
    dice_probs = compute_dice_score(pred_probs, gt_mask)
    print(f"[Test 4] Probabilistic Input -> Dice: {dice_probs:.4f}")
    assert abs(dice_probs - 1.0) < 1e-4, "Thresholded probabilities should yield Dice = 1.0"

    print("\nSUCCESS: Step 14 Metrics module passed all verification checks!")
    print("These functions are now ready to be used as real-time Reward signals for the RL Agent.")

if __name__ == "__main__":
    test_step14_metrics()
