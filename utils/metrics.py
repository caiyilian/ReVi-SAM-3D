import torch

def compute_dice_score(pred_mask: torch.Tensor, gt_mask: torch.Tensor, smooth: float = 1e-5) -> float:
    """
    计算二值掩码的 Dice Similarity Coefficient (DSC)。
    这个函数设计为高度优化且在 GPU 上直接运行 (纯 PyTorch 实现)，
    非常适合作为强化学习环境中的实时 Reward 计算。
    
    Args:
        pred_mask (torch.Tensor): 预测的掩码，形状通常为 [B, 1, H, W] 或 [H, W]。
                                  包含概率值 (0~1) 或二值化结果 (0 或 1)。
        gt_mask (torch.Tensor): 真实的掩码 (Ground Truth)，形状与 pred_mask 相同。
                                必须是二值化的 (0 或 1)。
        smooth (float): 平滑系数，防止分母为 0 导致 NaN。默认 1e-5。
        
    Returns:
        float: 计算得到的 Dice Score，范围 [0.0, 1.0]。
               如果是 Batch，返回的是整个 Batch 的平均 Dice。
    """
    # 确保预测掩码在 0-1 之间 (如果传入的是 logits，需要先过 sigmoid)
    # 为了保险起见，我们这里进行阈值化，将概率转为二值，因为 RL 的 Reward 通常基于硬掩码
    if pred_mask.is_floating_point():
        pred_bin = (pred_mask > 0.5).float()
    else:
        pred_bin = pred_mask.float()
        
    gt_bin = gt_mask.float()
    
    # 展平为 1D 向量以便于计算交集和并集
    pred_flat = pred_bin.contiguous().view(-1)
    gt_flat = gt_bin.contiguous().view(-1)
    
    # 计算交集: |A ∩ B|
    intersection = (pred_flat * gt_flat).sum()
    
    # 计算并集相关的项: |A| + |B|
    union = pred_flat.sum() + gt_flat.sum()
    
    # Dice 公式: 2 * |A ∩ B| / (|A| + |B|)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice.item()

def compute_iou_score(pred_mask: torch.Tensor, gt_mask: torch.Tensor, smooth: float = 1e-5) -> float:
    """
    计算二值掩码的 Intersection over Union (IoU) 或 Jaccard Index。
    作为辅助评估指标。
    """
    if pred_mask.is_floating_point():
        pred_bin = (pred_mask > 0.5).float()
    else:
        pred_bin = pred_mask.float()
        
    gt_bin = gt_mask.float()
    
    pred_flat = pred_bin.contiguous().view(-1)
    gt_flat = gt_bin.contiguous().view(-1)
    
    intersection = (pred_flat * gt_flat).sum()
    union = pred_flat.sum() + gt_flat.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()
