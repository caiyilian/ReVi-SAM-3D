import numpy as np
import gym
from gym import spaces
import torch
from utils.metrics import compute_dice_score

class SAMClosedLoopEnv(gym.Env):
    """
    RL与SAM交互的闭环训练环境。
    将 SAM 2 视为环境的一部分，Agent 通过调整 Bbox (9个离散动作) 来获得更好的分割 Mask，
    环境通过计算新 Mask 与 Ground Truth 的 Dice 提升值作为 Reward 返回。
    
    核心改动：改用 SAM2VideoPredictor.propagate_in_video() 激活 MemoryAttention 的LLM时空追踪
    """
    def __init__(self, sam_model, image_tensor, gt_mask, init_bbox, init_slice_idx=None, 
                 fused_features=None, image_size=(256, 256), step_size=4, max_steps=10):
        super(SAMClosedLoopEnv, self).__init__()
        
        self.sam = sam_model
        # 获取SAM所在的设备
        self.device = next(sam_model.parameters()).device
        
        # 存储原始图像张量（用于propagate_in_video())
        self.image_tensor = image_tensor.to(self.device)  # [D, C, H, W]
        
        # 可选：预融合特征（用于缓存以避免重新提取）
        self.fused_features = fused_features.to(self.device) if fused_features is not None else None

        self.gt_mask = gt_mask.to(self.device)  # [D, 1, H_orig, W_orig] 或 [D, H_orig, W_orig]
        # 标准化gt_mask维度（确保是[D, H, W]）
        if self.gt_mask.dim() == 4:
            self.gt_mask = self.gt_mask.squeeze(1) if self.gt_mask.shape[1] == 1 else self.gt_mask
        
        self.init_bbox = np.array(init_bbox.detach().cpu(), dtype=np.float32)
        # 防御性处理：确保 init_bbox 是 (4,) 而不是 (1, 4)
        if self.init_bbox.ndim == 2 and self.init_bbox.shape[0] == 1:
            self.init_bbox = self.init_bbox.squeeze(0)
        
        # 记录初始切片位置
        D = self.image_tensor.shape[0]
        self.init_slice_idx = init_slice_idx if init_slice_idx is not None else D // 2
        
        # 动态计算输出mask的目标分辨率（基于gt_mask的原始分辨率）
        self.gt_height, self.gt_width = self.gt_mask.shape[1], self.gt_mask.shape[2]
        self.image_size = image_size
        self.step_size = step_size
        self.max_steps = max_steps
        
        # 初始化推理状态缓存（SAM2VideoPredictor 需要）
        self.inference_state = None
        
        # 动作空间：9个离散动作
        self.action_space = spaces.Discrete(9)
        
        # 观测空间
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]), 
            high=np.array([image_size[1], image_size[0], image_size[1], image_size[0]]), 
            dtype=np.float32
        )
        
        self.current_bbox = None
        self.current_dsc = 0.0
        self.step_count = 0
        
    def reset(self):
        """
        重置环境状态（每个 Episode 开始时调用）。
        使用 propagate_in_video() 生成初始3D mask，激活 MemoryAttention 的LLM时空追踪。
        """
        self.current_bbox = self.init_bbox.copy()
        self.step_count = 0
        # 初始预测和 Dice 计算
        # 注意：现在处理3D数据，返回的mask是 [D, H, W]
        with torch.no_grad():
            # 使用真实的SAM2 API生成初始mask（3D双向传播，激活MemoryAttention）
            init_mask = self._predict_mask_with_full_api(
                self.current_bbox,     # [4] - bbox提示
                self.init_slice_idx   # 初始切片位置
            )
            
            self.current_dsc = compute_dice_score(init_mask, self.gt_mask)
            
        return self.current_bbox.copy()
        
    def step(self, action):
        """
        执行动作，更新状态，计算奖励。
        每次step都重新初始化推理状态，确保无残留。
        """
        self.step_count += 1
        
        # 1. 根据动作更新 Bbox
        new_bbox = self._update_bbox(self.current_bbox, action)
        
        done = False
        reward = 0.0
        
        # 动作 8 是终止动作
        if action == 8:
            done = True
        else:
            # 2. 闭环预测：用新 Bbox 使用真实SAM2 API生成新 Mask（3D）
            with torch.no_grad():
                # 每次step都重新初始化推理状态
                self.inference_state = None
                
                new_mask = self._predict_mask_with_full_api(
                    new_bbox,             # [4]
                    self.init_slice_idx  # 初始切片位置
                )
                new_dsc = compute_dice_score(new_mask, self.gt_mask)
            
            # 3. 奖励计算逻辑
            # 奖励 = (新 Dice - 旧 Dice) * 100 - 步数惩罚
            # 如果 Dice 提升了，给出正向奖励；否则给予负向奖励。
            # 0.1 是 step penalty，鼓励尽快找到最优 Bbox。
            reward = (new_dsc - self.current_dsc) * 100.0 - 0.1
            
            # 更新内部状态
            self.current_bbox = new_bbox
            self.current_dsc = new_dsc
            
        # 达到最大步数也强行终止
        if self.step_count >= self.max_steps:
            done = True
            
        info = {
            "current_dsc": self.current_dsc,
            "step_count": self.step_count
        }
        
        return self.current_bbox.copy(), float(reward), done, info
        
    def _update_bbox(self, bbox, action):
        """
        辅助函数：根据 9 个离散动作更新边界框
        bbox 格式: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox.copy()
        delta = self.step_size
        
        if action == 0:   y1 -= delta # 上边外扩 (y1 减小)
        elif action == 1: y2 += delta # 下边外扩 (y2 增大)
        elif action == 2: x1 -= delta # 左边外扩 (x1 减小)
        elif action == 3: x2 += delta # 右边外扩 (x2 增大)
        elif action == 4: y1 += delta # 上边内缩 (y1 增大)
        elif action == 5: y2 -= delta # 下边内缩 (y2 减小)
        elif action == 6: x1 += delta # 左边内缩 (x1 增大)
        elif action == 7: x2 -= delta # 右边内缩 (x2 减小)
        # action == 8 (终止) 已经在外部处理
        
        # 边界裁剪，防止越界
        H, W = self.image_size
        x1 = np.clip(x1, 0, W - 1)
        y1 = np.clip(y1, 0, H - 1)
        
        # 确保 x2 > x1 和 y2 > y1，防止内缩导致 Bbox 反转
        x2 = np.clip(x2, x1 + 1, W) 
        y2 = np.clip(y2, y1 + 1, H) 
        
        return np.array([x1, y1, x2, y2], dtype=np.float32)
    
    def _init_inference_state(self):
        """
        初始化 SAM2VideoPredictor 的推理状态字典。
        处理融合特征的情况：将融合特征作为缓存，避免重新提取。
        """
        D, C, H, W = self.image_tensor.shape
        device = self.device
        
        # 构建推理状态字典（模拟 SAM2VideoPredictor.init_state()）
        inference_state = {
            "images": self.image_tensor,  # [D, C, H, W] - 可以是融合特征或原始图像
            "num_frames": D,
            "device": device,
            "storage_device": device,
            "cached_features": {},  # 推荐：预先填充融合特征以避免重新提取
            "output_dict_per_obj": {},  # 存储每个对象的输出
            "point_inputs_per_obj": {},  # 存储每个对象的点提示
            "box_inputs_per_obj": {},    # 存储每个对象的bbox提示
            "mask_inputs_per_obj": {},   # 存储每个对象的mask提示
            "consolidated_pred_masks": {},  # 最后的预测masks
            "frames_tracked_per_obj": {},  # 跟踪信息
            "obj_ids": None,  # 将在添加点/框时设置
        }
        
        return inference_state
    
    def _predict_mask_with_full_api(self, bbox, init_slice_idx=0):
        """
        使用 SAM2VideoPredictor.propagate_in_video() 生成分割mask。
        
        核心设计：利用 propagate_in_video() 的自动时空特征提取和状态管理，
        激活 MemoryAttention 的LLM时空追踪器和DCNv4局部对齐模块。
        
        调用链：
        propagate_in_video() 
          → _run_single_frame_inference()
            → track_step()
              → _prepare_memory_conditioned_features()
                → self.memory_attention()  ✓ 激活LLM时空追踪！
        
        Args:
            bbox: [4] - bbox坐标 [x1, y1, x2, y2]
            init_slice_idx: int - 初始切片位置
            
        Returns:
            masks_3d: [D, H_orig, W_orig] - 整个3D体积的分割结果
        """
        try:
            with torch.no_grad():
                D, C, H_orig, W_orig = self.image_tensor.shape
                
                # Step 1: 初始化推理状态（如果还未初始化）
                if self.inference_state is None:
                    self.inference_state = self._init_inference_state()
                    # 初始化必要的output_dict_per_obj结构
                    self.inference_state["obj_ids"] = []
                    self.inference_state["frames_tracked_per_obj"] = {}
                else:
                    # 清空之前的对象输出
                    self.inference_state["output_dict_per_obj"] = {}
                    self.inference_state["consolidated_pred_masks"] = {}
                
                # Step 2: 标准化bbox到[0, 1]
                bbox_norm = np.array(bbox, dtype=np.float32)
                if bbox_norm.ndim == 2 and bbox_norm.shape[0] == 1:
                    bbox_norm = bbox_norm.squeeze(0)
                
                bbox_norm[0] = bbox_norm[0] / self.gt_width
                bbox_norm[1] = bbox_norm[1] / self.gt_height
                bbox_norm[2] = bbox_norm[2] / self.gt_width
                bbox_norm[3] = bbox_norm[3] / self.gt_height
                
                # 确保bbox有效
                bbox_norm = np.clip(bbox_norm, 0, 1)
                
                # Step 3: 在初始切片添加object detection提示
                # object_id=1 用于跟踪第一个目标
                self.sam.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=init_slice_idx,
                    obj_id=1,
                    box=bbox_norm  # [x1_norm, y1_norm, x2_norm, y2_norm]
                )
                
                # Step 4: 正向传播（从init_slice_idx到D-1）
                # propagate_in_video 返回 (frame_idx, obj_ids, video_res_masks)
                # video_res_masks: [num_objects, num_frames, 1, H_orig, W_orig] 张量（已自动缩放）
                masks_3d = torch.zeros(D, self.gt_height, self.gt_width, 
                                      device=self.device, dtype=torch.float32)
                
                # 正向传播
                # max_frame_num_to_track=D-init_slice_idx 表示从init_slice_idx往后传播D-init_slice_idx帧
                frame_idx, obj_ids, video_res_masks = self.sam.propagate_in_video(
                    inference_state=self.inference_state,
                    start_frame_idx=init_slice_idx,
                    max_frame_num_to_track=D - init_slice_idx,
                    reverse=False  # 正向传播
                )
                
                # 提取masks并填充到masks_3d
                # video_res_masks shape: [num_objects, num_frames, 1, H_orig, W_orig]
                # 需要找到obj_id=1在obj_ids中的索引
                if isinstance(obj_ids, list):
                    obj_idx = obj_ids.index(1) if 1 in obj_ids else 0
                else:
                    matches = (obj_ids == 1).nonzero(as_tuple=True)[0]
                    obj_idx = matches[0].item() if len(matches) > 0 else 0
                
                # 提取该对象的masks并应用sigmoid激活
                # video_res_masks可能的形状：[num_objects, num_frames, 1, H, W]
                if video_res_masks.dim() == 5:
                    obj_masks = video_res_masks[obj_idx, :, 0, :, :]  # [num_frames, H, W]
                else:
                    obj_masks = video_res_masks[obj_idx]  # 适应其他可能的形状
                
                obj_masks = torch.sigmoid(obj_masks.float())
                
                # 填充到masks_3d
                for i, f_idx in enumerate(range(init_slice_idx, min(init_slice_idx + obj_masks.shape[0], D))):
                    if f_idx < D:
                        masks_3d[f_idx] = obj_masks[i]
                
                # Step 5: 反向传播（从init_slice_idx-1到0）
                if init_slice_idx > 0:
                    # 重新初始化推理状态并添加相同提示用于反向传播
                    rev_inference_state = self._init_inference_state()
                    rev_inference_state["obj_ids"] = []
                    rev_inference_state["frames_tracked_per_obj"] = {}
                    
                    self.sam.add_new_points_or_box(
                        inference_state=rev_inference_state,
                        frame_idx=init_slice_idx,
                        obj_id=1,
                        box=bbox_norm
                    )
                    
                    # 反向传播：从init_slice_idx往回传播init_slice_idx+1帧
                    frame_idx, obj_ids, video_res_masks = self.sam.propagate_in_video(
                        inference_state=rev_inference_state,
                        start_frame_idx=init_slice_idx,
                        max_frame_num_to_track=init_slice_idx + 1,
                        reverse=True  # 反向传播
                    )
                    
                    # 提取masks
                    if isinstance(obj_ids, list):
                        obj_idx = obj_ids.index(1) if 1 in obj_ids else 0
                    else:
                        matches = (obj_ids == 1).nonzero(as_tuple=True)[0]
                        obj_idx = matches[0].item() if len(matches) > 0 else 0
                    
                    if video_res_masks.dim() == 5:
                        obj_masks = video_res_masks[obj_idx, :, 0, :, :]  # [num_frames, H, W]
                    else:
                        obj_masks = video_res_masks[obj_idx]
                    
                    obj_masks = torch.sigmoid(obj_masks.float())
                    
                    # 填充到masks_3d（反向传播的帧顺序）
                    rev_frames = list(range(init_slice_idx - 1, -1, -1))
                    for i, f_idx in enumerate(rev_frames):
                        if i < obj_masks.shape[0]:
                            masks_3d[f_idx] = obj_masks[i]
                
                return masks_3d
                
        except Exception as e:
            print(f"错误 in _predict_mask_with_full_api: {str(e)}")
            import traceback
            traceback.print_exc()
            # 容错：返回零mask而不是中止
            return torch.zeros(D, self.gt_height, self.gt_width, 
                             device=self.device, dtype=torch.float32)
