import warnings
warnings.filterwarnings("ignore")# 忽略警告
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import sys
import torch
import numpy as np
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import gc
# 确保项目根目录在 sys.path 中
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
sys.path.insert(0, "models/sam2_modified/sam2-main")
# 导入我们的自定义模块
from data.dataset import Medical3DDataset
from models.rl_agent.dqn_agent import DQNAgent
from models.rl_agent.llm_policy import LLMDrivenPromptAgentDDQN
from env.sam_closed_loop import SAMClosedLoopEnv
from tqdm import tqdm
# 导入真实的 SAM 2 模型及相关模块
from sam2.build_sam import build_sam2_video_predictor
from models.sam2_modified.text_encoder import TextPromptEncoder
from models.sam2_modified.projection import DeformableProjectionModule

def parse_args():
    parser = argparse.ArgumentParser(description="RL-SAM-VLM Training Pipeline")
    parser.add_argument("--data_dir", type=str, default="./data/Task02_Heart", help="Path to medical images")
    parser.add_argument("--sam2_ckpt", type=str, default="./models/sam2_modified/sam2-main/checkpoints/sam2_hiera_tiny.pt", help="Path to SAM 2 checkpoint")
    parser.add_argument("--sam2_config", type=str, default="configs/sam2/sam2_hiera_t.yaml", help="Path to SAM 2 config file")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (usually 1 for 3D volumes)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=20, help="Max RL steps per episode")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for DQN")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--epsilon_end", type=float, default=0.1, help="Final exploration rate")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Exploration decay rate")
    parser.add_argument("--memory_size", type=int, default=10000, help="Replay buffer size")
    
    # 环境与网络维度参数 (避免硬编码)
    # 根据 3D 分割的标准以及 SAM 的架构，我们将默认分辨率调回 256
    parser.add_argument("--image_size", type=int, default=256, help="Input image spatial size (H and W) for SAM Env")
    parser.add_argument("--visual_dim", type=int, default=256, help="Channel dimension of SAM visual features")
    # SAM 的 ViT 通常下采样 16 倍，所以如果输入 256，特征图尺寸是 16
    parser.add_argument("--feature_size", type=int, default=16, help="Spatial size of SAM visual features (H and W)")
    parser.add_argument("--llm_dim", type=int, default=1536, help="Hidden dimension of the LLM layer")
    
    return parser.parse_args()

def setup_training():
    """
    步骤 16：组装主训练循环的上半部
    负责解析参数、初始化设备、加载数据、实例化 Agent 和 准备环境工厂。
    """
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Training on device: {device}")

    # [Step 4] 自动化检查离线 VLM Text 是否存在
    vlm_json_path = os.path.join(args.data_dir, "..", "vlm_texts_3d.json")
    if not os.path.exists(vlm_json_path):
        print("\n" + "!"*60)
        print("[WARNING] 核心创新点 1 提醒:")
        print(f"未在 {vlm_json_path} 找到真实的 VLM 文本描述 JSON 文件！")
        print("当前将自动回退使用基础的 Mock 文本。")
        print("若要使用真实的医学多模态先验知识，请在开始训练前运行:")
        print(">>> python data/generate_vlm_texts_3d.py")
        print("!"*60 + "\n")
        exit()
    else:
        print(f"[INFO] 成功检测到真实 VLM 离线描述文件: {vlm_json_path}")
    
    # 1. 初始化 Dataset 和 DataLoader
    print("Loading real 3D Medical dataset...")
    dataset = Medical3DDataset(
        data_dir=args.data_dir, 
        split="imagesTr", 
        label_dir="labelsTr", 
        resolution=args.image_size,
        vlm_json_path=vlm_json_path
    ) 
    
    # 因为 3D 体素数据较大，batch_size 通常设为 1
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 2. 初始化 SAM 2 模型 (追踪器)
    print(f"Initializing SAM 2 Tracker from {args.sam2_ckpt}...")
    try:
        sam_tracker = build_sam2_video_predictor(
            config_file=args.sam2_config, 
            ckpt_path=args.sam2_ckpt, 
            device=device
        )
        
        # 冻结 SAM 2 的大部分权重，仅保留我们魔改注入的部分(如 LLM 层, DCNv4 等)可训练
        # 具体冻结逻辑可能需要在 build 函数内或此处通过遍历 parameters 实现
        for name, param in sam_tracker.named_parameters():
            # 示例：如果不包含 'deformable' 或 'llm_layer' 等关键字，则冻结
            if "deformable" not in name and "llm_layer" not in name:
                param.requires_grad = False
                
    except Exception as e:
        print(f"Failed to load real SAM 2 model: {e}. Make sure the weights and config are correct.")
        raise e

    # 2.5 实例化与冻结 TextPromptEncoder
    print("Initializing TextPromptEncoder...")
    try:
        # TextPromptEncoder 内部已实现了 parameters().requires_grad = False 和 eval() 模式
        text_encoder = TextPromptEncoder(device=device)
    except Exception as e:
        print(f"Failed to load TextPromptEncoder: {e}. Are transformers/CLIP installed?")
        raise e

    # 2.6 实例化 DeformableProjectionModule (跨模态特征注入模块)
    print("Initializing DeformableProjectionModule...")
    # 假设 SAM 2 输出特征通道数为 256，CLIP 文本维度为 512
    proj_module = DeformableProjectionModule(visual_dim=args.visual_dim, text_dim=512).to(device)

    # 3. 实例化 DQN Agent (包含了策略网络和目标网络的初始化)
    print("Instantiating DQN Agent...")
    agent = DQNAgent(
        visual_dim=args.visual_dim,
        llm_dim=args.llm_dim,
        action_size=9,
        gamma=args.gamma,
        lr=args.lr,
        buffer_capacity=args.memory_size
    )
    
    # 手动同步一些参数
    agent.epsilon = args.epsilon_start
    agent.epsilon_min = args.epsilon_end
    agent.epsilon_decay = args.epsilon_decay

    # 4. 初始化策略网络损失记录器与 TensorBoard
    loss_history = []
    
    # 动态生成带有时间戳的日志目录，避免多次运行互相覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("runs", f"ReVi_SAM_3D_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    
    # 创建 Checkpoint 保存目录
    ckpt_dir = os.path.join("checkpoints", f"ReVi_SAM_3D_{timestamp}")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {ckpt_dir}")

    print("Setup Complete. Ready for training loop.")
    
    return args, device, dataloader, sam_tracker, text_encoder, proj_module, agent, loss_history, writer, ckpt_dir

def train_loop(args, device, dataloader, sam_tracker, text_encoder, proj_module, agent, loss_history, writer, ckpt_dir):
    """
    步骤 17：编写闭环训练逻辑 (下半部)
    """
    print("\n" + "="*40)
    print("Starting RL-SAM-VLM Training Loop")
    print("="*40)
    
    total_steps = 0
    
    for epoch in range(args.epochs):
        epoch_reward = 0.0
        epoch_loss = 0.0
        
        # 遍历每个 3D 体素/切片 (Batch)
        for batch_idx, batch_data in enumerate(tqdm(dataloader)):
            
            # 1. 提取当前 3D 体素的特征和真值 (假设 dataset 返回 [B, D, C, H, W] 或 [B, D, H, W])
            # 对于 SAM，我们通常是在 2D 序列上进行操作
            image_tensor = batch_data["image"].to(device)  # [B, D, 3, H, W] 或 [B, D, 1, H, W]
            
            # print(f"已分配显存: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB, image_tensor.shape=", image_tensor.shape)
            gt_mask = batch_data["label"].to(device)       # [B, D, 1, H, W]
            vlm_text = batch_data["vlm_text"]              # List of strings
            # 降维处理：取出 Batch 中的第一个 3D 序列 (通常 Batch_Size=1)
            # 此时特征变为 [D, C, H, W]
            image_features = image_tensor[0]
            gt_sequence = gt_mask[0]
            
            # [Step 7] 文本特征提取
            # 使用冻结的 TextPromptEncoder 将文本列表转为高维语义特征
            # vlm_text 是一个列表，例如 ["This image contains..."]
            text_hidden_state, text_pooler_output = text_encoder(vlm_text)
            
            # [Step 7.3 & 7.4] 跨模态特征注入与特征更正
            # 1. 提取纯视觉多尺度特征 (SAM2返回特征金字塔支持高分辨率mask输出)
            with torch.no_grad():
                # 调整形状以适应 image_encoder，比如扩展为 3 通道 (如果原本是单通道)
                if image_features.shape[1] == 1:
                    img_input = image_features.repeat(1, 3, 1, 1)
                else:
                    img_input = image_features
                    
                # 提取SAM2的完整多尺度特征金字塔（创新点1需要精准的像素级边界）
                raw_visual_features = sam_tracker.image_encoder(img_input)
                
                # 智能解析多尺度特征（动态支持，避免硬编码）
                visual_feat = None
                high_res_feats = None
                if isinstance(raw_visual_features, dict):
                    # SAM2标准返回格式：包含backbone_fpn多尺度金字塔
                    if 'backbone_fpn' in raw_visual_features:
                        backbone_fpn = raw_visual_features['backbone_fpn']
                        # 最后一层是最低分辨率（用于RL特征空间）
                        visual_feat = backbone_fpn[-1]
                        # 其他层是高分辨率特征（用于高精度mask解码）
                        high_res_feats = backbone_fpn[:-1] if len(backbone_fpn) > 1 else None
                    else:
                        visual_feat = raw_visual_features.get('vision_features', raw_visual_features)
                elif isinstance(raw_visual_features, (tuple, list)):
                    visual_feat = raw_visual_features[-1]
                else:
                    visual_feat = raw_visual_features
                # 清理不需要的中间变量
                del raw_visual_features
                torch.cuda.empty_cache()
                gc.collect()
            
            
            # 2. 将文本特征批量扩展以匹配 3D 切片深度 (D)
            D = visual_feat.shape[0]
            text_feat_expanded = text_hidden_state.expand(D, -1, -1)
            
            # 3. 使用 DeformableProjectionModule 注入文本特征（低分辨率主干）
            fused_visual_features = proj_module(visual_feat, text_feat_expanded)
            
            # 同时在高分辨率层也进行文本特征注入（保留创新点1的像素级精准性）
            fused_high_res_feats = None
            if high_res_feats is not None:
                fused_high_res_feats = []
                for high_res_feat in high_res_feats:
                    # 每一尺度都进行文本特征融合，确保多尺度一致性
                    # fused_hrf = proj_module(high_res_feat, text_feat_expanded)
                    # 防止爆显存，所以分块处理
                    fused_hrf_chunks = []
                    chunk_size = 25
                    
                    for i in range(0, D, chunk_size):
                        chunk = high_res_feat[i:i+chunk_size]
                        text_chunk = text_feat_expanded[i:i+chunk_size]
                        fused_chunk = proj_module(chunk, text_chunk)
                        fused_hrf_chunks.append(fused_chunk)
                    fused_hrf = torch.cat(fused_hrf_chunks, dim=0)

                    fused_high_res_feats.append(fused_hrf)
            
            # 4. 特征更新：将RL环境所需的图像特征替换为融合后的特征
            # 低分辨率用于RL状态空间，高分辨率用于最终分割输出
            image_features = fused_visual_features
            
            # 动态寻找一个包含目标的切片作为初始帧 (Ground Truth 引导)
            # 在真实训练中，RL 需要一个起始切片来生成提示。我们寻找目标像素最多的切片
            target_pixels_per_slice = gt_sequence.sum(dim=(1, 2, 3))
            if target_pixels_per_slice.sum() == 0:
                # 这个 3D 体素完全没有目标，跳过
                continue
                
            # 找到目标最大的那层作为初始切片 (或者也可以随机选一层有目标的)
            init_slice_idx = torch.argmax(target_pixels_per_slice).item()
            init_slice_gt = gt_sequence[init_slice_idx, 0].cpu().numpy()
            
            # 根据该层的 GT 生成一个带噪声的初始 Bbox
            rows = np.any(init_slice_gt, axis=1)
            cols = np.any(init_slice_gt, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # 加入一些随机扰动作为初始状态，让 RL Agent 去学习微调
            noise_scale = 10
            init_bbox = np.array([
                max(0, cmin - np.random.randint(0, noise_scale)),
                max(0, rmin - np.random.randint(0, noise_scale)),
                min(args.image_size, cmax + np.random.randint(0, noise_scale)),
                min(args.image_size, rmax + np.random.randint(0, noise_scale))
            ], dtype=np.float32)
            
            # 将 numpy array 转为张量，保持 (4,) 的形状
            # 注意：不添加 unsqueeze(0)，bbox 应该是 (4,) 而不是 (1, 4)
            init_bbox_tensor = torch.tensor(init_bbox).to(device)
            
            # 2. 初始化闭环环境
            # SAM 2 环境现已支持3D处理，接收原始图像张量和初始切片位置
            # 环境内部会使用 propagate_in_video() 激活 MemoryAttention 的LLM时空追踪
            
            env = SAMClosedLoopEnv(
                sam_model=sam_tracker,
                image_tensor=image_tensor[0],      # [D, C, H, W] - 原始图像
                gt_mask=gt_sequence,               # [D, H, W] - 整个3D GT  
                init_bbox=init_bbox_tensor,        # [4] - 初始切片的bbox
                init_slice_idx=init_slice_idx,     # 初始切片位置（已标注目标最多的切片）
                fused_features=image_features,     # [D, C, H, W] - 预融合特征（可选缓存）
                image_size=(args.image_size, args.image_size), 
                step_size=4,
                max_steps=args.max_steps
            )
            
            # Reset 获取初始状态
            current_bbox = env.reset()
            # 提取初始切片的特征作为 RL 的初始状态输入
            state_img = image_features[init_slice_idx] # [C, H, W]
            state_bbox = torch.tensor(current_bbox, dtype=torch.float32)
            
            done = False
            episode_reward = 0.0
            # 3. 强化学习 Episode 内部循环 (试错)
            while not done:
                total_steps += 1
                
                # Agent 根据当前状态选择动作
                action = agent.select_action(state_img, state_bbox, is_training=True)
                
                # 环境执行动作，返回新状态和奖励
                next_bbox, reward, done, info = env.step(action)
                next_state_bbox = torch.tensor(next_bbox, dtype=torch.float32)
                
                # 累加 Reward
                episode_reward += float(reward)
                
                # 将经验存入 Replay Buffer
                # 注意：为了防止显存爆炸，一定要使用 .detach().cpu() 或者 clone()，确保不保存计算图
                agent.replay_buffer.push(
                    state_img.detach().cpu(), 
                    state_bbox.detach().cpu(), 
                    action, 
                    float(reward), 
                    state_img.detach().cpu(), # 图像特征在当前设定下不随动作改变
                    next_state_bbox.detach().cpu(), 
                    done
                )
                
                # 状态转移
                state_bbox = next_state_bbox
                
                # 执行网络更新
                loss = agent.learn()
                if loss is not None:
                    # loss 通常返回的是一个张量，如果不加 .item()，会一直累积计算图导致爆显存
                    epoch_loss += float(loss)
                    loss_history.append(float(loss))
                    
            epoch_reward += episode_reward
            
        
        # 每个 Epoch 结束后，衰减 Epsilon
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            
        avg_loss = epoch_loss / max(1, total_steps)
        final_dsc = info.get('current_dsc', 0.0)
        
        # 将指标写入 TensorBoard
        writer.add_scalar("Train/Avg_Loss", avg_loss, epoch)
        writer.add_scalar("Train/Episode_Reward", epoch_reward, epoch)
        writer.add_scalar("Train/Final_DSC", final_dsc, epoch)
        writer.add_scalar("Hyperparameters/Epsilon", agent.epsilon, epoch)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] | Total Steps: {total_steps} | "
              f"Epsilon: {agent.epsilon:.4f} | "
              f"Avg Loss: {avg_loss:.4f} | "
              f"Episode Reward: {epoch_reward:.4f} | "
              f"Final DSC: {final_dsc:.4f}")
              
        # 每个 Epoch 结束，保存策略网络 Checkpoint
        # 实际训练中也可以只保存 DSC 最高的模型
        ckpt_path = os.path.join(ckpt_dir, f"rl_policy_epoch_{epoch+1}.pth")
        torch.save(agent.policy_net.state_dict(), ckpt_path)
        
              
    writer.close()
    return loss_history

if __name__ == "__main__":
    # 步骤 18: 全链路空跑测试 (Dry Run)
    print("--- Starting Step 18: Full Pipeline Dry Run ---")
    # 我们故意将参数调小，以便快速跑通并验证 Loss 是否能正常计算和下降
    # 这等价于 python train_rl.py --epochs 3 --max_steps 5 --image_size 32 --feature_size 8
    # sys.argv = ["train_rl.py", "--epochs", "5", "--max_steps", "10", "--image_size", "256", "--feature_size", "8"]
    # sys.argv = ["train_rl.py", "--epochs", "5", "--max_steps", "10"]
    args, device, dataloader, sam_tracker, text_encoder, proj_module, agent, loss_history, writer, ckpt_dir = setup_training()
    loss_history = train_loop(args, device, dataloader, sam_tracker, text_encoder, proj_module, agent, loss_history, writer, ckpt_dir)
    
    print("\n" + "="*40)
    print("Step 18 Dry Run Completed!")
    print(f"Total optimization steps performed: {len(loss_history)}")
    if len(loss_history) > 0:
        print(f"Initial Loss: {loss_history[0]:.4f}")
        print(f"Final Loss:   {loss_history[-1]:.4f}")
    print("="*40)
    print("SUCCESS: The RL-SAM-VLM framework is fully operational!")
    # python train_rl.py  --epochs 50  --max_steps 20  --image_size 256  --lr 1e-5  --gamma 0.99  --epsilon_start 1.0  --epsilon_end 0.02  --epsilon_decay 0.95  --memory_size 20000  --batch_size 1
