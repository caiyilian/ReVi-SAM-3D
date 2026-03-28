import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import sys
from collections import deque

# Ensure the project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.rl_agent.llm_policy import LLMDrivenPromptAgentDDQN

class ReplayBuffer:
    """
    经验回放池 (Experience Replay Buffer)
    存储 Agent 与环境交互产生的四元组 (state, action, reward, next_state, done)
    注意：我们的 state 包含两个部分：(image_features, current_bbox)
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state_img, state_bbox, action, reward, next_state_img, next_state_bbox, done):
        self.buffer.append((state_img, state_bbox, action, reward, next_state_img, next_state_bbox, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_imgs, state_bboxes, actions, rewards, next_state_imgs, next_state_bboxes, dones = zip(*batch)
        return state_imgs, state_bboxes, actions, rewards, next_state_imgs, next_state_bboxes, dones
        
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    强化学习 DQN 智能体核心调度类
    复用并魔改自 RCI-Seg 项目，加入了对大模型策略网络 (LLM Policy) 的适配以及软更新 (Soft Update)
    """
    def __init__(
        self, 
        visual_dim=256, 
        llm_dim=1536, 
        action_size=9, 
        gamma=0.99, 
        lr=1e-4, 
        buffer_capacity=10000,
        tau=0.005 # 用于软更新
    ):
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        
        # 探索与利用参数 (Epsilon-Greedy)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化 Policy Network 和 Target Network (步骤 12 写的网络)
        print("Initializing Policy Network...")
        self.policy_net = LLMDrivenPromptAgentDDQN(visual_dim, llm_dim, action_size).to(self.device)
        print("Initializing Target Network...")
        self.target_net = LLMDrivenPromptAgentDDQN(visual_dim, llm_dim, action_size).to(self.device)
        
        # 初始时，使 Target 网络和 Policy 网络的参数完全一致
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # 目标网络只用于评估预期收益，不参与梯度反向传播
        
        # 优化器：只优化 policy_net 中 requires_grad=True 的参数 (即 MLP 和投影层，冻结的 LLM 层不会被优化)
        trainable_params = [p for p in self.policy_net.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable_params, lr=lr)
        
        # 初始化经验回放池
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    def select_action(self, state_img, state_bbox, is_training=True):
        """
        根据 Epsilon-Greedy 策略选择动作
        """
        if is_training and random.random() < self.epsilon:
            # 探索 (Exploration)
            return random.randrange(self.action_size)
        else:
            # 利用 (Exploitation)
            with torch.no_grad():
                # 增加 Batch 维度
                state_img = state_img.unsqueeze(0).to(self.device)
                state_bbox = state_bbox.unsqueeze(0).to(self.device)
                
                self.policy_net.eval()
                q_values = self.policy_net(state_img, state_bbox)
                self.policy_net.train()
                
                return q_values.argmax(dim=1).item()

    def soft_update_target_net(self):
        """
        相比于 RCI-Seg 原版代码中的硬更新 (直接覆盖)，软更新 (Soft Update) 能让训练更稳定
        θ_target = τ * θ_local + (1 - τ) * θ_target
        """
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def learn(self, batch_size=32):
        """
        核心的 DQN 学习与参数更新步骤
        """
        print(f"{len(self.replay_buffer)=}")
        if len(self.replay_buffer) < batch_size:
            return 0.0 # 经验不足时不训练
        # 1. 从回放池采样
        state_imgs, state_bboxes, actions, rewards, next_state_imgs, next_state_bboxes, dones = self.replay_buffer.sample(batch_size)

        # 2. 转换为 Tensor 并移动到设备
        # 注意：这里假设送入 buffer 的已经是张量
        state_imgs = torch.stack(state_imgs).to(self.device)
        state_bboxes = torch.stack(state_bboxes).to(self.device)
        next_state_imgs = torch.stack(next_state_imgs).to(self.device)
        next_state_bboxes = torch.stack(next_state_bboxes).to(self.device)
        
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # 3. 计算当前的 Q 值: Q(S, A)
        # self.policy_net(state_imgs, state_bboxes) 输出 [batch_size, action_size]
        # .gather 提取出每个样本实际执行的那个动作对应的 Q 值 -> [batch_size]
        q_values = self.policy_net(state_imgs, state_bboxes).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 4. 计算目标的 Q 值: R + γ * max Q(S', A')
        with torch.no_grad():
            # 使用目标网络计算下一状态的最大 Q 值
            next_q_values = self.target_net(next_state_imgs, next_state_bboxes).max(1)[0]
            # 如果是 done (终止状态)，预期收益只有当前的 Reward
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        # 5. 计算 Loss (MSE 或 Huber/SmoothL1)
        loss = F.smooth_l1_loss(q_values, targets)

        # 6. 反向传播与优化
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪 (防止梯度爆炸，这对包含大量参数的网络尤为重要)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()

        # 7. 更新 Epsilon (衰减探索率)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # 8. 软更新 Target 网络
        self.soft_update_target_net()
        
        return loss.item()
