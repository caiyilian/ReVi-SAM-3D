import os
import random
from collections import deque
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from monai.losses import DiceCELoss
from transformers import AutoModelForCausalLM
from tqdm import tqdm


class _SimpleReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=int(capacity))

    def push(self, state, action, reward, next_state, done):
        state_np = np.asarray(state, dtype=np.float32).copy()
        next_state_np = np.asarray(next_state, dtype=np.float32).copy()
        self.buffer.append((state_np, int(action), float(reward), next_state_np, float(done)))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class _SimpleQNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Phase-B: Adapter -> frozen single LLM block -> Adapter
        llm_model_name = '/public/cyl/fourth_works/pretrained_weights/DeepSeek-R1-Distill-Qwen-1.5B'
        llm_layer_idx = 27
        llm_hidden_dim = 1536

        self.adapter1 = nn.Linear(hidden_dim, llm_hidden_dim)
        self.adapter2 = nn.Linear(llm_hidden_dim, hidden_dim)

        # Load full LLM on CPU, extract one layer, free the rest, then move extracted layer to training device.
        full_model = AutoModelForCausalLM.from_pretrained(llm_model_name, torch_dtype=torch.float32)
        self.llm_layer = full_model.model.layers[llm_layer_idx]
        for p in self.llm_layer.parameters():
            p.requires_grad = False
        self.llm_layer.eval()

        del full_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.value_head = nn.Linear(hidden_dim, 1)
        self.advantage_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        feat = self.feature(x)
        llm_in = self.adapter1(feat).unsqueeze(1)
        seq_len = llm_in.shape[1]
        sin = torch.sin(torch.arange(seq_len, device=llm_in.device, dtype=llm_in.dtype)).view(1, seq_len, 1)
        cos = torch.cos(torch.arange(seq_len, device=llm_in.device, dtype=llm_in.dtype)).view(1, seq_len, 1)
        llm_out = self.llm_layer(hidden_states=llm_in, position_embeddings=[sin, cos])[0]
        feat = self.adapter2(llm_out.squeeze(1))
        value = self.value_head(feat)
        advantage = self.advantage_head(feat)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class BaseTrainer:
    """Joint trainer for SAM-Med3D and student 3DUNet.

    The two branches are optimized in parallel in each iteration, while keeping
    their supervision independent (no distillation in this stage).
    """

    def __init__(
        self,
        model,
        student_model,
        dataloaders,
        val_dataloaders,
        args,
        logger,
        model_save_path,
        click_methods,
        img_datas,
        device,
        pseudo_label_save_dir=None,
    ):
        """Initialize trainer state, optimizers, schedulers and resume state."""
        self.model = model
        self.student_model = student_model
        self.dataloaders = dataloaders
        self.val_dataloaders = val_dataloaders
        self.args = args
        self.logger = logger
        self.model_save_path = model_save_path
        self.click_methods = click_methods
        self.img_datas = img_datas
        self.device = device
        self.pseudo_label_save_dir = pseudo_label_save_dir or os.path.join(model_save_path, 'pseudo_labels')

        self._init_tracking_state()
        self._point_prompt_debug_count = 0
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()

        if args.resume:
            self.init_checkpoint(os.path.join(self.args.work_dir, self.args.task_name, 'sam_model_latest.pth'))
        else:
            self.init_checkpoint(self.args.checkpoint)

        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
        self._debug_validation_dataset()
        self._init_rl_prompt_agent()

    def _init_tracking_state(self):
        # Best scalar records
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.best_val_dice = 0.0
        self.best_student_dice = 0.0
        self.best_val_student_dice = 0.0

        # Best-epoch records
        self.best_train_dice_epoch = -1
        self.best_student_train_dice_epoch = -1
        self.best_val_dice_epoch = -1
        self.best_val_student_dice_epoch = -1

        # Best per-class snapshots
        self.best_val_dice_dict = {}
        self.best_val_student_dice_dict = {}

        # Historical curves (train/val)
        self.losses = []
        self.dices = []
        self.val_dices = []
        self.student_losses = []
        self.student_dices = []
        self.val_student_dices = []

        # Historical per-class buffers
        self.epoch_dices_dict_all = []
        self.epoch_student_dices_dict_all = []
        self.val_epochs = []
        self.val_epoch_dices_dict_all = []
        self.val_epoch_student_dices_dict_all = []

        self.step_best_loss = np.inf
        self.step_best_dice = 0.0
        self.ious = []
        
        # Stage-2A: Pseudo-label tracking
        self.pseudo_label_stats = []  # List of per-epoch pseudo label quality stats
        self.pseudo_label_epoch_dice = []  # Dice between pseudo labels and GT (for quality assessment)
        # Stage-1 RL prompt scaffold logs
        self.rl_prompt_step_logs = []
        self.rl_epoch_stats = {
            'episodes': 0,
            'returns': [],
            'rewards': [],
            'deltas': [],
            'invalid': 0,
            'steps': 0,
            'learn_steps': 0,
            'learn_losses': [],
            'learn_kl_losses': [],
        }

    def set_loss_fn(self):
        self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        self.student_seg_loss = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, reduction='mean')

    def _debug_validation_dataset(self):
        if self.val_dataloaders is None:
            print('[DEBUG] Validation dataloaders is None!')
            return

        try:
            print('[DEBUG] Validation dataset info:')
            print(f"  - Dataset length: {len(self.val_dataloaders.dataset)}")
            sample = self.val_dataloaders.dataset[0]
            print(f"  - First sample image shape: {sample['image'].shape}")
            print(f"  - First sample label shape: {sample['label'].shape}")
            print(f"  - First sample label unique values: {torch.unique(sample['label'])}")
        except Exception as e:
            print(f'[DEBUG] Error checking validation dataset: {e}')

    def set_optimizer(self):
        if self.args.multi_gpu:
            sam_model = self.model.module
            student_model = self.student_model.module
        else:
            sam_model = self.model
            student_model = self.student_model

        self.optimizer = torch.optim.AdamW(
            [
                {'params': sam_model.image_encoder.parameters()},
                {'params': sam_model.prompt_encoder.parameters(), 'lr': self.args.lr * 0.1},
                {'params': sam_model.mask_decoder.parameters(), 'lr': self.args.lr * 0.1},
            ],
            lr=self.args.lr,
            betas=(0.9, 0.999),
            weight_decay=self.args.weight_decay,
        )

        self.student_optimizer = torch.optim.AdamW(
            student_model.parameters(),
            lr=self.args.student_lr,
            betas=(0.9, 0.999),
            weight_decay=self.args.student_weight_decay,
        )

    def set_lr_scheduler(self):
        if self.args.lr_scheduler == 'multisteplr':
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.args.step_size, self.args.gamma)
            self.student_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.student_optimizer, self.args.step_size, self.args.gamma)
        elif self.args.lr_scheduler == 'steplr':
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.args.step_size[0], self.args.gamma)
            self.student_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.student_optimizer, self.args.step_size[0], self.args.gamma)
        elif self.args.lr_scheduler == 'coswarm':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer)
            self.student_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.student_optimizer)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 0.1)
            self.student_lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.student_optimizer, 0.1)

    def init_checkpoint(self, ckp_path):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.multi_gpu:
                dist.barrier()
                last_ckpt = torch.load(ckp_path, map_location=self.args.device, weights_only=False)
            else:
                last_ckpt = torch.load(ckp_path, map_location=self.args.device, weights_only=False)

        if last_ckpt:
            if self.args.allow_partial_weight:
                if self.args.multi_gpu:
                    self.model.module.load_state_dict(last_ckpt['model_state_dict'], strict=False)
                    if 'student_model_state_dict' in last_ckpt:
                        self.student_model.module.load_state_dict(last_ckpt['student_model_state_dict'], strict=False)
                else:
                    self.model.load_state_dict(last_ckpt['model_state_dict'], strict=False)
                    if 'student_model_state_dict' in last_ckpt:
                        self.student_model.load_state_dict(last_ckpt['student_model_state_dict'], strict=False)
            else:
                if self.args.multi_gpu:
                    self.model.module.load_state_dict(last_ckpt['model_state_dict'])
                    if 'student_model_state_dict' in last_ckpt:
                        self.student_model.module.load_state_dict(last_ckpt['student_model_state_dict'])
                else:
                    self.model.load_state_dict(last_ckpt['model_state_dict'])
                    if 'student_model_state_dict' in last_ckpt:
                        self.student_model.load_state_dict(last_ckpt['student_model_state_dict'])

            if not self.args.resume:
                self.start_epoch = 0
            else:
                self.start_epoch = last_ckpt['epoch']
                self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
                if 'student_optimizer_state_dict' in last_ckpt:
                    self.student_optimizer.load_state_dict(last_ckpt['student_optimizer_state_dict'])
                if 'student_lr_scheduler_state_dict' in last_ckpt:
                    self.student_lr_scheduler.load_state_dict(last_ckpt['student_lr_scheduler_state_dict'])
                self.losses = last_ckpt['losses']
                self.dices = last_ckpt['dices']
                self.best_loss = last_ckpt['best_loss']
                self.best_dice = last_ckpt['best_dice']
                self.student_losses = last_ckpt.get('student_losses', [])
                self.student_dices = last_ckpt.get('student_dices', [])
                self.val_student_dices = last_ckpt.get('val_student_dices', [])
                self.best_student_dice = last_ckpt.get('best_student_dice', 0.0)
                self.best_val_student_dice = last_ckpt.get('best_val_student_dice', 0.0)
                self.best_train_dice_epoch = last_ckpt.get('best_train_dice_epoch', -1)
                self.best_student_train_dice_epoch = last_ckpt.get('best_student_train_dice_epoch', -1)
                self.best_val_dice_epoch = last_ckpt.get('best_val_dice_epoch', -1)
                self.best_val_student_dice_epoch = last_ckpt.get('best_val_student_dice_epoch', -1)
                self.best_val_dice_dict = last_ckpt.get('best_val_dice_dict', {})
                self.best_val_student_dice_dict = last_ckpt.get('best_val_student_dice_dict', {})
                # Stage-2A: Restore pseudo label stats
                self.pseudo_label_stats = last_ckpt.get('pseudo_label_stats', [])
                self.pseudo_label_epoch_dice = last_ckpt.get('pseudo_label_epoch_dice', [])
            print(f'Loaded checkpoint from {ckp_path} (epoch {self.start_epoch})')
        else:
            self.start_epoch = 0
            print(f'No checkpoint found at {ckp_path}, start training from scratch')

    def save_checkpoint(self, epoch, state_dict, describe='last'):
        if self.args.multi_gpu:
            student_state_dict = self.student_model.module.state_dict()
        else:
            student_state_dict = self.student_model.state_dict()

        torch.save(
            {
                'epoch': epoch + 1,
                'model_state_dict': state_dict,
                'student_model_state_dict': student_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                'student_optimizer_state_dict': self.student_optimizer.state_dict(),
                'student_lr_scheduler_state_dict': self.student_lr_scheduler.state_dict(),
                'losses': self.losses,
                'dices': self.dices,
                'student_losses': self.student_losses,
                'student_dices': self.student_dices,
                'val_student_dices': self.val_student_dices,
                'best_loss': self.best_loss,
                'best_dice': self.best_dice,
                'best_student_dice': self.best_student_dice,
                'best_val_student_dice': self.best_val_student_dice,
                'best_train_dice_epoch': self.best_train_dice_epoch,
                'best_student_train_dice_epoch': self.best_student_train_dice_epoch,
                'best_val_dice_epoch': self.best_val_dice_epoch,
                'best_val_student_dice_epoch': self.best_val_student_dice_epoch,
                'best_val_dice_dict': self.best_val_dice_dict,
                'best_val_student_dice_dict': self.best_val_student_dice_dict,
                'args': self.args,
                'used_datas': self.img_datas,
                'pseudo_label_stats': self.pseudo_label_stats,
                'pseudo_label_epoch_dice': self.pseudo_label_epoch_dice,
            },
            os.path.join(self.model_save_path, f'sam_model_{describe}.pth'),
        )

    def _init_rl_prompt_agent(self):
        self.rl_agent = None
        self.rl_optimizer = None
        self.rl_replay = None
        self.rl_global_step = 0
        self.rl_opt_step = 0
        self.rl_pending_updates = 0
        self.rl_action_dim = 7  # +D, -D, +H, -H, +W, -W, STOP

        if not bool(getattr(self.args, 'rl_prompt_enable', 0)):
            return

        state_dim = int(getattr(self.args, 'rl_prompt_state_dim', 10))
        hidden_dim = int(getattr(self.args, 'rl_prompt_hidden_dim', 128))
        self.rl_agent = _SimpleQNet(state_dim, hidden_dim, self.rl_action_dim).to(self.device)
        self.rl_target_agent = _SimpleQNet(state_dim, hidden_dim, self.rl_action_dim).to(self.device)
        self.rl_target_agent.load_state_dict(self.rl_agent.state_dict())
        self.rl_target_agent.eval()

        self.rl_optimizer = torch.optim.Adam(self.rl_agent.parameters(), lr=float(self.args.rl_dqn_lr))
        self.rl_replay = _SimpleReplayBuffer(capacity=int(self.args.rl_dqn_buffer_size))
        self.rl_epsilon = float(self.args.rl_dqn_epsilon_start)

    def _rl_learning_enabled(self):
        return bool(self._is_rl_prompt_enabled() and (self.rl_agent is not None))

    def _store_rl_transition(self, state_vec, action, reward, next_state_vec, done):
        if not self._rl_learning_enabled():
            return

        self.rl_replay.push(
            state_vec.detach().float().cpu().numpy(),
            int(action),
            float(reward),
            next_state_vec.detach().float().cpu().numpy(),
            float(done),
        )
        self.rl_global_step += 1
        learn_every = int(getattr(self.args, 'rl_dqn_learn_every', 4))
        if self.rl_global_step % learn_every == 0:
            self.rl_pending_updates += 1

    def _init_rl_point_from_gt(self, gt3D_binary):
        # Step-2 keeps initialization simple and stable: GT centroid with small random perturbation.
        b, _, d, h, w = gt3D_binary.shape
        points = torch.zeros((b, 1, 3), device=self.device, dtype=torch.float32)
        for bi in range(b):
            pos = torch.nonzero(gt3D_binary[bi, 0] > 0, as_tuple=False)
            if pos.numel() == 0:
                center = torch.tensor([d // 2, h // 2, w // 2], device=self.device, dtype=torch.float32)
            else:
                center = pos.float().mean(dim=0)
                if random.random() < 0.2:
                    center = pos[random.randrange(pos.shape[0])].float()
            points[bi, 0] = center
        return points

    def _init_rl_point_mixed_strategy(self, gt3D_binary):
        """Mix reliable region initialization with random exploration.
        
        Strategy: 
        - With probability rl_init_reliable_ratio: initialize from reliable region (GT centroid)
        - With probability (1 - rl_init_reliable_ratio): initialize from random valid voxel
        """
        b, _, d, h, w = gt3D_binary.shape
        points = torch.zeros((b, 1, 3), device=self.device, dtype=torch.float32)
        reliable_ratio = float(getattr(self.args, 'rl_init_reliable_ratio', 0.7))
        
        for bi in range(b):
            pos = torch.nonzero(gt3D_binary[bi, 0] > 0, as_tuple=False)
            if pos.numel() == 0:
                center = torch.tensor([d // 2, h // 2, w // 2], device=self.device, dtype=torch.float32)
            else:
                if random.random() < reliable_ratio:
                    # Initialize from reliable region: GT centroid (70%)
                    center = pos.float().mean(dim=0)
                else:
                    # Initialize from random valid voxel (30%)
                    center = pos[random.randrange(pos.shape[0])].float()
            points[bi, 0] = center
        return points

    def _build_state_vector(self, prev_masks, gt3D_binary, point_dhw, click_idx, max_clicks, prev_dice):
        pred_prob = prev_masks.detach().float()
        pred_binary = (pred_prob > 0.5).float()
        gt_binary = gt3D_binary.detach().float()

        fp = ((pred_binary > 0.5) & (gt_binary <= 0.5)).float().mean()
        fn = ((pred_binary <= 0.5) & (gt_binary > 0.5)).float().mean()
        pred_pos = pred_binary.mean()
        gt_pos = gt_binary.mean()
        prob_mean = pred_prob.mean()
        prob_std = pred_prob.std()

        dsz, hsz, wsz = gt3D_binary.shape[-3:]
        point_norm = torch.tensor(
            [
                float(point_dhw[0] / max(dsz - 1, 1)),
                float(point_dhw[1] / max(hsz - 1, 1)),
                float(point_dhw[2] / max(wsz - 1, 1)),
            ],
            device=self.device,
            dtype=torch.float32,
        )
        step_ratio = torch.tensor(float(click_idx) / max(float(max_clicks - 1), 1.0), device=self.device, dtype=torch.float32)
        prev_dice_t = torch.tensor(float(prev_dice), device=self.device, dtype=torch.float32)

        state = torch.stack(
            [
                prob_mean,
                prob_std,
                pred_pos,
                gt_pos,
                fp,
                fn,
                point_norm[0],
                point_norm[1],
                point_norm[2],
                step_ratio,
                prev_dice_t,
            ]
        )

        target_dim = int(getattr(self.args, 'rl_prompt_state_dim', 10))
        if state.numel() > target_dim:
            state = state[:target_dim]
        elif state.numel() < target_dim:
            pad = torch.zeros(target_dim - state.numel(), device=self.device, dtype=state.dtype)
            state = torch.cat([state, pad], dim=0)
        return state

    def _select_rl_action(self, state_vec):
        if random.random() < self.rl_epsilon:
            return random.randrange(self.rl_action_dim)
        with torch.no_grad():
            q_values = self.rl_agent(state_vec.unsqueeze(0))
            return int(torch.argmax(q_values, dim=1).item())

    def _apply_discrete_action(self, point_dhw, action, volume_shape):
        dsz, hsz, wsz = volume_shape
        step = int(getattr(self.args, 'rl_prompt_step_size', 2))
        new_point = point_dhw.clone()

        if action == 0:
            new_point[0] += step
        elif action == 1:
            new_point[0] -= step
        elif action == 2:
            new_point[1] += step
        elif action == 3:
            new_point[1] -= step
        elif action == 4:
            new_point[2] += step
        elif action == 5:
            new_point[2] -= step

        was_invalid = False
        new_point_clamped = new_point.clone()
        new_point_clamped[0] = torch.clamp(new_point_clamped[0], 0, max(dsz - 1, 0))
        new_point_clamped[1] = torch.clamp(new_point_clamped[1], 0, max(hsz - 1, 0))
        new_point_clamped[2] = torch.clamp(new_point_clamped[2], 0, max(wsz - 1, 0))

        if not torch.allclose(new_point, new_point_clamped):
            was_invalid = True

        done = action == 6
        return new_point_clamped, done, was_invalid

    def _pack_manual_point_prompt(self, point_dhw):
        # point_dhw shape: [3]
        points = point_dhw.view(1, 1, 3).to(self.device).float()
        labels = torch.ones((1, 1), device=self.device, dtype=torch.int64)

        self.click_points.append(points)
        self.click_labels.append(labels)

        if self.args.multi_click:
            points_input = torch.cat(self.click_points, dim=1)
            labels_input = torch.cat(self.click_labels, dim=1)
        else:
            points_input = points
            labels_input = labels
        return points_input, labels_input

    def _maybe_optimize_rl_agent(self):
        if not self._rl_learning_enabled():
            return None

        batch_size = int(getattr(self.args, 'rl_dqn_batch_size', 32))
        if len(self.rl_replay) < batch_size:
            return None

        samples = self.rl_replay.sample(batch_size)
        states_np = np.stack([s[0] for s in samples], axis=0)
        next_states_np = np.stack([s[3] for s in samples], axis=0)

        states = torch.tensor(states_np, device=self.device, dtype=torch.float32)
        actions = torch.tensor([s[1] for s in samples], device=self.device, dtype=torch.long)
        rewards = torch.tensor([s[2] for s in samples], device=self.device, dtype=torch.float32)
        next_states = torch.tensor(next_states_np, device=self.device, dtype=torch.float32)
        dones = torch.tensor([s[4] for s in samples], device=self.device, dtype=torch.float32)

        self.rl_agent.train()
        with torch.enable_grad():
            with torch.amp.autocast('cuda', enabled=False):
                # Phase-C stabilizer 1: reward normalization (batch-wise z-score)
                reward_std = rewards.std(unbiased=False)
                if reward_std > 1e-6:
                    rewards_norm = (rewards - rewards.mean()) / reward_std
                else:
                    rewards_norm = rewards - rewards.mean()

                q_all = self.rl_agent(states)
                q_values = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = self.rl_target_agent(next_states).max(dim=1)[0]
                    targets = rewards_norm + float(self.args.rl_dqn_gamma) * next_q * (1.0 - dones)

                td_loss = F.smooth_l1_loss(q_values, targets)

                # Phase-C stabilizer 2: lightweight KL regularization to target policy.
                kl_coef = 0.01
                with torch.no_grad():
                    q_ref = self.rl_target_agent(states)
                    ref_prob = F.softmax(q_ref, dim=1)
                policy_log_prob = F.log_softmax(q_all, dim=1)
                kl_loss = F.kl_div(policy_log_prob, ref_prob, reduction='batchmean')

                loss = td_loss + kl_coef * kl_loss

        if (not q_values.requires_grad) or (not loss.requires_grad):
            self.logger.warning(
                '[RL-PROMPT-LEARN][SKIP] invalid autograd graph: '
                f'grad_enabled={torch.is_grad_enabled()}, '
                f'q_requires_grad={q_values.requires_grad}, '
                f'loss_requires_grad={loss.requires_grad}, '
                f'buffer_len={len(self.rl_replay)}'
            )
            return None

        self.rl_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rl_agent.parameters(), max_norm=1.0)
        self.rl_optimizer.step()

        if hasattr(self, 'rl_epoch_stats'):
            self.rl_epoch_stats['learn_kl_losses'].append(float(kl_loss.item()))

        self.rl_opt_step += 1
        if self.rl_opt_step % int(getattr(self.args, 'rl_dqn_target_update_every', 100)) == 0:
            self.rl_target_agent.load_state_dict(self.rl_agent.state_dict())

        decay = float(getattr(self.args, 'rl_dqn_epsilon_decay', 0.995))
        eps_end = float(getattr(self.args, 'rl_dqn_epsilon_end', 0.05))
        self.rl_epsilon = max(eps_end, self.rl_epsilon * decay)
        return float(loss.item())

    def batch_forward(self, sam_model, image_embedding, gt3D, low_res_masks, points=None):
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(points=points, boxes=None, masks=low_res_masks)
        low_res_masks, _ = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(self.device),
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)
        return low_res_masks, prev_masks

    def get_points(self, prev_masks, gt3D, use_gt_prompt=True):
        if not use_gt_prompt:
            click_fn = self.click_methods['no_gt_naive']
            batch_points, batch_labels = click_fn(prev_masks)
        else:
            click_fn = self.click_methods[self.args.click_type]
            if self.args.click_type == 'no_gt_naive':
                batch_points, batch_labels = click_fn(prev_masks)
            else:
                batch_points, batch_labels = click_fn(prev_masks, gt3D)

        points_co = torch.cat(batch_points, dim=0).to(self.device)
        points_la = torch.cat(batch_labels, dim=0).to(self.device)

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        points_multi = torch.cat(self.click_points, dim=1).to(self.device)
        labels_multi = torch.cat(self.click_labels, dim=1).to(self.device)

        if self.args.multi_click:
            points_input = points_multi
            labels_input = labels_multi
        else:
            points_input = points_co
            labels_input = points_la

        if self.args.debug_point_prompt_prints > self._point_prompt_debug_count:
            points_dbg = points_input.detach().float().cpu()
            labels_dbg = labels_input.detach().cpu()
            pmin = points_dbg.amin(dim=(0, 1)).tolist()
            pmax = points_dbg.amax(dim=(0, 1)).tolist()
            norm_points_dbg = points_dbg / float(self.args.img_size)
            pnmin = norm_points_dbg.amin(dim=(0, 1)).tolist()
            pnmax = norm_points_dbg.amax(dim=(0, 1)).tolist()

            print('[DEBUG_POINT] raw points shape:', tuple(points_dbg.shape))
            print('[DEBUG_POINT] raw coord min (d/h/w order):', [round(v, 4) for v in pmin])
            print('[DEBUG_POINT] raw coord max (d/h/w order):', [round(v, 4) for v in pmax])
            print('[DEBUG_POINT] proxy norm min (coord/img_size):', [round(v, 6) for v in pnmin])
            print('[DEBUG_POINT] proxy norm max (coord/img_size):', [round(v, 6) for v in pnmax])
            print('[DEBUG_POINT] labels unique:', torch.unique(labels_dbg).tolist())
            print('[DEBUG_POINT] gt shape:', tuple(gt3D.shape[-3:]), 'img_size arg:', self.args.img_size)
            self._point_prompt_debug_count += 1

        return points_input, labels_input

    def _dice_from_binary_masks(self, pred_mask, gt_mask):
        pred_binary = (pred_mask > 0.5).float()
        gt_binary = gt_mask.float()
        inter = (pred_binary * gt_binary).sum()
        union = pred_binary.sum() + gt_binary.sum()
        if union <= 0:
            return 1.0
        return float((2.0 * inter / (union + 1e-8)).item())

    def _iou_from_binary_masks(self, pred_mask, gt_mask):
        """Calculate Intersection over Union (IoU) metric."""
        pred_binary = (pred_mask > 0.5).float()
        gt_binary = gt_mask.float()
        inter = (pred_binary * gt_binary).sum()
        union = (pred_binary + gt_binary - pred_binary * gt_binary).sum()
        if union <= 0:
            return 1.0
        return float((inter / (union + 1e-8)).item())

    def _boundary_f1_from_binary_masks(self, pred_mask, gt_mask, kernel_size=5):
        """Calculate Boundary F1 metric (detection of boundary regions)."""
        pred_binary = (pred_mask > 0.5).float()
        gt_binary = gt_mask.float()
        
        # Ensure input is 5D: [batch, channel, D, H, W]
        # Input might be [1, D, H, W] or [D, H, W]
        if pred_binary.ndim == 3:
            pred_binary = pred_binary.unsqueeze(0).unsqueeze(0)  # [D,H,W] -> [1,1,D,H,W]
        elif pred_binary.ndim == 4:
            pred_binary = pred_binary.unsqueeze(1)  # [1,D,H,W] -> [1,1,D,H,W]
        # else: already 5D, do nothing
        
        if gt_binary.ndim == 3:
            gt_binary = gt_binary.unsqueeze(0).unsqueeze(0)
        elif gt_binary.ndim == 4:
            gt_binary = gt_binary.unsqueeze(1)
        # else: already 5D, do nothing
        
        # Now both should be 5D: [batch, channel, D, H, W]
        kernel = torch.ones(1, 1, kernel_size, kernel_size, kernel_size, device=pred_binary.device) / (kernel_size ** 3)
        
        pred_border = torch.clamp(
            torch.nn.functional.conv3d(
                pred_binary,
                kernel,
                padding=kernel_size // 2
            ).squeeze(0).squeeze(0),
            0, 1
        )
        gt_border = torch.clamp(
            torch.nn.functional.conv3d(
                gt_binary,
                kernel,
                padding=kernel_size // 2
            ).squeeze(0).squeeze(0),
            0, 1
        )
        
        inter = ((pred_border > 0.3) * (gt_border > 0.3)).sum().float()
        union = ((pred_border > 0.3) + (gt_border > 0.3)).sum().float()
        if union <= 0:
            return 1.0
        boundary_iou = inter / (union + 1e-8)
        return float(boundary_iou.item())

    def _compute_rl_reward(self, curr_dice, prev_dice, curr_masks=None, prev_masks=None, gt3D_binary=None, was_invalid=False):
        """Compute RL reward based on configured composition mode for LABELED data.
        
        Composition modes:
        - 'simple': reward = alpha * (curr_dice - prev_dice) - step_penalty - invalid_penalty
        - 'extended': reward = alpha*delta_dice + beta*delta_iou + gamma*delta_boundary - step_penalty - invalid_penalty
        
        Args:
            curr_dice: Current step Dice score
            prev_dice: Previous step Dice score
            curr_masks: Current predicted masks (for IoU/Boundary computation)
            prev_masks: Previous predicted masks (for IoU/Boundary computation)
            gt3D_binary: Ground truth binary mask (for metric computation)
            was_invalid: Whether this action was invalid (out of bounds)
        
        Returns:
            float: Clipped reward value in [-clip_abs, clip_abs]
        """
        composition = getattr(self.args, 'rl_reward_composition', 'extended')
        step_penalty = float(getattr(self.args, 'rl_reward_step_penalty', 0.01))
        invalid_penalty = float(getattr(self.args, 'rl_reward_invalid_penalty', 0.05))
        clip_abs = float(getattr(self.args, 'rl_reward_clip_abs', 1.0))
        
        delta_dice = curr_dice - prev_dice
        reward = 0.0
        
        if composition == 'simple':
            alpha = float(getattr(self.args, 'rl_reward_alpha', 1.0))
            reward = alpha * delta_dice
        else:  # 'extended'
            alpha = float(getattr(self.args, 'rl_reward_alpha', 1.0))
            beta = float(getattr(self.args, 'rl_reward_beta', 0.3))
            gamma = float(getattr(self.args, 'rl_reward_gamma', 0.2))
            
            # Primary term: Delta Dice
            reward = alpha * delta_dice
            
            # Extended terms: Delta IoU and Delta Boundary F1
            if curr_masks is not None and prev_masks is not None and gt3D_binary is not None:
                curr_iou = self._iou_from_binary_masks(curr_masks, gt3D_binary)
                prev_iou = self._iou_from_binary_masks(prev_masks, gt3D_binary)
                delta_iou = curr_iou - prev_iou
                reward += beta * delta_iou
                
                curr_boundary = self._boundary_f1_from_binary_masks(curr_masks, gt3D_binary)
                prev_boundary = self._boundary_f1_from_binary_masks(prev_masks, gt3D_binary)
                delta_boundary = curr_boundary - prev_boundary
                reward += gamma * delta_boundary
        
        reward -= step_penalty
        if was_invalid:
            reward -= invalid_penalty
        
        reward = max(-clip_abs, min(clip_abs, reward))
        return float(reward)

    def _compute_proxy_reward_for_unlabeled(self, curr_sam, curr_student, prev_sam, prev_student, curr_entropy=None, prev_entropy=None):
        """Compute proxy reward for UNLABELED data using three sub-signals.
        
        Since GT is unavailable, use proxy signals:
        1. Consistency gain: SAM-Student agreement improvement
        2. Confidence gain: Predictions moving away from 0.5 (higher margin)
        3. Entropy reduction: Decreased prediction uncertainty
        
        Args:
            curr_sam: Current SAM prediction (binary or soft)
            curr_student: Current Student prediction (binary or soft)
            prev_sam: Previous SAM prediction
            prev_student: Previous Student prediction
            curr_entropy: Current entropy (if precomputed, optional)
            prev_entropy: Previous entropy (if precomputed, optional)
        
        Returns:
            float: Clipped proxy reward in [-clip_abs, clip_abs]
        """
        step_penalty = float(getattr(self.args, 'rl_reward_step_penalty', 0.01))
        clip_abs = float(getattr(self.args, 'rl_reward_clip_abs', 1.0))
        
        # Sub-reward weights from config
        consistency_wt = float(getattr(self.args, 'rl_proxy_consistency_weight', 0.5))
        confidence_wt = float(getattr(self.args, 'rl_proxy_confidence_weight', 0.3))
        entropy_wt = float(getattr(self.args, 'rl_proxy_entropy_weight', 0.2))
        
        # Normalize weights so they sum to 1
        total_wt = consistency_wt + confidence_wt + entropy_wt
        if total_wt <= 0:
            # Default weights if all were zero
            consistency_wt, confidence_wt, entropy_wt = 0.5, 0.3, 0.2
            total_wt = 1.0
        consistency_wt /= total_wt
        confidence_wt /= total_wt
        entropy_wt /= total_wt
        
        # Convert to binary if soft predictions
        curr_sam_bin = (curr_sam > 0.5).float() if curr_sam.dtype != torch.bool else curr_sam.float()
        curr_student_bin = (curr_student > 0.5).float() if curr_student.dtype != torch.bool else curr_student.float()
        prev_sam_bin = (prev_sam > 0.5).float() if prev_sam.dtype != torch.bool else prev_sam.float()
        prev_student_bin = (prev_student > 0.5).float() if prev_student.dtype != torch.bool else prev_student.float()
        
        reward = 0.0
        
        # Sub-reward 1: Consistency gain (SAM-Student agreement improvement)
        # Delta agreement = higher agreement is better
        curr_agreement = (curr_sam_bin == curr_student_bin).float().mean()
        prev_agreement = (prev_sam_bin == prev_student_bin).float().mean()
        delta_agreement = curr_agreement - prev_agreement
        consistency_reward = float(delta_agreement.item()) * consistency_wt
        reward += consistency_reward
        
        # Sub-reward 2: Confidence gain (margin from 0.5)
        # Higher confidence (farther from 0.5) is better
        curr_margin = torch.abs(curr_sam - 0.5).mean() + torch.abs(curr_student - 0.5).mean()
        prev_margin = torch.abs(prev_sam - 0.5).mean() + torch.abs(prev_student - 0.5).mean()
        delta_margin = curr_margin - prev_margin
        confidence_reward = float((delta_margin / 2.0).item()) * confidence_wt
        reward += confidence_reward
        
        # Sub-reward 3: Entropy reduction (decreased uncertainty)
        # Lower entropy is better. Use SAM uncertainty here so the term changes within the episode.
        if curr_entropy is None and prev_entropy is None:
            curr_sam_prob = torch.sigmoid(curr_sam) if (float(curr_sam.min().item()) < 0.0 or float(curr_sam.max().item()) > 1.0) else curr_sam
            prev_sam_prob = torch.sigmoid(prev_sam) if (float(prev_sam.min().item()) < 0.0 or float(prev_sam.max().item()) > 1.0) else prev_sam
            curr_entropy_val = -torch.sum(
                curr_sam_prob * torch.log(torch.clamp(curr_sam_prob, min=1e-8)) +
                (1 - curr_sam_prob) * torch.log(torch.clamp(1 - curr_sam_prob, min=1e-8))
            ) / curr_sam_prob.numel()
            prev_entropy_val = -torch.sum(
                prev_sam_prob * torch.log(torch.clamp(prev_sam_prob, min=1e-8)) +
                (1 - prev_sam_prob) * torch.log(torch.clamp(1 - prev_sam_prob, min=1e-8))
            ) / prev_sam_prob.numel()
        else:
            curr_entropy_val = curr_entropy if curr_entropy is not None else torch.tensor(0.0)
            prev_entropy_val = prev_entropy if prev_entropy is not None else torch.tensor(0.0)
        
        delta_entropy = prev_entropy_val - curr_entropy_val  # Negative entropy reduction
        entropy_reward = float(delta_entropy.item()) * entropy_wt if isinstance(delta_entropy, torch.Tensor) else delta_entropy * entropy_wt
        reward += entropy_reward
        
        # Apply step penalty
        reward -= step_penalty
        
        # Clip reward
        reward = max(-clip_abs, min(clip_abs, reward))
        return float(reward)

    def _is_rl_prompt_enabled(self):
        return bool(getattr(self.args, 'rl_prompt_enable', 0))

    def _run_supervised_prompt_episode_for_class(self, sam_model, image_embedding, gt3D_binary, low_res_masks, class_id_int, num_clicks):
        """GT-prompt supervised branch used to compute SAM loss and training Dice."""
        prev_masks = F.interpolate(
            low_res_masks,
            size=gt3D_binary.shape[-3:],
            mode='trilinear',
            align_corners=False,
        )
        class_total_loss = torch.tensor(0.0, device=gt3D_binary.device)
        prev_step_dice = 0.0
        random_insert = np.random.randint(2, 9)

        for num_click in range(num_clicks):
            points_input, labels_input = self.get_points(prev_masks, gt3D_binary, use_gt_prompt=True)
            if num_click == random_insert or num_click == num_clicks - 1:
                low_res_masks, prev_masks = self.batch_forward(
                    sam_model,
                    image_embedding,
                    gt3D_binary,
                    low_res_masks,
                    points=None,
                )
            else:
                low_res_masks, prev_masks = self.batch_forward(
                    sam_model,
                    image_embedding,
                    gt3D_binary,
                    low_res_masks,
                    points=[points_input, labels_input],
                )

            step_dice = self._dice_from_binary_masks(prev_masks, gt3D_binary)
            step_dice_delta = step_dice - prev_step_dice
            self._append_rl_prompt_log(
                class_id_int=class_id_int,
                click_idx=num_click,
                prompt_source='gt_supervised',
                points_input=points_input,
                step_dice=step_dice,
                step_dice_delta=step_dice_delta,
            )
            prev_step_dice = step_dice
            class_total_loss += self.seg_loss(prev_masks, gt3D_binary)

        return prev_masks, class_total_loss

    def _append_rl_prompt_log(self, class_id_int, click_idx, prompt_source, points_input, step_dice, step_dice_delta):
        if not self._is_rl_prompt_enabled():
            return

        if len(self.rl_prompt_step_logs) >= int(getattr(self.args, 'rl_prompt_log_limit', 200)):
            return

        points_cpu = points_input.detach().float().cpu()
        point_xyz = points_cpu[0, 0].tolist() if points_cpu.ndim == 3 else []
        self.rl_prompt_step_logs.append(
            {
                'class_id': int(class_id_int),
                'click_idx': int(click_idx),
                'source': prompt_source,
                'point_dhw': [float(v) for v in point_xyz],
                'dice': float(step_dice),
                'delta_dice': float(step_dice_delta),
            }
        )

    def _run_rl_prompt_episode_for_class(self, sam_model, image_embedding, gt3D_binary, low_res_masks, class_id_int, num_clicks):
        """RL reward branch: no SAM supervision loss, no SAM gradient update."""
        max_steps = min(int(getattr(self.args, 'rl_prompt_max_steps', 5)), int(num_clicks))
        
        # Select initialization strategy (simple, mixed, guided)
        init_strategy = getattr(self.args, 'rl_init_strategy', 'mixed')
        if init_strategy == 'mixed':
            point_dhw = self._init_rl_point_mixed_strategy(gt3D_binary)[0, 0]
        else:
            # Default to GT-based initialization for 'random' and 'guided'
            point_dhw = self._init_rl_point_from_gt(gt3D_binary)[0, 0]

        prev_dice = self._dice_from_binary_masks(low_res_masks.new_zeros(gt3D_binary.shape), gt3D_binary)
        # Keep copy of mask at full resolution for IoU/Boundary computation
        prev_masks_full = F.interpolate(low_res_masks, size=gt3D_binary.shape[-3:], mode='trilinear', align_corners=False)
        episode_return = 0.0

        for num_click in range(max_steps):
            state_vec = self._build_state_vector(
                prev_masks=prev_masks_full,
                gt3D_binary=gt3D_binary,
                point_dhw=point_dhw,
                click_idx=num_click,
                max_clicks=max_steps,
                prev_dice=prev_dice,
            )
            action = self._select_rl_action(state_vec)
            new_point_dhw, done_by_action, was_invalid = self._apply_discrete_action(
                point_dhw,
                action,
                gt3D_binary.shape[-3:],
            )
            point_dhw = new_point_dhw

            points_input, labels_input = self._pack_manual_point_prompt(point_dhw)
            with torch.no_grad():
                low_res_masks, prev_masks = self.batch_forward(
                    sam_model,
                    image_embedding,
                    gt3D_binary,
                    low_res_masks,
                    points=[points_input, labels_input],
                )

            step_dice = self._dice_from_binary_masks(prev_masks, gt3D_binary)
            delta_dice = step_dice - prev_dice
            
            # Compute reward with full support for extended metrics (IoU, Boundary)
            reward = self._compute_rl_reward(
                curr_dice=step_dice,
                prev_dice=prev_dice,
                curr_masks=prev_masks,
                prev_masks=prev_masks_full,
                gt3D_binary=gt3D_binary,
                was_invalid=was_invalid
            )
            
            done = bool(done_by_action or (num_click == max_steps - 1))
            next_state_vec = self._build_state_vector(
                prev_masks=prev_masks,
                gt3D_binary=gt3D_binary,
                point_dhw=point_dhw,
                click_idx=min(num_click + 1, max_steps - 1),
                max_clicks=max_steps,
                prev_dice=step_dice,
            )

            self._store_rl_transition(
                state_vec=state_vec,
                action=action,
                reward=reward,
                next_state_vec=next_state_vec,
                done=done,
            )

            self._append_rl_prompt_log(
                class_id_int=class_id_int,
                click_idx=num_click,
                prompt_source='rl_reward_only',
                points_input=points_input,
                step_dice=step_dice,
                step_dice_delta=delta_dice,
            )

            self.rl_epoch_stats['steps'] += 1
            self.rl_epoch_stats['rewards'].append(float(reward))
            self.rl_epoch_stats['deltas'].append(float(delta_dice))
            self.rl_epoch_stats['invalid'] += int(was_invalid)

            prev_dice = step_dice
            prev_masks_full = prev_masks.detach()  # Update for next iteration's IoU/Boundary computation
            episode_return += float(reward)

            if done_by_action:
                break

        self.rl_epoch_stats['episodes'] += 1
        self.rl_epoch_stats['returns'].append(float(episode_return))
        return

    def _run_rl_prompt_episode_for_unlabeled_class(self, sam_model, image_embedding, pseudo_binary, 
                                                    low_res_masks, class_id_int, num_clicks, 
                                                    student_logits):
        """RL episode for UNLABELED data using proxy rewards (no ground truth).
        
        Uses student model predictions + proxy reward signals to guide exploration.
        Three proxy signals: consistency (SAM-Student agreement), confidence (margin from 0.5),
        entropy reduction (Student uncertainty decrease).
        
        Args:
            sam_model: SAM2 model
            image_embedding: Pre-computed image embeddings from SAM encoder
            pseudo_binary: Pseudo-labels (student predictions binarized) as initial mask
            low_res_masks: Low-resolution masks from previous iteration
            class_id_int: Class ID (int)
            num_clicks: Max number of clicks in this episode
            student_logits: Full student model logits [B, C, D, H, W], will extract class_id_int
            
        Returns:
            prev_masks_full: Final high-resolution masks after RL episode
        """
        if not self._rl_learning_enabled():
            # Return interpolated masks if RL disabled
            return F.interpolate(low_res_masks, size=pseudo_binary.shape[-3:], 
                               mode='trilinear', align_corners=False)
        proxy_reward_weight = float(getattr(self.args, 'rl_proxy_reward_weight', 0.5))
            
        max_steps = min(int(getattr(self.args, 'rl_prompt_max_steps', 5)), int(num_clicks))
        
        # Initialize point from pseudo-labels (no GT centroid available)
        point_dhw = self._init_rl_point_mixed_strategy(pseudo_binary)[0, 0]
        
        # Compute initial Dice against pseudo-labels (as proxy reference)
        prev_dice = self._dice_from_binary_masks(low_res_masks.new_zeros(pseudo_binary.shape), pseudo_binary)
        prev_masks_full = F.interpolate(low_res_masks, size=pseudo_binary.shape[-3:], 
                                        mode='trilinear', align_corners=False)
        
        # Extract student predictions for this class [1, 1, D, H, W]
        if student_logits.shape[1] > class_id_int:
            student_logits_class = student_logits[:, class_id_int:class_id_int+1, :, :, :]
        else:
            # Fallback if class_id_int not found
            return prev_masks_full
        
        prev_student_pred = torch.sigmoid(student_logits_class.detach())
        
        episode_return = 0.0
        for num_click in range(max_steps):
            state_vec = self._build_state_vector(
                prev_masks=prev_masks_full,
                gt3D_binary=pseudo_binary,  # Use pseudo as reference for state building
                point_dhw=point_dhw,
                click_idx=num_click,
                max_clicks=max_steps,
                prev_dice=prev_dice,
            )
            action = self._select_rl_action(state_vec)
            new_point_dhw, done_by_action, was_invalid = self._apply_discrete_action(
                point_dhw,
                action,
                pseudo_binary.shape[-3:],
            )
            point_dhw = new_point_dhw
            
            points_input, labels_input = self._pack_manual_point_prompt(point_dhw)
            with torch.no_grad():
                low_res_masks, curr_masks = self.batch_forward(
                    sam_model,
                    image_embedding,
                    pseudo_binary,  # Use pseudo-binary as the target reference
                    low_res_masks,
                    points=[points_input, labels_input],
                )
            
            step_dice = self._dice_from_binary_masks(curr_masks, pseudo_binary)
            delta_dice = step_dice - prev_dice
            
            # Proxy reward: Use student predictions + SAM predictions for guidance
            # Convert to matching resolution for comparison
            curr_student_pred = torch.sigmoid(student_logits_class.detach())
            curr_sam_pred = curr_masks  # Already binary from batch_forward
            prev_sam_pred = prev_masks_full  # Previous SAM prediction
            
            reward = self._compute_proxy_reward_for_unlabeled(
                curr_sam=curr_sam_pred,
                curr_student=curr_student_pred,
                prev_sam=prev_sam_pred,
                prev_student=prev_student_pred,
            )
            reward *= proxy_reward_weight
            
            done = bool(done_by_action or (num_click == max_steps - 1))
            next_state_vec = self._build_state_vector(
                prev_masks=curr_masks,
                gt3D_binary=pseudo_binary,
                point_dhw=point_dhw,
                click_idx=min(num_click + 1, max_steps - 1),
                max_clicks=max_steps,
                prev_dice=step_dice,
            )
            
            # Store transition for RL learning
            self._store_rl_transition(
                state_vec=state_vec,
                action=action,
                reward=reward,
                next_state_vec=next_state_vec,
                done=done,
            )
            
            self._append_rl_prompt_log(
                class_id_int=class_id_int,
                click_idx=num_click,
                prompt_source='rl_proxy_unlabeled',
                points_input=points_input,
                step_dice=step_dice,
                step_dice_delta=delta_dice,
            )
            
            self.rl_epoch_stats['steps'] += 1
            self.rl_epoch_stats['rewards'].append(float(reward))
            self.rl_epoch_stats['deltas'].append(float(delta_dice))
            self.rl_epoch_stats['invalid'] += int(was_invalid)
            
            prev_dice = step_dice
            prev_masks_full = curr_masks.detach()
            prev_student_pred = curr_student_pred.detach()
            episode_return += float(reward)
            
            if done_by_action:
                break
        
        self.rl_epoch_stats['episodes'] += 1
        self.rl_epoch_stats['returns'].append(float(episode_return))
        return prev_masks_full  # Return final high-resolution masks

    def interaction(self, sam_model, image_embedding, gt3D, num_clicks):
        unique_classes = torch.unique(gt3D)
        unique_classes = unique_classes[unique_classes != 0]

        if len(unique_classes) == 0:
            return {}, torch.tensor(0.0, device=gt3D.device)

        class_weights = {}
        if bool(self.args.use_dynamic_class_weight):
            total_foreground_pixels = (gt3D != 0).sum().float()
            for class_id in unique_classes:
                class_id_int = class_id.item()
                class_pixels = (gt3D == class_id_int).sum().float()
                pixel_ratio = class_pixels / (total_foreground_pixels + 1e-8)
                weight = 1.0 / (pixel_ratio + self.args.class_weight_smooth)
                class_weights[class_id_int] = weight
            avg_weight = sum(class_weights.values()) / len(class_weights)
            class_weights = {k: v / avg_weight for k, v in class_weights.items()}
        else:
            for class_id in unique_classes:
                class_weights[class_id.item()] = 1.0

        masks_dict = {}
        total_loss = torch.tensor(0.0, device=gt3D.device)
        for class_id in unique_classes:
            class_id_int = class_id.item()
            gt3D_binary = (gt3D == class_id_int).long()

            prev_masks = torch.zeros_like(gt3D_binary).float()
            low_res_masks = F.interpolate(
                prev_masks,
                size=(self.args.img_size // 4, self.args.img_size // 4, self.args.img_size // 4),
            )

            self.click_points = []
            self.click_labels = []
            prev_masks, class_total_loss = self._run_supervised_prompt_episode_for_class(
                sam_model,
                image_embedding,
                gt3D_binary,
                low_res_masks,
                class_id_int,
                num_clicks,
            )

            if self._rl_learning_enabled():
                self.click_points = []
                self.click_labels = []
                self._run_rl_prompt_episode_for_class(
                    sam_model,
                    image_embedding,
                    gt3D_binary,
                    low_res_masks,
                    class_id_int,
                    num_clicks,
                )

            total_loss += class_total_loss * class_weights[class_id_int]
            masks_dict[class_id_int] = prev_masks

        return masks_dict, total_loss

    @staticmethod
    def _as_bool_label_status(is_labeled):
        if isinstance(is_labeled, torch.Tensor):
            if is_labeled.numel() == 1:
                return bool(is_labeled.item())
            all_labeled = bool(torch.all(is_labeled > 0).item())
            all_unlabeled = bool(torch.all(is_labeled <= 0).item())
            if not (all_labeled or all_unlabeled):
                raise ValueError(
                    'A batch contains both labeled and unlabeled samples. '
                    'Please use a homogeneous batch for Stage-2B (recommended: batch_size=1).'
                )
            return all_labeled
        if isinstance(is_labeled, (list, tuple)):
            all_labeled = all(bool(x) for x in is_labeled)
            all_unlabeled = all(not bool(x) for x in is_labeled)
            if not (all_labeled or all_unlabeled):
                raise ValueError(
                    'A batch contains both labeled and unlabeled samples. '
                    'Please use a homogeneous batch for Stage-2B (recommended: batch_size=1).'
                )
            return all_labeled
        return bool(is_labeled)

    def generate_pseudo_labels_for_sample(self, sam_model, image_embedding, gt3D, num_clicks):
        """
        Generate pseudo labels using SAM3D with automatic clicks.
        
        For Stage-2A validation, we generate pseudo labels for each class by:
        1. Iterating through detected foreground classes from GT
        2. Performing automatic multi-click iterations WITHOUT using loss
        3. Returning final masks and confidence scores as pseudo labels
        
        Args:
            sam_model: SAM-Med3D model
            image_embedding: Encoded image features
            gt3D: Ground truth labels (used only to detect which classes exist)
            num_clicks: Number of automatic click rounds per class
            
        Returns:
            pseudo_masks_dict: Dict of {class_id: {'mask': mask_tensor, 'confidence': confidence_tensor}}
            torch.tensor(0.0): Placeholder return for compatibility
        """
        unique_classes = torch.unique(gt3D)
        unique_classes = unique_classes[unique_classes != 0]

        if len(unique_classes) == 0:
            return {}, torch.tensor(0.0, device=gt3D.device)

        pseudo_masks_dict = {}

        for class_id in unique_classes:
            class_id_int = class_id.item()
            # Use GT as initialization reference only, not as direct supervision
            gt3D_binary = (gt3D == class_id_int).long()

            prev_masks = torch.zeros_like(gt3D_binary).float()
            low_res_masks = F.interpolate(
                prev_masks,
                size=(self.args.img_size // 4, self.args.img_size // 4, self.args.img_size // 4),
            )

            self.click_points = []
            self.click_labels = []
            
            # Generate pseudo labels through automatic clicks
            for num_click in range(num_clicks):
                # Auto point generation using GT-guided strategy (for validation only)
                points_input, labels_input = self.get_points(prev_masks, gt3D_binary, use_gt_prompt=True)
                
                random_insert = np.random.randint(2, 9)
                if num_click == random_insert or num_click == num_clicks - 1:
                    low_res_masks, prev_masks = self.batch_forward(
                        sam_model,
                        image_embedding,
                        gt3D_binary,
                        low_res_masks,
                        points=None,
                    )
                else:
                    low_res_masks, prev_masks = self.batch_forward(
                        sam_model,
                        image_embedding,
                        gt3D_binary,
                        low_res_masks,
                        points=[points_input, labels_input],
                    )
                # Note: We don't compute loss here, just accumulate masks

            # Detach pseudo masks to prevent gradient flow back to SAM
            # SAM should act as an independent teacher, not modified by student loss
            prev_masks_detached = prev_masks.detach()
            
            # prev_masks already contains sigmoid output (probability 0-1), use as confidence
            pseudo_masks_dict[class_id_int] = {
                'mask': prev_masks_detached,
                'confidence': prev_masks_detached,  # SAM output is already a probability
            }

        return pseudo_masks_dict, torch.tensor(0.0, device=gt3D.device)

    def generate_pseudo_labels_without_gt(self, sam_model, image_embedding, student_logits, num_clicks):
        """Generate pseudo labels for unlabeled data using RL-guided prompts (no GT)."""
        pred_labels = torch.argmax(student_logits.detach(), dim=1)
        unique_classes = torch.unique(pred_labels)
        unique_classes = unique_classes[unique_classes != 0]

        if len(unique_classes) == 0:
            return {}, torch.tensor(0.0, device=student_logits.device)

        pseudo_masks_dict = {}
        for class_id in unique_classes:
            class_id_int = class_id.item()
            pseudo_binary = (pred_labels == class_id_int).unsqueeze(1).long()

            prev_masks = torch.zeros_like(pseudo_binary).float()
            low_res_masks = F.interpolate(
                prev_masks,
                size=(self.args.img_size // 4, self.args.img_size // 4, self.args.img_size // 4),
            )

            self.click_points = []
            self.click_labels = []
            
            # ===== CHANGE: Use RL agent with proxy rewards for unlabeled data =====
            # This ensures unlabeled data also benefits from RL-guided point selection
            # instead of naive heuristics (no_gt_naive)
            use_proxy_rl = bool(getattr(self.args, 'rl_proxy_reward_enable', 0))
            if use_proxy_rl and self._rl_learning_enabled():
                final_masks = self._run_rl_prompt_episode_for_unlabeled_class(
                    sam_model,
                    image_embedding,
                    pseudo_binary,
                    low_res_masks,
                    class_id_int,
                    num_clicks,
                    student_logits=student_logits,
                )
            else:
                # Fallback to original no_gt_naive if RL disabled
                random_insert = np.random.randint(2, 9)
                for num_click in range(num_clicks):
                    points_input, labels_input = self.get_points(prev_masks, pseudo_binary, use_gt_prompt=False)
                    if num_click == random_insert or num_click == num_clicks - 1:
                        low_res_masks, prev_masks = self.batch_forward(
                            sam_model,
                            image_embedding,
                            pseudo_binary,
                            low_res_masks,
                            points=None,
                        )
                    else:
                        low_res_masks, prev_masks = self.batch_forward(
                            sam_model,
                            image_embedding,
                            pseudo_binary,
                            low_res_masks,
                            points=[points_input, labels_input],
                        )
                final_masks = prev_masks

            prev_masks_detached = final_masks.detach()
            pseudo_masks_dict[class_id_int] = {
                'mask': prev_masks_detached,
                'confidence': prev_masks_detached,
            }

        return pseudo_masks_dict, torch.tensor(0.0, device=student_logits.device)

    def _build_pseudo_supervision_targets(self, target_template, pseudo_masks_dict):
        pseudo_labels = torch.zeros_like(target_template).long()
        confidence_weight = torch.ones_like(target_template).float()

        threshold = self.args.pseudo_label_confidence_threshold
        for class_id, pseudo_mask_data in pseudo_masks_dict.items():
            if isinstance(pseudo_mask_data, dict):
                pseudo_mask = pseudo_mask_data['mask']
                confidence = pseudo_mask_data['confidence']
            else:
                pseudo_mask = pseudo_mask_data
                confidence = pseudo_mask_data

            pseudo_mask_detached = pseudo_mask.detach() if isinstance(pseudo_mask, torch.Tensor) else pseudo_mask
            confidence_detached = confidence.detach() if isinstance(confidence, torch.Tensor) else confidence

            mask_binary = pseudo_mask_detached > 0.5
            pseudo_labels[mask_binary] = int(class_id)

            high_confidence_region = confidence_detached > threshold
            low_confidence_region = mask_binary & ~high_confidence_region
            confidence_weight[low_confidence_region] = torch.clamp(
                confidence_detached[low_confidence_region] / (threshold + 1e-8), 0.0, 1.0
            )

        low_confidence_pixels = (confidence_weight < 1.0).sum().float()
        total_pseudo_pixels = (pseudo_labels != 0).sum().float()
        confidence_filtered_ratio = (low_confidence_pixels / (total_pseudo_pixels + 1e-8)).item()
        return pseudo_labels, confidence_weight, confidence_filtered_ratio

    def _compute_weighted_pseudo_loss(self, student_logits, pseudo_labels, confidence_weight):
        pseudo_dice_loss = self.student_seg_loss(student_logits, pseudo_labels)
        pseudo_mask_bool = pseudo_labels != 0
        if pseudo_mask_bool.any():
            avg_confidence_weight = confidence_weight[pseudo_mask_bool].mean()
            pseudo_dice_loss = pseudo_dice_loss * avg_confidence_weight

        if not bool(self.args.student_use_dynamic_class_weight):
            return pseudo_dice_loss

        class_weights = self.compute_dynamic_class_weights_for_student(pseudo_labels)
        pseudo_ce_loss = F.cross_entropy(student_logits, pseudo_labels, weight=class_weights, reduction='none')
        pseudo_ce_loss_weighted = (pseudo_ce_loss * confidence_weight).mean()
        return 0.5 * pseudo_dice_loss + 0.5 * pseudo_ce_loss_weighted

    def compute_student_loss_with_pseudo_labels(self, student_logits, gt3D=None, pseudo_masks_dict=None, is_labeled=True):
        """Strict semi-supervised student supervision.

        - Labeled sample: always use GT supervision; optionally add pseudo supervision with small weight.
        - Unlabeled sample: use pseudo supervision only; never consume GT.
        """
        if is_labeled and gt3D is None:
            raise ValueError('Labeled samples require gt3D for student supervision.')

        if is_labeled:
            gt_loss = self.compute_student_loss(student_logits, gt3D)
            if pseudo_masks_dict is None or float(self.args.student_labeled_pseudo_weight) <= 0.0:
                return {
                    'loss': gt_loss,
                    'gt_loss': gt_loss.item(),
                    'pseudo_loss': 0.0,
                    'confidence_filtered_ratio': 0.0,
                }

            pseudo_labels, confidence_weight, confidence_filtered_ratio = self._build_pseudo_supervision_targets(
                gt3D,
                pseudo_masks_dict,
            )
            pseudo_loss = self._compute_weighted_pseudo_loss(student_logits, pseudo_labels, confidence_weight)
            total_loss = gt_loss + float(self.args.student_labeled_pseudo_weight) * pseudo_loss
            return {
                'loss': total_loss,
                'gt_loss': gt_loss.item(),
                'pseudo_loss': pseudo_loss.item(),
                'confidence_filtered_ratio': confidence_filtered_ratio,
            }

        if pseudo_masks_dict is None or len(pseudo_masks_dict) == 0:
            zero_loss = student_logits.sum() * 0.0
            return {
                'loss': zero_loss,
                'gt_loss': 0.0,
                'pseudo_loss': 0.0,
                'confidence_filtered_ratio': 0.0,
            }

        pseudo_template = torch.zeros(
            (student_logits.shape[0], 1, student_logits.shape[2], student_logits.shape[3], student_logits.shape[4]),
            device=student_logits.device,
            dtype=torch.long,
        )
        pseudo_labels, confidence_weight, confidence_filtered_ratio = self._build_pseudo_supervision_targets(
            pseudo_template,
            pseudo_masks_dict,
        )
        pseudo_loss = self._compute_weighted_pseudo_loss(student_logits, pseudo_labels, confidence_weight)
        return {
            'loss': pseudo_loss,
            'gt_loss': 0.0,
            'pseudo_loss': pseudo_loss.item(),
            'confidence_filtered_ratio': confidence_filtered_ratio,
        }

    def compute_dice_per_class_from_dict(self, masks_dict, gt3D):
        dice_dict = {}
        valid_dices = []

        for class_id, pred_mask in masks_dict.items():
            class_id_int = class_id if isinstance(class_id, int) else class_id.item()
            gt_binary = (gt3D == class_id_int).float()
            pred_binary = (pred_mask > 0.5).float()

            inter = (gt_binary * pred_binary).sum()
            union = gt_binary.sum() + pred_binary.sum()

            if union == 0:
                dice_class = float('nan')
            else:
                dice_class = (2 * inter / union).item()

            dice_dict[f'class_{class_id_int}'] = dice_class
            if not np.isnan(dice_class):
                valid_dices.append(dice_class)

        dice_dict['avg'] = sum(valid_dices) / len(valid_dices) if valid_dices else 0.0
        return dice_dict

    def compute_pseudo_label_quality(self, pseudo_masks_dict, gt3D):
        """
        Compute Dice between pseudo labels and ground truth to assess label quality.
        
        Stage-2A: This metric monitors how well SAM3D-generated pseudo labels align with GT,
        serving as a proxy for pseudo label reliability.
        
        Args:
            pseudo_masks_dict: Dict of {class_id: pseudo_mask_tensor}
            gt3D: Ground truth labels
            
        Returns:
            quality_dict: Dict of {f'class_{id}': dice_value, 'avg': avg_dice}
        """
        quality_dict = {}
        valid_dices = []

        for class_id, pseudo_mask_item in pseudo_masks_dict.items():
            class_id_int = class_id if isinstance(class_id, int) else class_id.item()
            gt_binary = (gt3D == class_id_int).float()
            
            # Handle nested dict structure: {'mask': tensor, 'confidence': tensor}
            if isinstance(pseudo_mask_item, dict):
                pseudo_mask = pseudo_mask_item['mask']
            else:
                pseudo_mask = pseudo_mask_item
            
            pseudo_binary = (pseudo_mask > 0.5).float()

            inter = (gt_binary * pseudo_binary).sum()
            union = gt_binary.sum() + pseudo_binary.sum()

            if union == 0:
                dice_quality = float('nan')
            else:
                dice_quality = (2 * inter / union).item()

            quality_dict[f'class_{class_id_int}'] = dice_quality
            if not np.isnan(dice_quality):
                valid_dices.append(dice_quality)

        quality_dict['avg'] = sum(valid_dices) / len(valid_dices) if valid_dices else 0.0
        return quality_dict

    def compute_student_dice_per_class_from_logits(self, student_logits, gt3D):
        if gt3D.ndim == 5 and gt3D.shape[1] == 1:
            gt_labels = gt3D[:, 0].long()
        else:
            gt_labels = gt3D.long()

        pred_labels = torch.argmax(student_logits, dim=1)

        dice_dict = {}
        valid_dices = []
        for class_id in range(1, self.args.student_num_classes):
            gt_binary = (gt_labels == class_id).float()
            pred_binary = (pred_labels == class_id).float()

            inter = (gt_binary * pred_binary).sum()
            union = gt_binary.sum() + pred_binary.sum()

            if union == 0:
                # Empty GT and empty prediction are treated as perfect match.
                dice_class = 1.0
            else:
                dice_class = (2 * inter / union).item()

            class_name = f'class_{class_id}'
            dice_dict[class_name] = dice_class
            if not np.isnan(dice_class):
                valid_dices.append(dice_class)

        dice_dict['avg'] = sum(valid_dices) / len(valid_dices) if valid_dices else 0.0
        return dice_dict

    @staticmethod
    def _masked_mean(values, mask):
        if mask.dtype != torch.bool:
            mask = mask > 0
        valid = mask.sum().item()
        if valid == 0:
            return 0.0
        return values[mask].mean().item()

    def _compute_stage3_difficulty_scores(self, student_logits):
        probs = F.softmax(student_logits, dim=1)
        pred_labels = torch.argmax(probs, dim=1)

        eps = 1e-8
        entropy_map = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1)
        entropy_map = entropy_map / np.log(float(probs.shape[1]))

        top2 = torch.topk(probs, k=2, dim=1).values
        margin_uncertainty = 1.0 - (top2[:, 0] - top2[:, 1]).clamp(min=0.0, max=1.0)

        band_width = int(self.args.stage3_boundary_band_width)
        kernel_size = 2 * band_width + 1

        scores = {}
        for class_id in range(1, self.args.student_num_classes):
            class_mask = (pred_labels == class_id).unsqueeze(1)
            class_prob = probs[:, class_id]

            region_mask = class_mask.squeeze(1)
            if region_mask.sum().item() == 0:
                region_mask = class_prob > 0.2

            class_mask_f = class_mask.float()
            dilated = F.max_pool3d(class_mask_f, kernel_size=kernel_size, stride=1, padding=band_width)
            eroded = -F.max_pool3d(-class_mask_f, kernel_size=kernel_size, stride=1, padding=band_width)
            boundary_band = (dilated - eroded).squeeze(1) > 0.5

            h_score = self._masked_mean(entropy_map, region_mask)
            m_score = self._masked_mean(margin_uncertainty, region_mask)
            b_score = self._masked_mean(margin_uncertainty, boundary_band)

            if self.args.stage3_uncertainty_type == 'entropy':
                difficulty = h_score
            elif self.args.stage3_uncertainty_type == 'margin':
                difficulty = m_score
            else:
                a = float(self.args.stage3_uncertainty_alpha)
                b = float(self.args.stage3_uncertainty_beta)
                g = float(self.args.stage3_uncertainty_gamma)
                denom = max(a + b + g, eps)
                difficulty = (a * h_score + b * m_score + g * b_score) / denom

            scores[class_id] = float(difficulty)

        return scores

    def _select_stage3_routed_classes(self, difficulty_scores):
        min_trigger = float(self.args.stage3_route_min_trigger)
        topk = int(self.args.stage3_route_topk)

        candidates = []
        for class_id, score in difficulty_scores.items():
            if score >= min_trigger:
                candidates.append((class_id, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [class_id for class_id, _ in candidates[:topk]], candidates

    def compute_dynamic_class_weights_for_student(self, gt3D):
        if gt3D.ndim == 5 and gt3D.shape[1] == 1:
            gt_labels = gt3D[:, 0].long()
        else:
            gt_labels = gt3D.long()

        num_classes = self.args.student_num_classes
        class_counts = torch.bincount(gt_labels.reshape(-1), minlength=num_classes).float()

        weights = torch.zeros(num_classes, device=gt_labels.device, dtype=torch.float32)
        include_bg = bool(self.args.student_class_weight_include_background)
        start_cls = 0 if include_bg else 1
        valid_mask = torch.zeros(num_classes, device=gt_labels.device, dtype=torch.bool)

        total_considered = class_counts[start_cls:].sum()
        if total_considered <= 0:
            weights[:] = 1.0
            return weights

        for cls_idx in range(start_cls, num_classes):
            if class_counts[cls_idx] > 0:
                pixel_ratio = class_counts[cls_idx] / (total_considered + 1e-8)
                weights[cls_idx] = 1.0 / (pixel_ratio + self.args.class_weight_smooth)
                valid_mask[cls_idx] = True

        if valid_mask.any():
            avg_weight = weights[valid_mask].mean()
            weights[valid_mask] = weights[valid_mask] / (avg_weight + 1e-8)

        if not include_bg:
            weights[0] = 1.0

        weights[~valid_mask] = 1.0
        return weights

    def compute_student_loss(self, student_logits, gt3D):
        if gt3D.ndim == 5 and gt3D.shape[1] == 1:
            gt_labels = gt3D[:, 0].long()
        else:
            gt_labels = gt3D.long()

        dice_loss_part = self.student_seg_loss(student_logits, gt3D)
        if not bool(self.args.student_use_dynamic_class_weight):
            return dice_loss_part

        class_weights = self.compute_dynamic_class_weights_for_student(gt3D)
        ce_loss_part = F.cross_entropy(student_logits, gt_labels, weight=class_weights)
        return 0.5 * dice_loss_part + 0.5 * ce_loss_part

    def train_epoch(self, epoch, num_clicks):
        def _optimizer_has_grads(optimizer):
            for group in optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        return True
            return False

        epoch_loss = 0
        epoch_student_loss = 0
        epoch_iou = 0
        self.model.train()
        self.student_model.train()

        if self.args.multi_gpu:
            sam_model = self.model.module
            student_model = self.student_model.module
        else:
            sam_model = self.model
            student_model = self.student_model
            self.args.rank = -1

        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            tbar = tqdm(self.dataloaders)
        else:
            tbar = self.dataloaders

        self.optimizer.zero_grad()
        self.student_optimizer.zero_grad()
        step_loss = 0
        epoch_dice = 0
        epoch_student_dice = 0
        epoch_dice_dict = {}
        epoch_student_dice_dict = {}
        self.rl_prompt_step_logs = []
        self.rl_epoch_stats = {
            'episodes': 0,
            'returns': [],
            'rewards': [],
            'deltas': [],
            'invalid': 0,
            'steps': 0,
            'learn_steps': 0,
            'learn_losses': [],
            'learn_kl_losses': [],
        }
        # Stage-2A: Track pseudo label losses
        epoch_pseudo_label_gt_loss = 0.0
        epoch_pseudo_label_pseudo_loss = 0.0
        # Stage-2B: Track semi-supervised learning statistics
        epoch_labeled_count = 0
        epoch_unlabeled_count = 0
        # Confidence filtering: Track average confidence filtering ratio
        epoch_confidence_filtered_ratio = 0.0
        sam_dice_steps = 0
        semi_ratio = float(self.args.semi_supervised_labeled_ratio)
        semi_enabled = 0.0 < semi_ratio < 1.0
        labeled_pseudo_enabled = float(getattr(self.args, 'student_labeled_pseudo_weight', 0.0)) > 0.0

        for step, data3D in enumerate(tbar):
            try:
                image3D, gt3D = data3D['image'], data3D['label']
                # Stage-2B: Check if sample is labeled or unlabeled
                is_labeled = data3D.get('is_labeled', True)  # Default to True for backward compatibility
            except Exception as e:
                print(f'Error processing batch at step {step}: {e}')
                continue

            is_labeled_bool = self._as_bool_label_status(is_labeled)

            my_context = self.model.no_sync if self.args.rank != -1 and (step + 1) % self.args.accumulation_steps != 0 else nullcontext

            with my_context():
                image3D = self.norm_transform(image3D.squeeze(dim=1))
                image3D = image3D.unsqueeze(dim=1)

                image3D = image3D.to(self.device)
                gt3D = gt3D.to(self.device).type(torch.long)

                with torch.amp.autocast('cuda'):
                    
                    _, student_logits = student_model(image3D, return_logits=True)
                    self.click_points = []
                    self.click_labels = []
                    pred_list = []

                    # Stage-2B: For unlabeled data, either skip SAM or generate pseudo labels
                    if semi_enabled and not is_labeled_bool:
                        # Unlabeled data: skip SAM supervision, only use pseudo labels for student
                        sam_loss = torch.tensor(0.0, device=self.device)
                        masks_dict = {}

                        # Generate pseudo labels for unlabeled data without GT guidance
                        # Use torch.no_grad() to completely prevent gradient flow from student back to SAM
                        # SAM forward does NOT contribute to loss, only generates teaching signal for student
                        with torch.no_grad():
                            image_embedding = sam_model.image_encoder(image3D)
                            pseudo_masks_dict, _ = self.generate_pseudo_labels_without_gt(
                                sam_model,
                                image_embedding,
                                student_logits,
                                num_clicks=self.args.pseudo_label_clicks,
                            )
                        
                        epoch_unlabeled_count += 1
                    else:
                        image_embedding = sam_model.image_encoder(image3D)
                        # Labeled data: normal processing with SAM supervision
                        masks_dict, sam_loss = self.interaction(sam_model, image_embedding, gt3D, num_clicks=11)
                        
                        # Generate pseudo labels for labeled data when requested.
                        pseudo_masks_dict = None
                        pseudo_label_quality_dict = None
                        need_labeled_pseudo = labeled_pseudo_enabled
                        if need_labeled_pseudo:
                            with torch.no_grad():
                                pseudo_masks_dict, _ = self.generate_pseudo_labels_for_sample(
                                    sam_model, image_embedding, gt3D, 
                                    num_clicks=self.args.pseudo_label_clicks
                                )
                            pseudo_label_quality_dict = self.compute_pseudo_label_quality(pseudo_masks_dict, gt3D)
                            self.pseudo_label_epoch_dice.append(pseudo_label_quality_dict.get('avg', 0.0))

                        epoch_labeled_count += 1

                    
                    # Compute student loss with potential pseudo label supervision
                    loss_dict = self.compute_student_loss_with_pseudo_labels(
                        student_logits,
                        gt3D=gt3D if is_labeled_bool else None,
                        pseudo_masks_dict=pseudo_masks_dict,
                        is_labeled=is_labeled_bool,
                    )
                    student_loss = loss_dict['loss']
                    
                    # Stage-2B: Apply loss weight for unlabeled data
                    if semi_enabled and not is_labeled_bool:
                        student_loss = student_loss * self.args.semi_supervised_unlabeled_loss_weight

                    total_loss = sam_loss + self.args.student_loss_weight * student_loss

                epoch_loss += sam_loss.item()
                epoch_student_loss += student_loss.item()
                
                # Stage-2A: Track pseudo label loss components
                if labeled_pseudo_enabled:
                    epoch_pseudo_label_gt_loss += loss_dict.get('gt_loss', 0.0)
                    epoch_pseudo_label_pseudo_loss += loss_dict.get('pseudo_loss', 0.0)
                
                # Confidence filtering: Track average ratio of filtered pseudo labels
                epoch_confidence_filtered_ratio += loss_dict.get('confidence_filtered_ratio', 0.0)

                # Stage-2B: Only accumulate SAM Dice and Student Dice for labeled data
                # Both metrics should be computed on labeled data for fair comparison
                if semi_enabled and not is_labeled_bool:
                    # For unlabeled data: skip both SAM and student Dice accumulation
                    pass
                else:
                    # For labeled data: accumulate SAM Dice
                    dice_dict = self.compute_dice_per_class_from_dict(masks_dict, gt3D)
                    epoch_dice += dice_dict['avg']
                    sam_dice_steps += 1
                    for class_name, dice_val in dice_dict.items():
                        if class_name != 'avg' and not np.isnan(dice_val):
                            epoch_dice_dict.setdefault(class_name, []).append(dice_val)

                    # For labeled data: also accumulate Student Dice (only on labeled data for comparison)
                    student_dice_dict = self.compute_student_dice_per_class_from_logits(student_logits, gt3D)
                    epoch_student_dice += student_dice_dict['avg']
                    for class_name, dice_val in student_dice_dict.items():
                        if class_name != 'avg' and not np.isnan(dice_val):
                            epoch_student_dice_dict.setdefault(class_name, []).append(dice_val)

                cur_loss = total_loss.item()
                total_loss /= self.args.accumulation_steps
                self.scaler.scale(total_loss).backward()

            if (step + 1) % self.args.accumulation_steps == 0 or (step == len(self.dataloaders) - 1):
                stepped = False
                if _optimizer_has_grads(self.optimizer):
                    self.scaler.step(self.optimizer)
                    stepped = True
                if _optimizer_has_grads(self.student_optimizer):
                    self.scaler.step(self.student_optimizer)
                    stepped = True
                if stepped:
                    self.scaler.update()
                self.optimizer.zero_grad()
                self.student_optimizer.zero_grad()
                step_loss = 0
            else:
                step_loss += cur_loss

            # Run RL optimization in an isolated phase after main forward/backward of this step.
            if self._rl_learning_enabled() and self.rl_pending_updates > 0:
                updates_to_run = int(self.rl_pending_updates)
                self.rl_pending_updates = 0
                for _ in range(updates_to_run):
                    rl_loss = self._maybe_optimize_rl_agent()
                    if rl_loss is not None:
                        self.rl_epoch_stats['learn_steps'] += 1
                        self.rl_epoch_stats['learn_losses'].append(rl_loss)

        epoch_loss /= step + 1
        epoch_student_loss /= step + 1
        epoch_dice = epoch_dice / sam_dice_steps if sam_dice_steps > 0 else 0.0
        epoch_student_dice /= step + 1
        
        # Stage-2A: Average pseudo label losses
        if labeled_pseudo_enabled and step > 0:
            epoch_pseudo_label_gt_loss /= (step + 1)
            epoch_pseudo_label_pseudo_loss /= (step + 1)
        
        # Confidence filtering: Average the filtering ratio
        if step > 0:
            epoch_confidence_filtered_ratio /= (step + 1)

        if self._is_rl_prompt_enabled() and self.rl_prompt_step_logs:
            source_counter = {}
            for item in self.rl_prompt_step_logs:
                src = item['source']
                source_counter[src] = source_counter.get(src, 0) + 1
            src_summary = ', '.join([f'{k}:{v}' for k, v in sorted(source_counter.items())])
            self.logger.info(
                f'[RL-PROMPT-SCAFFOLD] steps={len(self.rl_prompt_step_logs)}, source_breakdown={src_summary}'
            )

        if self._rl_learning_enabled() and self.rl_epoch_stats['steps'] > 0:
            avg_return = float(np.mean(self.rl_epoch_stats['returns'])) if self.rl_epoch_stats['returns'] else 0.0
            avg_reward = float(np.mean(self.rl_epoch_stats['rewards'])) if self.rl_epoch_stats['rewards'] else 0.0
            avg_delta = float(np.mean(self.rl_epoch_stats['deltas'])) if self.rl_epoch_stats['deltas'] else 0.0
            invalid_ratio = float(self.rl_epoch_stats['invalid']) / max(float(self.rl_epoch_stats['steps']), 1.0)
            avg_steps_per_episode = float(self.rl_epoch_stats['steps']) / max(float(self.rl_epoch_stats['episodes']), 1.0)
            avg_rl_loss = float(np.mean(self.rl_epoch_stats['learn_losses'])) if self.rl_epoch_stats['learn_losses'] else 0.0
            avg_rl_kl = float(np.mean(self.rl_epoch_stats['learn_kl_losses'])) if self.rl_epoch_stats['learn_kl_losses'] else 0.0
            
            # Log comprehensive RL learning statistics
            self.logger.info(
                '[RL-PROMPT-LEARN] '
                f'episodes={self.rl_epoch_stats["episodes"]}, '
                f'steps={self.rl_epoch_stats["steps"]}, '
                f'avg_steps_per_ep={avg_steps_per_episode:.2f}, '
                f'avg_return={avg_return:.4f}, '
                f'avg_reward={avg_reward:.4f}, '
                f'avg_delta_dice={avg_delta:.4f}, '
                f'invalid_ratio={invalid_ratio:.4f}, '
                f'epsilon={self.rl_epsilon:.4f}, '
                f'learn_steps={self.rl_epoch_stats["learn_steps"]}, '
                f'avg_rl_loss={avg_rl_loss:.6f}, '
                f'avg_rl_kl={avg_rl_kl:.6f}'
            )

        for class_name in epoch_dice_dict:
            valid_dices = [d for d in epoch_dice_dict[class_name] if not np.isnan(d)]
            epoch_dice_dict[class_name] = sum(valid_dices) / len(valid_dices) if valid_dices else 0.0

        for class_name in epoch_student_dice_dict:
            valid_dices = [d for d in epoch_student_dice_dict[class_name] if not np.isnan(d)]
            epoch_student_dice_dict[class_name] = sum(valid_dices) / len(valid_dices) if valid_dices else 0.0

        return (
            epoch_loss,
            epoch_student_loss,
            epoch_iou,
            epoch_dice,
            epoch_student_dice,
            pred_list,
            epoch_dice_dict,
            epoch_student_dice_dict,
            epoch_pseudo_label_gt_loss,
            epoch_pseudo_label_pseudo_loss,
            epoch_labeled_count,
            epoch_unlabeled_count,
            epoch_confidence_filtered_ratio,
        )

    def eval_epoch(self, epoch, num_clicks):
        if self.val_dataloaders is None:
            return 0, {}, 0, {}

        self.model.eval()
        self.student_model.eval()
        val_dice_list = []
        val_student_dice_list = []
        val_dice_per_class_all = {}
        val_student_dice_per_class_all = {}
        stage3_route_counts = []

        if self.args.multi_gpu:
            sam_model = self.model.module
            student_model = self.student_model.module
        else:
            sam_model = self.model
            student_model = self.student_model

        refine_enabled = bool(getattr(self.args, 'infer_enable_teacher_refinement', 0))

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(self.val_dataloaders)):
                try:
                    image3D = batch_data['image']
                except Exception as e:
                    print(f'Error in validation batch {batch_idx}: {e}')
                    continue

                if 'label' in batch_data:
                    gt3D = batch_data['label']
                else:
                    continue

                image3D = self.norm_transform(image3D.squeeze(dim=1))
                image3D = image3D.unsqueeze(dim=1)
                image3D = image3D.to(self.device)
                gt3D = gt3D.to(self.device).type(torch.long)

                with torch.amp.autocast('cuda'):
                    _, student_logits = student_model(image3D, return_logits=True)
                    student_probs = F.softmax(student_logits, dim=1)
                    student_pred_labels = torch.argmax(student_probs, dim=1)
                    student_dice_dict = self.compute_student_dice_per_class_from_logits(student_logits, gt3D)
                    val_student_dice_list.append(student_dice_dict['avg'])
                    for class_name, dice_val in student_dice_dict.items():
                        if class_name != 'avg' and not np.isnan(dice_val):
                            val_student_dice_per_class_all.setdefault(class_name, []).append(dice_val)

                    if not refine_enabled:
                        val_dice_list.append(student_dice_dict['avg'])
                        for class_name, dice_val in student_dice_dict.items():
                            if class_name != 'avg' and not np.isnan(dice_val):
                                val_dice_per_class_all.setdefault(class_name, []).append(dice_val)
                        continue

                    image_embedding = sam_model.image_encoder(image3D)

                    stage3_enabled = bool(getattr(self.args, 'stage3_enable_difficulty_routing', 0))
                    masks_dict = {}
                    for class_id in range(1, self.args.student_num_classes):
                        masks_dict[class_id] = (student_pred_labels == class_id).unsqueeze(1).float()

                    if stage3_enabled:
                        difficulty_scores = self._compute_stage3_difficulty_scores(student_logits)
                        routed_class_ids, _ = self._select_stage3_routed_classes(difficulty_scores)
                    else:
                        routed_class_ids = list(range(1, self.args.student_num_classes))

                    stage3_route_counts.append(len(routed_class_ids))

                    for class_id_int in routed_class_ids:
                        pseudo_binary = (student_pred_labels == class_id_int).unsqueeze(1).long()
                        if pseudo_binary.sum().item() == 0:
                            continue

                        prev_masks = torch.zeros_like(pseudo_binary).float()
                        low_res_masks = F.interpolate(
                            prev_masks,
                            size=(self.args.img_size // 4, self.args.img_size // 4, self.args.img_size // 4),
                            mode='trilinear',
                            align_corners=False,
                        )

                        self.click_points = []
                        self.click_labels = []

                        for _ in range(num_clicks):
                            use_gt_prompt = bool(getattr(self.args, 'stage3_infer_use_gt_prompt', 0))
                            points_input, labels_input = self.get_points(
                                prev_masks,
                                pseudo_binary,
                                use_gt_prompt=use_gt_prompt,
                            )
                            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                                points=[points_input, labels_input],
                                boxes=None,
                                masks=low_res_masks,
                            )

                            low_res_masks, _ = sam_model.mask_decoder(
                                image_embeddings=image_embedding.to(self.device),
                                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sparse_embeddings,
                                dense_prompt_embeddings=dense_embeddings,
                                multimask_output=False,
                            )

                            prev_masks = F.interpolate(
                                low_res_masks,
                                size=pseudo_binary.shape[-3:],
                                mode='trilinear',
                                align_corners=False,
                            )

                        masks_dict[class_id_int] = prev_masks

                    batch_dice_dict = self.compute_dice_per_class_from_dict(masks_dict, gt3D)
                    val_dice_list.append(batch_dice_dict['avg'])
                    for class_name, dice_val in batch_dice_dict.items():
                        if class_name != 'avg' and not np.isnan(dice_val):
                            val_dice_per_class_all.setdefault(class_name, []).append(dice_val)

        if bool(getattr(self.args, 'stage3_enable_difficulty_routing', 0)) and stage3_route_counts:
            avg_route = float(sum(stage3_route_counts)) / len(stage3_route_counts)
            route_msg = f'[Stage-3] avg routed classes per val volume: {avg_route:.3f}'
            print(route_msg)
            self.logger.info(route_msg)

        avg_val_dice_dict = {'avg': (sum(val_dice_list) / len(val_dice_list)) if val_dice_list else 0}
        for class_name in val_dice_per_class_all:
            class_dices = val_dice_per_class_all[class_name]
            avg_val_dice_dict[class_name] = sum(class_dices) / len(class_dices) if class_dices else 0.0

        avg_val_student_dice_dict = {'avg': (sum(val_student_dice_list) / len(val_student_dice_list)) if val_student_dice_list else 0}
        for class_name in val_student_dice_per_class_all:
            class_dices = val_student_dice_per_class_all[class_name]
            avg_val_student_dice_dict[class_name] = sum(class_dices) / len(class_dices) if class_dices else 0.0

        self.model.train()
        self.student_model.train()
        return avg_val_dice_dict['avg'], avg_val_dice_dict, avg_val_student_dice_dict['avg'], avg_val_student_dice_dict

    def plot_result(self, plot_data, description, save_name, val_data=None):
        plt.figure(figsize=(10, 6))
        epochs = list(range(len(plot_data)))
        plt.plot(epochs, plot_data, label='Train', linewidth=2, marker='o', markersize=4)

        if val_data is not None and len(val_data) > 0:
            val_epochs = list(range(0, len(plot_data), self.args.val_interval))[:len(val_data)]
            plt.plot(val_epochs, val_data, label='Validation', linewidth=2, marker='s', markersize=4)
            plt.legend(loc='best', fontsize=10)

        plt.title(description, fontsize=12)
        plt.xlabel('Epoch', fontsize=11)
        plt.ylabel(f'{save_name}', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_path, f'{save_name}.png'), dpi=100)
        plt.close()

    def _format_class_metrics(self, metrics_dict):
        parts = []
        for class_name in sorted(metrics_dict.keys()):
            if class_name == 'avg':
                continue
            parts.append(f'{class_name}={metrics_dict[class_name]:.4f}')
        return ', '.join(parts) if parts else 'none'

    def train(self):
        self.scaler = torch.amp.GradScaler('cuda')
        refine_enabled = bool(getattr(self.args, 'infer_enable_teacher_refinement', 0))
        if not refine_enabled:
            primary_metric_name = 'STUDENT_ONLY'
        elif bool(getattr(self.args, 'stage3_enable_difficulty_routing', 0)):
            primary_metric_name = 'HYBRID'
        else:
            primary_metric_name = 'SAM_REFINE'
        labeled_pseudo_enabled = float(getattr(self.args, 'student_labeled_pseudo_weight', 0.0)) > 0.0
        semi_ratio = float(self.args.semi_supervised_labeled_ratio)
        semi_enabled = 0.0 < semi_ratio < 1.0
        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f'Epoch: {epoch}/{self.args.num_epochs - 1}')

            if self.args.multi_gpu:
                dist.barrier()
                self.dataloaders.sampler.set_epoch(epoch)

            num_clicks = np.random.randint(1, 21)
            (
                epoch_loss,
                epoch_student_loss,
                _,
                epoch_dice,
                epoch_student_dice,
                _,
                epoch_dice_dict,
                epoch_student_dice_dict,
                epoch_pseudo_label_gt_loss,
                epoch_pseudo_label_pseudo_loss,
                epoch_labeled_count,
                epoch_unlabeled_count,
                epoch_confidence_filtered_ratio,
            ) = self.train_epoch(epoch, num_clicks)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.student_lr_scheduler is not None:
                self.student_lr_scheduler.step()
            if self.args.multi_gpu:
                dist.barrier()

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                self.losses.append(epoch_loss)
                self.dices.append(epoch_dice)
                self.student_losses.append(epoch_student_loss)
                self.student_dices.append(epoch_student_dice)
                self.epoch_dices_dict_all.append(epoch_dice_dict)
                self.epoch_student_dices_dict_all.append(epoch_student_dice_dict)

                primary_train_metrics = self._format_class_metrics(epoch_dice_dict)
                student_train_metrics = self._format_class_metrics(epoch_student_dice_dict)
                print(
                    f'EPOCH {epoch} TRAIN | '
                    f'{primary_metric_name} avg={epoch_dice:.4f} ({primary_train_metrics}) | '
                    f'STUDENT avg={epoch_student_dice:.4f} ({student_train_metrics})'
                )

                self.logger.info(
                    f'Epoch\t {epoch}\t : loss: {epoch_loss}, train_dice: {epoch_dice}, '
                    f'student_loss: {epoch_student_loss}, student_dice: {epoch_student_dice}'
                )
                # Stage-2A: Log pseudo label loss details if enabled
                if labeled_pseudo_enabled:
                    self.logger.info(
                        '  [Pseudo Supervision] Labeled samples use GT + weighted pseudo, '
                        f'GT_loss: {epoch_pseudo_label_gt_loss:.6f}, '
                        f'Pseudo_loss: {epoch_pseudo_label_pseudo_loss:.6f}, '
                        f'Confidence_filtered_ratio: {epoch_confidence_filtered_ratio:.3f}'
                    )
                # Stage-2B: Log semi-supervised learning statistics if enabled
                if semi_enabled:
                    self.logger.info(
                        f'  [Stage-2B] Semi-Supervised: Labeled={epoch_labeled_count}, '
                        f'Unlabeled={epoch_unlabeled_count}'
                    )
                for class_name, dice_val in epoch_dice_dict.items():
                    self.logger.info(f'  {class_name}: {dice_val}')
                for class_name, dice_val in epoch_student_dice_dict.items():
                    self.logger.info(f'  student_{class_name}: {dice_val}')

                if self.args.multi_gpu:
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()

                self.save_checkpoint(epoch, state_dict, describe='latest')

                if epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                    self.save_checkpoint(epoch, state_dict, describe='loss_best')

                if epoch_dice > self.best_dice:
                    self.best_dice = epoch_dice
                    self.best_train_dice_epoch = epoch
                    self.save_checkpoint(epoch, state_dict, describe='dice_best')

                if epoch_student_dice > self.best_student_dice:
                    self.best_student_dice = epoch_student_dice
                    self.best_student_train_dice_epoch = epoch
                    self.save_checkpoint(epoch, state_dict, describe='student_dice_best')

                if epoch % self.args.val_interval == 0 and self.val_dataloaders is not None:
                    val_dice, val_dice_dict, val_student_dice, val_student_dice_dict = self.eval_epoch(
                        epoch, self.args.eval_num_clicks
                    )
                    self.val_dices.append(val_dice)
                    self.val_student_dices.append(val_student_dice)
                    self.val_epochs.append(epoch)
                    self.val_epoch_dices_dict_all.append(val_dice_dict)
                    self.val_epoch_student_dices_dict_all.append(val_student_dice_dict)

                    primary_val_metrics = self._format_class_metrics(val_dice_dict)
                    student_val_metrics = self._format_class_metrics(val_student_dice_dict)
                    print(
                        f'EPOCH {epoch} VAL   | '
                        f'{primary_metric_name} avg={val_dice:.4f} ({primary_val_metrics}) | '
                        f'STUDENT avg={val_student_dice:.4f} ({student_val_metrics})'
                    )

                    self.logger.info(
                        f'Epoch\t {epoch}\t : val_dice: {val_dice}, val_student_dice: {val_student_dice}'
                    )
                    for class_name, dice_val in val_dice_dict.items():
                        if class_name != 'avg':
                            self.logger.info(f'  {class_name}: {dice_val}')
                    for class_name, dice_val in val_student_dice_dict.items():
                        if class_name != 'avg':
                            self.logger.info(f'  student_{class_name}: {dice_val}')

                    if val_dice > self.best_val_dice:
                        self.best_val_dice = val_dice
                        self.best_val_dice_epoch = epoch
                        self.best_val_dice_dict = dict(val_dice_dict)
                        self.save_checkpoint(epoch, state_dict, describe='val_dice_best')

                    if val_student_dice > self.best_val_student_dice:
                        self.best_val_student_dice = val_student_dice
                        self.best_val_student_dice_epoch = epoch
                        self.best_val_student_dice_dict = dict(val_student_dice_dict)
                        self.save_checkpoint(epoch, state_dict, describe='val_student_dice_best')

                self.plot_result(self.losses, 'Dice + Cross Entropy Loss', 'Loss')
                self.plot_result(self.dices, 'Dice Score', 'Dice', val_data=self.val_dices if self.val_dices else None)
                self.plot_result(self.student_losses, 'Student Dice + Cross Entropy Loss', 'Student_Loss')
                self.plot_result(
                    self.student_dices,
                    'Student Dice Score',
                    'Student_Dice',
                    val_data=self.val_student_dices if self.val_student_dices else None,
                )

        self.logger.info('=====================================================================')
        self.logger.info(f'Best loss: {self.best_loss}')
        self.logger.info(f'Best train dice: {self.best_dice}')
        self.logger.info(f'Best val dice: {self.best_val_dice}')
        self.logger.info(f'Best student train dice: {self.best_student_dice}')
        self.logger.info(f'Best student val dice: {self.best_val_student_dice}')
        self.logger.info(f'Total loss: {self.losses}')
        self.logger.info(f'Total train dice: {self.dices}')
        self.logger.info(f'Total val dice: {self.val_dices}')
        self.logger.info(f'Total student loss: {self.student_losses}')
        self.logger.info(f'Total student train dice: {self.student_dices}')
        self.logger.info(f'Total student val dice: {self.val_student_dices}')
        self.logger.info('=====================================================================')

        self.logger.info('=====================================================================')
        self.logger.info(f'Best {primary_metric_name} Summary:')
        self.logger.info(f'Best {primary_metric_name} train dice epoch: {self.best_train_dice_epoch}')
        self.logger.info(f'Best {primary_metric_name} train dice: {self.best_dice:.4f}')
        self.logger.info(f'Best {primary_metric_name} val dice epoch: {self.best_val_dice_epoch}')
        self.logger.info(f'Best {primary_metric_name} val dice: {self.best_val_dice:.4f}')
        if self.best_val_dice_dict:
            self.logger.info(f'Best {primary_metric_name} val per-class metrics:')
            for class_name, dice_val in self.best_val_dice_dict.items():
                self.logger.info(f'  {class_name}: {dice_val:.4f}')
        self.logger.info('=====================================================================')

        self.logger.info('=====================================================================')
        self.logger.info('Best Student Summary:')
        self.logger.info(f'Best Student train dice epoch: {self.best_student_train_dice_epoch}')
        self.logger.info(f'Best Student train dice: {self.best_student_dice:.4f}')
        self.logger.info(f'Best Student val dice epoch: {self.best_val_student_dice_epoch}')
        self.logger.info(f'Best Student val dice: {self.best_val_student_dice:.4f}')
        if self.best_val_student_dice_dict:
            self.logger.info('Best Student val per-class metrics:')
            for class_name, dice_val in self.best_val_student_dice_dict.items():
                self.logger.info(f'  {class_name}: {dice_val:.4f}')
        self.logger.info('=====================================================================')

        print('=====================================================================')
        print(f'Best {primary_metric_name} Summary:')
        print(f'  Best {primary_metric_name} train dice epoch: {self.best_train_dice_epoch}')
        print(f'  Best {primary_metric_name} train dice: {self.best_dice:.4f}')
        print(f'  Best {primary_metric_name} val dice epoch: {self.best_val_dice_epoch}')
        print(f'  Best {primary_metric_name} val dice: {self.best_val_dice:.4f}')
        if self.best_val_dice_dict:
            print(f'  Best {primary_metric_name} val per-class metrics:')
            for class_name, dice_val in self.best_val_dice_dict.items():
                print(f'    {class_name}: {dice_val:.4f}')
        print('=====================================================================')

        print('=====================================================================')
        print('Best Student Summary:')
        print(f'  Best Student train dice epoch: {self.best_student_train_dice_epoch}')
        print(f'  Best Student train dice: {self.best_student_dice:.4f}')
        print(f'  Best Student val dice epoch: {self.best_val_student_dice_epoch}')
        print(f'  Best Student val dice: {self.best_val_student_dice:.4f}')
        if self.best_val_student_dice_dict:
            print('  Best Student val per-class metrics:')
            for class_name, dice_val in self.best_val_student_dice_dict.items():
                print(f'    {class_name}: {dice_val:.4f}')
        print('=====================================================================')

        self.logger.info('=====================================================================')
        self.logger.info(f'args : {self.args}')
        self.logger.info(f'Used datasets : {self.img_datas}')
        self.logger.info('=====================================================================')
