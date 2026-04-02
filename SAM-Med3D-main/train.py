import warnings
warnings.filterwarnings("ignore")

import argparse
import datetime
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
import torch.multiprocessing as mp

from train_defaults import apply_fixed_runtime_defaults
from train_core import BaseTrainer, build_sam_model, build_student_model, get_dataloaders
from utils.click_method import get_next_click3D_torch_2, get_next_click3D_torch_no_gt_naive
from utils.data_paths import img_datas
from utils.runtime_helpers import cleanup_distributed, device_config, init_seeds, setup_distributed

join = os.path.join

parser = argparse.ArgumentParser()
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_epochs', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=1)

parser.add_argument('--val_split', type=float, default=0.3, help='Validation split ratio (0-1)')
parser.add_argument('--semi_supervised_labeled_ratio', type=float, default=0.5,
                    help='Semi-supervised switch by ratio: 0 means full-supervised mode, (0,1) means labeled/unlabeled split.')

args = parser.parse_args()
args = apply_fixed_runtime_defaults(args)

logger = logging.getLogger(__name__)
LOG_OUT_DIR = join(args.work_dir, args.task_name)
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
PSEUDO_LABEL_DIR = join(args.work_dir, args.task_name, args.pseudo_label_save_dir)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(PSEUDO_LABEL_DIR, exist_ok=True)

click_methods = {
    'random': get_next_click3D_torch_2,
    'no_gt_naive': get_next_click3D_torch_no_gt_naive,
}


def _validate_args(runtime_args):
    if runtime_args.student_labeled_pseudo_weight < 0:
        raise ValueError('--student_labeled_pseudo_weight must be >= 0.')
    if not (0.0 <= runtime_args.semi_supervised_labeled_ratio <= 1.0):
        raise ValueError('--semi_supervised_labeled_ratio must be in [0, 1].')
    if int(runtime_args.infer_enable_teacher_refinement) not in (0, 1):
        raise ValueError('infer_enable_teacher_refinement must be 0 or 1.')

    if runtime_args.stage3_route_topk < 0:
        raise ValueError('--stage3_route_topk must be >= 0.')
    if not (0.0 <= runtime_args.stage3_route_min_trigger <= 1.0):
        raise ValueError('--stage3_route_min_trigger must be in [0, 1].')
    if runtime_args.stage3_boundary_band_width < 1:
        raise ValueError('--stage3_boundary_band_width must be >= 1.')

    if runtime_args.rl_prompt_max_steps < 1:
        raise ValueError('--rl_prompt_max_steps must be >= 1.')
    if runtime_args.rl_prompt_log_limit < 1:
        raise ValueError('--rl_prompt_log_limit must be >= 1.')
    if runtime_args.rl_prompt_step_size < 1:
        raise ValueError('--rl_prompt_step_size must be >= 1.')
    if runtime_args.rl_prompt_state_dim < 8:
        raise ValueError('--rl_prompt_state_dim must be >= 8 for current state encoder.')
    if runtime_args.rl_prompt_hidden_dim < 16:
        raise ValueError('--rl_prompt_hidden_dim must be >= 16.')
    if runtime_args.rl_dqn_lr <= 0:
        raise ValueError('--rl_dqn_lr must be > 0.')
    if not (0.0 < runtime_args.rl_dqn_gamma <= 1.0):
        raise ValueError('--rl_dqn_gamma must be in (0, 1].')
    if not (0.0 <= runtime_args.rl_dqn_epsilon_end <= runtime_args.rl_dqn_epsilon_start <= 1.0):
        raise ValueError('--rl_dqn_epsilon_start/end must satisfy 0 <= end <= start <= 1.')
    if not (0.0 < runtime_args.rl_dqn_epsilon_decay <= 1.0):
        raise ValueError('--rl_dqn_epsilon_decay must be in (0, 1].')
    if runtime_args.rl_dqn_buffer_size < 100:
        raise ValueError('--rl_dqn_buffer_size must be >= 100.')
    if runtime_args.rl_dqn_batch_size < 4:
        raise ValueError('--rl_dqn_batch_size must be >= 4.')
    if runtime_args.rl_dqn_learn_every < 1:
        raise ValueError('--rl_dqn_learn_every must be >= 1.')
    if runtime_args.rl_dqn_target_update_every < 1:
        raise ValueError('--rl_dqn_target_update_every must be >= 1.')
    if runtime_args.rl_reward_clip_abs <= 0:
        raise ValueError('--rl_reward_clip_abs must be > 0.')
    if runtime_args.rl_reward_composition not in ['simple', 'extended']:
        raise ValueError('--rl_reward_composition must be "simple" or "extended".')
    if runtime_args.rl_reward_alpha < 0 or runtime_args.rl_reward_beta < 0 or runtime_args.rl_reward_gamma < 0:
        raise ValueError('Reward weight coefficients (alpha, beta, gamma) must be >= 0.')
    if runtime_args.rl_init_strategy not in ['random', 'mixed', 'guided']:
        raise ValueError('--rl_init_strategy must be "random", "mixed", or "guided".')
    if not (0.0 <= runtime_args.rl_init_reliable_ratio <= 1.0):
        raise ValueError('--rl_init_reliable_ratio must be in [0, 1].')
    if not (0.0 < runtime_args.rl_proxy_reward_weight <= 1.0):
        raise ValueError('--rl_proxy_reward_weight must be in (0, 1].')
    if runtime_args.rl_proxy_consistency_weight < 0 or runtime_args.rl_proxy_confidence_weight < 0 or runtime_args.rl_proxy_entropy_weight < 0:
        raise ValueError('Proxy reward sub-weights (consistency, confidence, entropy) must be >= 0.')
    if bool(runtime_args.rl_prompt_enable) and int(runtime_args.batch_size) != 1:
        raise ValueError('RL prompt learning currently requires --batch_size 1 for stable per-sample episodes.')
    if runtime_args.stage3_uncertainty_type == 'hybrid':
        weight_sum = (
            runtime_args.stage3_uncertainty_alpha
            + runtime_args.stage3_uncertainty_beta
            + runtime_args.stage3_uncertainty_gamma
        )
        if weight_sum <= 0:
            raise ValueError('Hybrid uncertainty weights must sum to a positive value.')

    if bool(runtime_args.stage3_enable_difficulty_routing) and runtime_args.stage3_route_topk == 0:
        raise ValueError('Stage-3 routing is enabled, but --stage3_route_topk is 0.')

def _log_stage3_config(runtime_args):
    enabled = bool(runtime_args.stage3_enable_difficulty_routing)
    msg = (
        '[Stage-3] '
        f'run_mode={runtime_args.run_mode}, '
        f'infer_teacher_refine={bool(runtime_args.infer_enable_teacher_refinement)}, '
        f'enabled={enabled}, '
        f'route_topk={runtime_args.stage3_route_topk}, '
        f'min_trigger={runtime_args.stage3_route_min_trigger}, '
        f'uncertainty_type={runtime_args.stage3_uncertainty_type}, '
        f'weights=({runtime_args.stage3_uncertainty_alpha}, '
        f'{runtime_args.stage3_uncertainty_beta}, {runtime_args.stage3_uncertainty_gamma}), '
        f'boundary_band={runtime_args.stage3_boundary_band_width}, '
        f'infer_use_gt_prompt={runtime_args.stage3_infer_use_gt_prompt}'
    )
    print(msg)
    logger.info(msg)


def _build_trainer(runtime_args):
    train_dataloader, val_dataloader = get_dataloaders(runtime_args, img_datas)
    sam_model = build_sam_model(runtime_args, runtime_args.device)
    student_model = build_student_model(runtime_args, runtime_args.device)
    return BaseTrainer(
        model=sam_model,
        student_model=student_model,
        dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        args=runtime_args,
        logger=logger,
        model_save_path=MODEL_SAVE_PATH,
        click_methods=click_methods,
        img_datas=img_datas,
        device=runtime_args.device,
        pseudo_label_save_dir=PSEUDO_LABEL_DIR,
    )


def main_worker(rank, runtime_args):
    _validate_args(runtime_args)
    setup_distributed(rank, runtime_args.world_size, runtime_args.port)

    torch.cuda.set_device(rank)
    runtime_args.num_workers = int(runtime_args.num_workers / runtime_args.ngpus_per_node)
    runtime_args.device = torch.device(f"cuda:{rank}")
    runtime_args.rank = rank

    init_seeds(2023 + rank)

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO if rank in [-1, 0] else logging.WARN,
        filemode='w',
        filename=os.path.join(LOG_OUT_DIR, f'output_{cur_time}.log'),
    )

    trainer = _build_trainer(runtime_args)
    _log_stage3_config(runtime_args)
    if runtime_args.run_mode == 'eval':
        if rank in [-1, 0]:
            val_dice, _, val_student_dice, _ = trainer.eval_epoch(0, runtime_args.eval_num_clicks)
            print(f'[Eval] primary_dice={val_dice:.4f}, student_dice={val_student_dice:.4f}')
        cleanup_distributed()
        return

    trainer.train()
    cleanup_distributed()


def main():
    mp.set_sharing_strategy('file_system')
    device_config(args)
    _validate_args(args)

    if args.multi_gpu:
        mp.spawn(main_worker, nprocs=args.world_size, args=(args,))
        return

    init_seeds(2023)
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filemode='w',
        filename=os.path.join(LOG_OUT_DIR, f'output_{cur_time}.log'),
    )

    trainer = _build_trainer(args)
    _log_stage3_config(args)
    if args.run_mode == 'eval':
        val_dice, _, val_student_dice, _ = trainer.eval_epoch(0, args.eval_num_clicks)
        print(f'[Eval] primary_dice={val_dice:.4f}, student_dice={val_student_dice:.4f}')
        return

    trainer.train()


if __name__ == '__main__':
    main()
