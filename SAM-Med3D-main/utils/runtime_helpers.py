import random

import numpy as np
import torch
import torch.distributed as dist
from torch.backends import cudnn


def init_seeds(seed=0, cuda_deterministic=True):
    """Set Python/Numpy/PyTorch random seeds.

    Args:
        seed: Base random seed.
        cuda_deterministic: If True, favors reproducibility over speed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True


def device_config(args):
    """Populate runtime device fields on args."""
    if not args.multi_gpu:
        if args.device == 'mps':
            args.device = torch.device('mps')
        else:
            args.device = torch.device(f"cuda:{args.gpu_ids[0]}")
    else:
        args.nodes = 1
        args.ngpus_per_node = len(args.gpu_ids)
        args.world_size = args.nodes * args.ngpus_per_node


def setup_distributed(rank, world_size, port):
    """Initialize NCCL process group for DDP."""
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{port}',
        world_size=world_size,
        rank=rank,
    )


def cleanup_distributed():
    """Destroy process group when distributed training ends."""
    dist.destroy_process_group()
