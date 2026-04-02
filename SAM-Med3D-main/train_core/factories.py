import os
import sys
from fractions import Fraction
from pathlib import Path

import torch
import torchio as tio
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from segment_anything.build_sam3D import sam_model_registry3D
from utils.data_loader import Dataset_Union_ALL, Union_Dataloader


class _LabeledSubsetWrapper:
    """Wrapper to mark a subset of a dataset as labeled or unlabeled.
    
    Used in Stage-2B semi-supervised learning to split training data into
    labeled and unlabeled portions while maintaining the original dataset interface.
    """
    def __init__(self, dataset, indices, is_labeled=True):
        self.dataset = dataset
        self.indices = indices
        self.is_labeled = is_labeled
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        sample = self.dataset[actual_idx]
        # Override the is_labeled flag
        sample['is_labeled'] = self.is_labeled
        return sample


def _import_unet3d_standalone():
    unet_dir = 'pytorch-3dunet-master'
    unet_dir_str = str(unet_dir)
    if unet_dir_str not in sys.path:
        sys.path.insert(0, unet_dir_str)

    from unet3d_standalone import UNet3DWithPretrain

    return UNet3DWithPretrain


def build_sam_model(args, device):
    """Build SAM-Med3D model and wrap with DDP when requested."""
    sam_model = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
    if args.multi_gpu:
        sam_model = DDP(sam_model, device_ids=[args.rank], output_device=args.rank)
    return sam_model


def build_student_model(args, device):
    """Build student 3DUNet model and optionally load pretrained checkpoint."""
    UNet3DWithPretrain = _import_unet3d_standalone()

    student_model = UNet3DWithPretrain(
        in_channels=1,
        out_channels=args.student_num_classes,
        checkpoint_path=args.student_checkpoint,
        load_pretrained=bool(args.student_load_pretrained),
    ).to(device)

    if args.multi_gpu:
        student_model = DDP(student_model, device_ids=[args.rank], output_device=args.rank)

    if hasattr(student_model, 'pretrained_load_info') and student_model.pretrained_load_info is not None:
        print(f"[INFO] Student pretrained load info: {student_model.pretrained_load_info}")

    return student_model


def get_dataloaders(args, img_datas):
    """Build train/val dataloaders using split_num/split_idx partitioning.
    
    Supports Stage-2B semi-supervised learning:
    - If 0 < semi_supervised_labeled_ratio < 1: splits training data into labeled and unlabeled subsets
    - labeled samples: is_labeled=True (both SAM and Student use GT supervision)
    - unlabeled samples: is_labeled=False (Student uses SAM pseudo-labels only)
    """
    if args.val_split > 0:
        frac = Fraction(args.val_split).limit_denominator(100)
        split_num = frac.denominator
        val_indices_count = frac.numerator
        train_indices_count = split_num - val_indices_count
    else:
        split_num = 1
        train_indices_count = 1
        val_indices_count = 0

    print(f"[DEBUG] img_datas = {img_datas}")
    print(
        f"[DEBUG] Data split ratio: train {train_indices_count}/{split_num} "
        f"({train_indices_count / split_num * 100:.1f}%), val {val_indices_count}/{split_num} "
        f"({val_indices_count / split_num * 100:.1f}%)"
    )
    
    semi_ratio = float(args.semi_supervised_labeled_ratio)
    semi_enabled = 0.0 < semi_ratio < 1.0

    if semi_enabled:
        print(f"[DEBUG] Stage-2B semi-supervised learning enabled")
        print(f"[DEBUG] Labeled ratio: {semi_ratio:.1%}, "
              f"Unlabeled ratio: {1.0 - semi_ratio:.1%}")

    train_datasets = []
    for split_idx in range(train_indices_count):
        ds = Dataset_Union_ALL(
            paths=img_datas,
            transform=tio.Compose([
                tio.ToCanonical(),
                tio.CropOrPad(mask_name='label', target_shape=(args.img_size, args.img_size, args.img_size)),
                tio.RandomFlip(axes=(0, 1, 2)),
            ]),
            threshold=1000,
            split_num=split_num,
            split_idx=split_idx,
            is_labeled=True,  # Default: all labeled
        )
        train_datasets.append(ds)

    train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    
    # Stage-2B: Split training data into labeled and unlabeled subsets.
    if semi_enabled:
        total_size = len(train_dataset)
        labeled_size = int(total_size * semi_ratio)
        
        # Create labeled subset (indices 0 to labeled_size-1)
        labeled_indices = list(range(0, labeled_size))
        labeled_dataset = _LabeledSubsetWrapper(train_dataset, labeled_indices, is_labeled=True)
        
        # Create unlabeled subset (indices labeled_size to end)
        unlabeled_indices = list(range(labeled_size, total_size))
        unlabeled_dataset = _LabeledSubsetWrapper(train_dataset, unlabeled_indices, is_labeled=False)
        
        # Concatenate labeled and unlabeled subsets
        train_dataset = ConcatDataset([labeled_dataset, unlabeled_dataset])
        
        print(f"[DEBUG] Stage-2B dataset split: {labeled_size} labeled, {len(unlabeled_indices)} unlabeled")
    else:
        print('[DEBUG] Semi-supervised split disabled (ratio <= 0 or >= 1); using full labeled training.')

    if args.multi_gpu:
        train_sampler = DistributedSampler(train_dataset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_dataloader = Union_Dataloader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_dataloader = None
    if val_indices_count > 0:
        val_datasets = []
        for split_idx in range(train_indices_count, split_num):
            ds = Dataset_Union_ALL(
                paths=img_datas,
                transform=tio.Compose([
                    tio.ToCanonical(),
                    tio.CropOrPad(mask_name='label', target_shape=(args.img_size, args.img_size, args.img_size)),
                ]),
                threshold=1000,
                split_num=split_num,
                split_idx=split_idx,
            )
            val_datasets.append(ds)

        val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]

        print('[DEBUG] Data split details:')
        print(f'  - Train dataset size: {len(train_dataset)}')
        print(f'  - Val dataset size: {len(val_dataset)}')
        print(f'  - split_num: {split_num}')
        print(f'  - Train split_idx(s): {list(range(train_indices_count))}')
        print(f'  - Val split_idx(s): {list(range(train_indices_count, split_num))}')

        val_dataloader = Union_Dataloader(
            dataset=val_dataset,
            sampler=None,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        print(f"[DEBUG] No validation split (val_split={args.val_split})")

    return train_dataloader, val_dataloader
