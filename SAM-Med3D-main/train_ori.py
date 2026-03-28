import warnings
warnings.filterwarnings("ignore")# 忽略警告
# set up environment
import argparse
import datetime
import logging
import os
import random
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchio as tio
from monai.losses import DiceCELoss
from torch.backends import cudnn
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from segment_anything.build_sam3D import sam_model_registry3D
from utils.click_method import get_next_click3D_torch_2
from utils.data_loader import Dataset_Union_ALL, Union_Dataloader
from utils.data_paths import img_datas

join = os.path.join

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='union_train')
parser.add_argument('--click_type', type=str, default='random')
parser.add_argument('--multi_click', action='store_true', default=False)
parser.add_argument('--model_type', type=str, default='vit_b_ori')
parser.add_argument('--checkpoint', type=str, default='ckpt/sam_med3d.pth')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--work_dir', type=str, default='work_dir')

# train
parser.add_argument('--num_workers', type=int, default=24)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0, 1])
parser.add_argument('--multi_gpu', action='store_true', default=False)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--allow_partial_weight', action='store_true', default=False)

# lr_scheduler
parser.add_argument('--lr_scheduler', type=str, default='multisteplr')
parser.add_argument('--step_size', type=list, default=[120, 180])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--accumulation_steps', type=int, default=20)
parser.add_argument('--lr', type=float, default=8e-4)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--port', type=int, default=12361)

# validation
parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio (0-1)')
parser.add_argument('--val_interval', type=int, default=5, help='Validation interval (every N epochs)')
parser.add_argument('--eval_num_clicks', type=int, default=6, help='Number of clicks for validation')

args = parser.parse_args()

device = args.device
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])
logger = logging.getLogger(__name__)
LOG_OUT_DIR = join(args.work_dir, args.task_name)
click_methods = {
    'random': get_next_click3D_torch_2,
}
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)


def build_model(args):
    sam_model = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
    if args.multi_gpu:
        sam_model = DDP(sam_model, device_ids=[args.rank], output_device=args.rank)
    return sam_model


def get_dataloaders(args):
    """Load training and validation dataloaders.
    Splits data by train/val ratio if val_split > 0.
    """
    # Calculate split index
    split_num = max(1, int(1.0 / args.val_split)) if args.val_split > 0 else 1
    
    
    # Training dataset (use indices 0, split_num, 2*split_num, ...)
    train_dataset = Dataset_Union_ALL(
        paths=img_datas,
        transform=tio.Compose([
            tio.ToCanonical(),
            tio.CropOrPad(mask_name='label',
                          target_shape=(args.img_size, args.img_size,
                                        args.img_size)),  # crop only object region
            tio.RandomFlip(axes=(0, 1, 2)),
        ]),
        threshold=1000,
        split_num=split_num,
        split_idx=0)

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
    
    # Validation dataset (use indices 1, 1+split_num, 1+2*split_num, ...)
    val_dataloader = None
    if args.val_split > 0 and split_num > 1:
        val_dataset = Dataset_Union_ALL(
            paths=img_datas,
            transform=tio.Compose([
                tio.ToCanonical(),
                tio.CropOrPad(mask_name='label',
                              target_shape=(args.img_size, args.img_size,
                                            args.img_size)),
                # No random flip for validation
            ]),
            threshold=1000,
            split_num=split_num,
            split_idx=1)
        
        
        val_dataloader = Union_Dataloader(
            dataset=val_dataset,
            sampler=None,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    
    return train_dataloader, val_dataloader


class BaseTrainer:

    def __init__(self, model, dataloaders, val_dataloaders, args):

        self.model = model
        self.dataloaders = dataloaders
        self.val_dataloaders = val_dataloaders
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.best_val_dice = 0.0
        self.step_best_loss = np.inf
        self.step_best_dice = 0.0
        self.losses = []
        self.dices = []
        self.val_dices = []
        self.ious = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        if (args.resume):
            self.init_checkpoint(
                join(self.args.work_dir, self.args.task_name, 'sam_model_latest.pth'))
        else:
            self.init_checkpoint(self.args.checkpoint)

        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
        


    def set_loss_fn(self):
        self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')


    def set_optimizer(self):
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model

        self.optimizer = torch.optim.AdamW(
            [
                {
                    'params': sam_model.image_encoder.parameters()
                },  # , 'lr': self.args.lr * 0.1},
                {
                    'params': sam_model.prompt_encoder.parameters(),
                    'lr': self.args.lr * 0.1
                },
                {
                    'params': sam_model.mask_decoder.parameters(),
                    'lr': self.args.lr * 0.1
                },
            ],
            lr=self.args.lr,
            betas=(0.9, 0.999),
            weight_decay=self.args.weight_decay)

    def set_lr_scheduler(self):
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                     self.args.step_size,
                                                                     self.args.gamma)
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                self.args.step_size[0],
                                                                self.args.gamma)
        elif self.args.lr_scheduler == 'coswarm':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 0.1)

    def init_checkpoint(self, ckp_path):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.multi_gpu:
                dist.barrier()
                last_ckpt = torch.load(ckp_path, map_location=self.args.device, weights_only=False)
            else:
                last_ckpt = torch.load(ckp_path, map_location=self.args.device, weights_only=False)

        if last_ckpt:
            if (self.args.allow_partial_weight):
                if self.args.multi_gpu:
                    self.model.module.load_state_dict(last_ckpt['model_state_dict'], strict=False)
                else:
                    self.model.load_state_dict(last_ckpt['model_state_dict'], strict=False)
            else:
                if self.args.multi_gpu:
                    self.model.module.load_state_dict(last_ckpt['model_state_dict'])
                else:
                    self.model.load_state_dict(last_ckpt['model_state_dict'])
            if not self.args.resume:
                self.start_epoch = 0
            else:
                self.start_epoch = last_ckpt['epoch']
                self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
                self.losses = last_ckpt['losses']
                self.dices = last_ckpt['dices']
                self.best_loss = last_ckpt['best_loss']
                self.best_dice = last_ckpt['best_dice']
            print(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch})")
        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")

    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "losses": self.losses,
                "dices": self.dices,
                "best_loss": self.best_loss,
                "best_dice": self.best_dice,
                "args": self.args,
                "used_datas": img_datas,
            }, join(MODEL_SAVE_PATH, f"sam_model_{describe}.pth"))

    def batch_forward(self, sam_model, image_embedding, gt3D, low_res_masks, points=None):

        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=points,
            boxes=None,
            masks=low_res_masks,
        )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        prev_masks = F.interpolate(low_res_masks,
                                   size=gt3D.shape[-3:],
                                   mode='trilinear',
                                   align_corners=False)
        return low_res_masks, prev_masks

    def get_points(self, prev_masks, gt3D):
        batch_points, batch_labels = click_methods[self.args.click_type](prev_masks, gt3D)

        points_co = torch.cat(batch_points, dim=0).to(device)
        points_la = torch.cat(batch_labels, dim=0).to(device)

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        points_multi = torch.cat(self.click_points, dim=1).to(device)
        labels_multi = torch.cat(self.click_labels, dim=1).to(device)

        if self.args.multi_click:
            points_input = points_multi
            labels_input = labels_multi
        else:
            points_input = points_co
            labels_input = points_la
        return points_input, labels_input

    def interaction(self, sam_model, image_embedding, gt3D, num_clicks):
        return_loss = 0
        prev_masks = torch.zeros_like(gt3D).to(gt3D.device)
        low_res_masks = F.interpolate(prev_masks.float(),
                                      size=(args.img_size // 4, args.img_size // 4,
                                            args.img_size // 4))
        random_insert = np.random.randint(2, 9)
        for num_click in range(num_clicks):
            points_input, labels_input = self.get_points(prev_masks, gt3D)

            if num_click == random_insert or num_click == num_clicks - 1:
                low_res_masks, prev_masks = self.batch_forward(sam_model,
                                                               image_embedding,
                                                               gt3D,
                                                               low_res_masks,
                                                               points=None)
            else:
                low_res_masks, prev_masks = self.batch_forward(sam_model,
                                                               image_embedding,
                                                               gt3D,
                                                               low_res_masks,
                                                               points=[points_input, labels_input])
            loss = self.seg_loss(prev_masks, gt3D)
            return_loss += loss
        return prev_masks, return_loss

    def get_dice_score(self, prev_masks, gt3D):

        def compute_dice(mask_pred, mask_gt):
            mask_threshold = 0.5

            mask_pred = (mask_pred > mask_threshold)
            mask_gt = (mask_gt > 0)

            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return np.NaN
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2 * volume_intersect / volume_sum

        pred_masks = (prev_masks > 0.5)
        true_masks = (gt3D > 0)
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        return (sum(dice_list) / len(dice_list)).item()

    def train_epoch(self, epoch, num_clicks):
        epoch_loss = 0
        epoch_iou = 0
        self.model.train()
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model
            self.args.rank = -1

        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            tbar = tqdm(self.dataloaders)
        else:
            tbar = self.dataloaders

        self.optimizer.zero_grad()
        step_loss = 0
        epoch_dice = 0
        for step, data3D in enumerate(tbar):
            try:
                image3D, gt3D = data3D["image"], data3D["label"]
            except Exception as e:
                print(f"Error processing batch at step {step}: {e}")
            # import pdb; pdb.set_trace()
            my_context = self.model.no_sync if self.args.rank != - \
                1 and step % self.args.accumulation_steps != 0 else nullcontext

            with my_context():

                image3D = self.norm_transform(image3D.squeeze(dim=1))  # (N, C, W, H, D)
                image3D = image3D.unsqueeze(dim=1)

                image3D = image3D.to(device)
                gt3D = gt3D.to(device).type(torch.long)
                with torch.amp.autocast("cuda"):
                    image_embedding = sam_model.image_encoder(image3D)

                    self.click_points = []
                    self.click_labels = []

                    pred_list = []

                    prev_masks, loss = self.interaction(sam_model,
                                                        image_embedding,
                                                        gt3D,
                                                        num_clicks=11)

                epoch_loss += loss.item()
                epoch_dice += self.get_dice_score(prev_masks, gt3D)
                cur_loss = loss.item()

                loss /= self.args.accumulation_steps

                self.scaler.scale(loss).backward()

            if step % self.args.accumulation_steps == 0 and step != 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                print_loss = step_loss / self.args.accumulation_steps
                step_loss = 0
                print_dice = self.get_dice_score(prev_masks, gt3D)
            else:
                step_loss += cur_loss

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                if step % self.args.accumulation_steps == 0 and step != 0:
                    if print_dice > self.step_best_dice:
                        self.step_best_dice = print_dice
                        if print_dice > 0.9:
                            self.save_checkpoint(epoch,
                                                 sam_model.state_dict(),
                                                 describe=f'{epoch}_step_dice:{print_dice}_best')
                    if print_loss < self.step_best_loss:
                        self.step_best_loss = print_loss

        epoch_loss /= step + 1
        epoch_dice /= step + 1

        return epoch_loss, epoch_iou, epoch_dice, pred_list

    def eval_epoch(self, epoch, num_clicks):
        """Validation epoch without ground truth leakage.
        
        For validation, we use the model's own predictions to generate prompts
        instead of ground truth labels, simulating real interactive usage:
        - First iteration: Use empty mask (zero predictions), no point clicks
        - Subsequent iterations: Use previous model prediction as mask prompt
        - Points are NOT derived from ground truth (no label information leakage)
        """
        if self.val_dataloaders is None:
            return 0
        
        self.model.eval()
        val_dice_list = []
        
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model
        
        embed_dim = sam_model.prompt_encoder.embed_dim
        
        # Debug: check model state
        model_param_sample = next(sam_model.image_encoder.parameters())

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(self.val_dataloaders)):
                try:
                    image3D = batch_data["image"]
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue
                
                # Get ground truth for evaluation metrics only (NOT for generating prompts)
                if "label" in batch_data:
                    gt3D = batch_data["label"]
                else:
                    continue
                
                image3D = self.norm_transform(image3D.squeeze(dim=1))
                image3D = image3D.unsqueeze(dim=1)
                image3D = image3D.to(device)
                gt3D = gt3D.to(device).type(torch.long)
                
                with torch.amp.autocast("cuda"):
                    # Get image embedding once
                    image_embedding = sam_model.image_encoder(image3D)
                    
                    # Initialize prediction (empty masks)
                    prev_masks = torch.zeros_like(gt3D).float()
                    low_res_masks = F.interpolate(
                        prev_masks,
                        size=(self.args.img_size // 4, self.args.img_size // 4,
                              self.args.img_size // 4),
                        mode='trilinear',
                        align_corners=False
                    )
                    
                    # Interactive prediction loop (fewer clicks than training for speed)
                    for num_click in range(num_clicks):
                        # For validation, we don't use point clicks from ground truth
                        # We only use the mask from previous iteration as prompt
                        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                            points=None,  # No point prompts (no ground truth leakage)
                            boxes=None,
                            masks=low_res_masks,  # Use previous prediction as mask
                        )
                        
                        low_res_masks, iou_predictions = sam_model.mask_decoder(
                            image_embeddings=image_embedding.to(device),
                            image_pe=sam_model.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False,
                        )
                        
                        # Upsample to original size
                        prev_masks = F.interpolate(
                            low_res_masks,
                            size=gt3D.shape[-3:],
                            mode='trilinear',
                            align_corners=False
                        )
                    
                    # Compute Dice score against ground truth
                    batch_dice = self.get_dice_score(prev_masks, gt3D)
                    val_dice_list.append(batch_dice)
        
        # Return average validation Dice
        avg_val_dice = (sum(val_dice_list) / len(val_dice_list)) if val_dice_list else 0
        
        self.model.train()
        return avg_val_dice

    def plot_result(self, plot_data, description, save_name, val_data=None):
        """Plot training (and optionally validation) curves.
        
        Args:
            plot_data: Training data list
            description: Plot title
            save_name: Name for saved file
            val_data: Optional validation data list for comparison
        """
        plt.figure(figsize=(10, 6))
        epochs = list(range(len(plot_data)))
        plt.plot(epochs, plot_data, label='Train', linewidth=2, marker='o', markersize=4)
        
        if val_data is not None and len(val_data) > 0:
            # Validation data is sampled every val_interval epochs
            val_epochs = list(range(0, len(plot_data), self.args.val_interval))[:len(val_data)]
            plt.plot(val_epochs, val_data, label='Validation', linewidth=2, marker='s', markersize=4)
            plt.legend(loc='best', fontsize=10)
        
        plt.title(description, fontsize=12)
        plt.xlabel('Epoch', fontsize=11)
        plt.ylabel(f'{save_name}', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(join(MODEL_SAVE_PATH, f'{save_name}.png'), dpi=100)
        plt.close()

    def train(self):
        self.scaler = torch.amp.GradScaler("cuda")
        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f'Epoch: {epoch}/{self.args.num_epochs - 1}')

            if self.args.multi_gpu:
                dist.barrier()
                self.dataloaders.sampler.set_epoch(epoch)
            num_clicks = np.random.randint(1, 21)
            epoch_loss, epoch_iou, epoch_dice, pred_list = self.train_epoch(epoch, num_clicks)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.args.multi_gpu:
                dist.barrier()

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                self.losses.append(epoch_loss)
                self.dices.append(epoch_dice)
                print(f'Epoch\t {epoch}\t : loss: {epoch_loss}, train_dice: {epoch_dice}')
                logger.info(f'Epoch\t {epoch}\t : loss: {epoch_loss}, train_dice: {epoch_dice}')

                if self.args.multi_gpu:
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()

                # save latest checkpoint
                self.save_checkpoint(epoch, state_dict, describe='latest')

                # save train loss best checkpoint
                if epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                    self.save_checkpoint(epoch, state_dict, describe='loss_best')

                # save train dice best checkpoint
                if epoch_dice > self.best_dice:
                    self.best_dice = epoch_dice
                    self.save_checkpoint(epoch, state_dict, describe='dice_best')

                # Validation
                if epoch % self.args.val_interval == 0 and self.val_dataloaders is not None:
                    val_dice = self.eval_epoch(epoch, self.args.eval_num_clicks)
                    self.val_dices.append(val_dice)
                    print(f'Epoch\t {epoch}\t : val_dice: {val_dice}')
                    logger.info(f'Epoch\t {epoch}\t : val_dice: {val_dice}')
                    
                    # save validation best checkpoint
                    if val_dice > self.best_val_dice:
                        self.best_val_dice = val_dice
                        self.save_checkpoint(epoch, state_dict, describe='val_dice_best')

                # Plot results (with validation if available)
                self.plot_result(self.losses, 'Dice + Cross Entropy Loss', 'Loss')
                self.plot_result(self.dices, 'Dice Score', 'Dice', val_data=self.val_dices if self.val_dices else None)
        logger.info('=====================================================================')
        logger.info(f'Best loss: {self.best_loss}')
        logger.info(f'Best train dice: {self.best_dice}')
        logger.info(f'Best val dice: {self.best_val_dice}')
        logger.info(f'Total loss: {self.losses}')
        logger.info(f'Total train dice: {self.dices}')
        logger.info(f'Total val dice: {self.val_dices}')
        logger.info('=====================================================================')
        logger.info(f'args : {self.args}')
        logger.info(f'Used datasets : {img_datas}')
        logger.info('=====================================================================')


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def device_config(args):
    try:
        if not args.multi_gpu:
            # Single GPU
            if args.device == 'mps':
                args.device = torch.device('mps')
            else:
                args.device = torch.device(f"cuda:{args.gpu_ids[0]}")
        else:
            args.nodes = 1
            args.ngpus_per_node = len(args.gpu_ids)
            args.world_size = args.nodes * args.ngpus_per_node

    except RuntimeError as e:
        print(e)


def main():
    mp.set_sharing_strategy('file_system')
    device_config(args)
    if args.multi_gpu:
        mp.spawn(main_worker, nprocs=args.world_size, args=(args, ))
    else:
        random.seed(2023)
        np.random.seed(2023)
        torch.manual_seed(2023)
        # Load datasets
        train_dataloader, val_dataloader = get_dataloaders(args)
        # Build model
        model = build_model(args)
        # Create trainer
        trainer = BaseTrainer(model, train_dataloader, val_dataloader, args)
        # Train
        trainer.train()


def main_worker(rank, args):
    setup(rank, args.world_size)

    torch.cuda.set_device(rank)
    args.num_workers = int(args.num_workers / args.ngpus_per_node)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank

    init_seeds(2023 + rank)

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(format='[%(asctime)s] - %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        level=logging.INFO if rank in [-1, 0] else logging.WARN,
                        filemode='w',
                        filename=os.path.join(LOG_OUT_DIR, f'output_{cur_time}.log'))

    train_dataloader, val_dataloader = get_dataloaders(args)
    model = build_model(args)
    trainer = BaseTrainer(model, train_dataloader, val_dataloader, args)
    trainer.train()
    cleanup()


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(backend='nccl',
                            init_method=f'tcp://127.0.0.1:{args.port}',
                            world_size=world_size,
                            rank=rank)


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
