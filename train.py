# import argparse
# import os
# import sys
# import datetime
# import time
# import math
# import json
# from pathlib import Path

# # This implementation utilizes utility functions from Meta Platforms repositories
# # (DeiT: https://github.com/facebookresearch/deit and MAE: https://github.com/facebookresearch/mae)
# # These functions are in the utils directory and retain their original copyright notices.

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.backends.cudnn as cudnn
# import torch.nn.functional as F
# from torchvision import datasets, transforms

import argparse
from pathlib import Path
import time
import datetime
import json
import math

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torchvision import transforms as T
import numpy as np

import models
from utils import misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.transforms import RandomSelectiveErasing
# from utils import (
#     BigVisionRandAugment, TwoHotMixUp, NativeScaler, adjust_learning_rate,
#     misc
# )4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('ViT/SE training', add_help=True)

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)

    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Name of the experiment, used for saving checkpoints and logs')
    parser.add_argument('--model_name', default='vit_se_h_14', type=str,
                        help='Name of the ViT/SE model to use')
    parser.add_argument('--resize_size', default=518, type=int,
                        help='256 for ViT/SE-Base, 242 for ViT/SE-Large, 518 for ViT/SE-Huge')
    parser.add_argument('--crop_size', default=518, type=int,
                        help='224 for ViT/SE-Base and ViT-Large, 518 for ViT/SE-Huge')
    parser.add_argument('--interpolation', default='bicubic', type=str,
                        help='"bilinear" for ViT/SE-Base and ViT/SE-Large, "bicubic" for ViT/SE-Huge')

    parser.add_argument('--lr', type=float, default=1e-6,
                        help='Base learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Amount of warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')

    ###
    parser.add_argument('--randaug_n', type=int, default=2,
                        help='RandAugment number of ops')
    parser.add_argument('--randaug_m', type=int, default=10,
                        help='RandAugment magnitude')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='mixup alpha value')

    parser.add_argument('--weights', required=False, type=str,
                        help='Path to checkpoint to use (optional)')
    parser.add_argument('--data_path', default='/path/to/imagenet-1k', type=str,
                        help='Path to dataset')
    parser.add_argument('--output_dir', default='.', type=str,
                        help='Path to save checkpoints')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Training device')

    return parser.parse_args()


def main(args):
    device = torch.device(args.device)

    if args.interpolation == 'bilinear':
        interpolation = transforms.InterpolationMode.BILINEAR
    elif args.interpolation == 'bicubic':
        interpolation = transforms.InterpolationMode.BICUBIC
    else:
        raise NotImplementedError('Only bilinear and bicubic interpolations are supported!')

    # Match data augmentation parameters with big_vision
    # - Use RandomResizedCrop with scale=(0.05, 1.0) matching big_vision's inception_crop
    # - Use our BigVisionRandAugment with the same parameters as big_vision
    # - Normalize to [-1, 1] range like big_vision's value_range(-1, 1)
    # - Match big_vision's resize_small(256) | central_crop(224)
    transform_train = T.Compose([
        T.RandomResizedCrop(args.crop_size, scale=(0.05, 1.0), interpolation=interpolation),
        T.RandomHorizontalFlip(),
        # BigVisionRandAugment(num_ops=args.randaug_n, magnitude=args.randaug_m, fill=[128, 128, 128]),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        RandomSelectiveErasing(ratio=(0.0, 0.99))
    ])
    transform_val = T.Compose([
        T.Resize(args.resize_size, interpolation=interpolation),
        T.CenterCrop(args.crop_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        RandomSelectiveErasing(ratio=(0.0, 0.99))
    ])

    # mixup_fn = TwoHotMixUp(alpha=args.mixup_alpha) if args.mixup_alpha > 0 else None
    mixup_fn = None

    dataset_train = datasets.ImageFolder(Path(args.data_path) / 'train', transform=transform_train)
    dataset_val = datasets.ImageFolder(Path(args.data_path) / 'val', transform=transform_val)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    model = getattr(models, args.model_name)(pretrained=args.weights is None, weights=args.weights)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(dataloader_train) * args.epochs
    warmup_steps = len(dataloader_train) * args.warmup_epochs
    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        cycle_momentum=False
    )
    
    loss_scaler = NativeScaler()
    best_loss = torch.tensor(torch.inf).unsqueeze(0).to(device)  # mutable tensor

    print(f'Started training for {args.epochs} epochs')
    start_time = time.time()

    for epoch in range(args.epochs):
        train_stats = train_epoch(
            model,
            dataloader_train,
            optimizer,
            lr_scheduler,
            device,
            epoch,
            loss_scaler,
            mixup_fn,
            args
        )
        test_stats = validate_epoch(dataloader_val, model, device, best_loss, args)

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
        }

        with open(Path(args.output_dir) / 'log.txt', mode='a', encoding='utf-8') as f:
            f.write(json.dumps(log_stats) + '\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if device == 'cuda':
        torch.cuda.empty_cache()


def train_epoch(
    model,
    data_loader,
    optimizer,
    lr_scheduler,
    device,
    epoch,
    loss_scaler,
    mixup_fn,
    args: argparse.Namespace
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter='  ')
    header = 'Epoch: [{}]'.format(epoch)

    for (samples, targets) in metric_logger.log_every(data_loader, 20, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, lam, targets1, targets2 = mixup_fn(samples, targets)
            with torch.amp.autocast('cuda'):
                outputs = model(samples)
                loss = lam * F.cross_entropy(outputs, targets1) + (1 - lam) * F.cross_entropy(outputs, targets2)
        else:
            with torch.amp.autocast('cuda'):
                outputs = model(samples)
                loss = F.cross_entropy(outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError('Loss is {}, stopping training'.format(loss_value))

        optimizer.zero_grad()
        loss_scaler(loss, optimizer, clip_grad=1.0, parameters=model.parameters())
        lr_scheduler.step()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

    torch.save(model.state_dict(), Path(args.output_dir) / 'last.pth')

    metric_logger.synchronize_between_processes()

    print('Averaged stats:', metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_epoch(data_loader, model, device, best_loss: torch.Tensor, args: argparse.Namespace):
    metric_logger = misc.MetricLogger(delimiter='  ')
    header = 'Validation:'
    model.eval()

    for (samples, targets) in metric_logger.log_every(data_loader, 20, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            output = model(samples)
            loss = F.cross_entropy(output, targets)

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))

        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if metric_logger.loss.global_avg < best_loss[0]:
        best_loss[0] = metric_logger.loss.global_avg
        torch.save(model.state_dict(), Path(args.output_dir) / 'best.pth')

    metric_logger.synchronize_between_processes()

    print('* Acc@1 {top1.global_avg:.4f}  Acc@5 {top5.global_avg:.4f}  loss {losses.global_avg:.4f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def accuracy(output, target, topk=(1, 5)):
    '''Computes the accuracy over the k top predictions for the specified values of k'''
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    args = parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
