"""
Evaluation script. Distributed evaluation has been axed due to limited resources of the research. Some changes were 
applied to make it more intuitive.

Modified from Plain ViT-S/16 ImageNet-1K Pre-training in PyTorch (https://github.com/ddgoede/vit_s_i1k_torch).
Originally licensed under MIT License.
"""
import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torchvision import transforms as T
import numpy as np

import models
from utils import MetricLogger
from utils.transforms import RandomSelectiveErasing, ThresholdBackgroundZeroing, NormalizePreservingZeros


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('ViT/SE evaluation', add_help=True)

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)

    parser.add_argument('--model_name', default='vit_se_h_14', type=str,
                        help='Name of the ViT/SE model to use')
    parser.add_argument('--encoder_mode', default='sparse', type=str, choices=('sparse', 'masked', 'default'),
                        help='Encoder forward mode: sparse, masked, or default')
    parser.add_argument('--num_classes', default=1000, type=int,
                        help='Number of output classes')
    parser.add_argument('--resize_size', default=518, type=int,
                        help='256 for ViT/SE-B, 242 for ViT/SE-L, 518 for ViT/SE-H')
    parser.add_argument('--crop_size', default=518, type=int,
                        help='224 for ViT/SE-B and ViT-L, 518 for ViT/SE-H')
    parser.add_argument('--interpolation', default='bicubic', type=str,
                        help='"bilinear" for ViT/SE-B and ViT/SE-L, "bicubic" for ViT/SE-H')

    parser.add_argument('--weights', required=False, type=str,
                        help='Path to weights of trained model (optional)')
    parser.add_argument('--data_path', default='/path/to/imagenet-1k', type=str,
                        help='Path to dataset')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Evaluation device')

    parser.add_argument('--erase_ratio', default=0.5, type=float)
    parser.add_argument('--background_threshold', default=-1.0, type=float,
                        help='Apply ThresholdBackgroundZeroing after ToTensor if non-negative')
    parser.add_argument('--seed', default=66, type=int)

    return parser.parse_args()


def main(args: argparse.Namespace):
    device = torch.device(args.device)

    if args.interpolation == 'bilinear':
        interpolation = transforms.InterpolationMode.BILINEAR
    elif args.interpolation == 'bicubic':
        interpolation = transforms.InterpolationMode.BICUBIC
    else:
        raise NotImplementedError('Only bilinear and bicubic interpolations are supported!')

    set_seed(args.seed)
    patch_size = get_patch_size(args.model_name)

    normalize_cls = NormalizePreservingZeros if args.background_threshold >= 0 else T.Normalize

    ops = [
        T.Resize(args.resize_size, interpolation=interpolation),
        T.CenterCrop(args.crop_size),
        T.ToTensor(),
    ]
    if args.background_threshold >= 0:
        ops.append(ThresholdBackgroundZeroing(threshold=args.background_threshold))
    ops.extend([
        normalize_cls(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        RandomSelectiveErasing(ratio=(args.erase_ratio, args.erase_ratio), patch_size=patch_size)
    ])
    transform = T.Compose(ops)
    dataset = datasets.ImageFolder(Path(args.data_path) / 'val', transform=transform)

    sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    model = getattr(models, args.model_name)(
        pretrained=args.weights is None,
        weights=args.weights,
        encoder_mode=args.encoder_mode,
        num_classes=args.num_classes,
    )
    model = model.to(device)

    evaluate(dataloader, model, device)


@torch.no_grad()
def evaluate(dataloader, model, device):
    metric_logger = MetricLogger(delimiter='  ')
    header = 'Testing:'
    model.eval()

    for (samples, targets) in metric_logger.log_every(dataloader, 20, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # img = T.ToPILImage()(samples[0])
        # img.show()

        with torch.amp.autocast('cuda'):
            output = model(samples)
            loss = F.cross_entropy(output, targets)

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))

        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    print('* Acc@1: {top1.global_avg:.4f}  Acc@5: {top5.global_avg:.4f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: List[int] = (1, 5)):
    maxk = min(max(topk), output.size(1))
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        effective_k = min(k, output.size(1))
        correct_k = correct[:effective_k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    np.random.seed(seed)


def get_patch_size(model_name: str) -> int:
    try:
        return int(model_name.split('_')[-1])
    except (ValueError, IndexError) as e:
        raise ValueError(f'Unable to infer patch size from model name: {model_name}') from e


if __name__ == '__main__':
    args = parse_args()
    main(args)
