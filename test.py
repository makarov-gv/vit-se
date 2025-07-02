import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torchvision import transforms as T
import numpy as np

import models
from utils import misc
from utils.transforms import RandomSelectiveErasing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('ViT/SE evaluation', add_help=True)

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)

    parser.add_argument('--model_name', default='vit_se_h_14', type=str,
                        help='Name of the ViT/SE model to use')
    parser.add_argument('--resize_size', default=518, type=int,
                        help='256 for ViT/SE-Base, 242 for ViT/SE-Large, 518 for ViT/SE-Huge')
    parser.add_argument('--crop_size', default=518, type=int,
                        help='224 for ViT/SE-Base and ViT-Large, 518 for ViT/SE-Huge')
    parser.add_argument('--interpolation', default='bicubic', type=str,
                        help='"bilinear" for ViT/SE-Base and ViT/SE-Large, "bicubic" for ViT/SE-Huge')

    parser.add_argument('--weights', required=False, type=str,
                        help='Path to weights of trained model (optional)')
    parser.add_argument('--data_path', default='/path/to/imagenet-1k', type=str,
                        help='Path to dataset')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Evaluation device')

    parser.add_argument('--erase_ratio', default=0.5, type=float)
    parser.add_argument('--seed', default=66, type=int)

    return parser.parse_args()


def set_seed(args):
    seed = args.seed + misc.get_rank()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    np.random.seed(seed)


def main(args):
    device = torch.device(args.device)

    if args.interpolation == 'bilinear':
        interpolation = transforms.InterpolationMode.BILINEAR
    elif args.interpolation == 'bicubic':
        interpolation = transforms.InterpolationMode.BICUBIC
    else:
        raise NotImplementedError('Only bilinear and bicubic interpolations are supported!')

    set_seed(args)

    transform = T.Compose([
        T.Resize(args.resize_size, interpolation=interpolation),
        T.CenterCrop(args.crop_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        RandomSelectiveErasing(ratio=(args.erase_ratio, args.erase_ratio))
    ])
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

    model = getattr(models, args.model_name)(pretrained=args.weights is None, weights=args.weights)
    model = model.to(device)

    evaluate(dataloader, model, device)


@torch.no_grad()
def evaluate(data_loader, model, device):
    metric_logger = misc.MetricLogger(delimiter='  ')
    header = 'Testing:'
    model.eval()

    for (samples, targets) in metric_logger.log_every(data_loader, 20, header):
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

    metric_logger.synchronize_between_processes()

    print('* Acc@1: {top1.global_avg:.4f}  Acc@5: {top5.global_avg:.4f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))


def accuracy(output, target, topk=(1, 5)):
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
    main(args)
