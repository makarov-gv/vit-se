"""
Some of the utilities and TwoHotMixUp implementation from Plain ViT-S/16 ImageNet-1K Pre-training in PyTorch. Little to
none changes were made in this file. As well as transformations for images with sparsification augmentation to sparsify
ImageNet or other datasets.

Modified from Plain ViT-S/16 ImageNet-1K Pre-training in PyTorch (https://github.com/ddgoede/vit_s_i1k_torch).
Originally licensed under MIT License.
"""
from .transforms import *

from collections import defaultdict, deque
from typing import Optional, Tuple, Any
import time
import datetime
from math import inf

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size: int = 20, fmt: Optional[str] = None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"

        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value: float, n: int = 1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self) -> float:
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self) -> float:
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self) -> float:
        return self.total / (self.count + 1e-10)

    @property
    def max(self) -> float:
        return max(self.deque)

    @property
    def value(self) -> float:
        return self.deque[-1]

    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter: Optional[str] = "\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue

            if isinstance(v, torch.Tensor):
                v = v.item()

            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr: str) -> float:
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self) -> str:
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name: str, meter: SmoothedValue):
        self.meters[name] = meter

    def log_every(self, iterable: DataLoader, print_freq: int, header: Optional[str] = None):
        i = 0
        if not header:
            header = ''

        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]

        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')

        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj

            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler('cuda')

    def __call__(self, loss: torch.Tensor, optimizer: AdamW, clip_grad: Optional[float] = None,
                 parameters: Optional[Any] = None, create_graph: bool = False,
                 update_grad: bool = True) -> torch.Tensor:
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None

        return norm

    def state_dict(self) -> dict:
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict: dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters: Any, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)

    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack(
            [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)

    return total_norm


class TwoHotMixUp:
    """
    Implementation of MixUp that returns both targets as class indices instead of
    class probabilities, matching big_vision implementation.
    """

    def __init__(self, alpha: float = 0.2):
        self._dist = None
        if alpha > 0.0:
            self._dist = torch.distributions.Beta(alpha, alpha)

    def __call__(self,
                 images: torch.Tensor,
                 labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor,
                                                                                    torch.Tensor, torch.Tensor]:
        if self._dist is None:
            return images, labels

        batch_size = images.size(0)
        indices = torch.randperm(batch_size, device=images.device)
        lam = self._dist.sample().to(images.device)

        mixed_images = lam * images + (1 - lam) * images[indices]
        targets1 = labels
        targets2 = labels[indices]

        return mixed_images, lam, targets1, targets2
