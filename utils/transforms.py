
"""
Augmentations used during experiments with Vision Transformer with Sparse Encoder (ViT/SE).
"""
import random
from typing import Tuple

import torch


class RandomSelectiveErasing(object):
    def __init__(
        self,
        p: float = 1.0,
        ratio: Tuple[float, float] = (0.2, 0.4),
        width: Tuple[int, int] = (16, 48),
        max_attempts: int = 200
    ):
        self.p = p
        self.ratio = ratio
        self.width = width
        self.max_attempts = max_attempts

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        _, w, h = img.shape
        area = h * w
        pixels = random.uniform(self.ratio[0], self.ratio[1]) * area

        mask = torch.ones((h, w), device=img.device)
        mask_pixels = 0

        attempts = 0
        while mask_pixels < pixels and attempts < self.max_attempts:
            attempts += 1

            is_horizontal = random.random() < 0.5
            line_width = random.randint(self.width[0], self.width[1])

            if is_horizontal:
                y = random.randint(0, h - line_width)

                # if random.random() < 0.3:
                #     if random.random() < 0.5:
                #         y = 0
                #     else:
                #         y = h - line_width

                mask_pixels_ = w * line_width - torch.sum(mask[y:y+line_width, :] == 0).item()
                if mask_pixels + mask_pixels_ > pixels:
                    needed_pixels = int(pixels - mask_pixels)
                    if needed_pixels <= 0:
                        break

                    line_width_ = max(1, needed_pixels // w)
                    if line_width_ >= self.width[0]:
                        line_width = min(line_width_, line_width)
                        mask_pixels_ = w * line_width - torch.sum(mask[y:y+line_width, :] == 0).item()
                    else:
                        continue

                mask[y:y+line_width, :] = 0
                mask_pixels += mask_pixels_

            else:
                x = random.randint(0, w - line_width)

                # if random.random() < 0.3:
                #     if random.random() < 0.5:
                #         x = 0
                #     else:
                #         x = w - line_width

                mask_pixels_ = h * line_width - torch.sum(mask[:, x:x+line_width] == 0).item()
                if mask_pixels + mask_pixels_ > pixels:
                    needed_pixels = int(pixels - mask_pixels)
                    if needed_pixels <= 0:
                        break

                    line_width_ = max(1, needed_pixels // h)
                    if line_width_ >= self.width[0]:
                        line_width = min(line_width_, line_width)
                        mask_pixels_ = h * line_width - torch.sum(mask[:, x:x+line_width] == 0).item()
                    else:
                        continue

                mask[:, x:x+line_width] = 0
                mask_pixels += mask_pixels_

        mask = mask.unsqueeze(0).expand_as(img)
        img = img * mask

        return img
