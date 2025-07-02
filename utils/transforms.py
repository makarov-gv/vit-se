"""
Augmentations used during experiments with the Vision Transformer with Sparse Encoder (ViT/SE).

BigVisionRandAugment implementation from Plain ViT-S/16 ImageNet-1K Pre-training in PyTorch 
(https://github.com/ddgoede/vit_s_i1k_torch).
Originally licensed under MIT License.
"""
from collections import defaultdict
from typing import List, Optional, Tuple, Any
import math
import random

import torch
from torchvision.transforms import v2
import torchvision.transforms.functional as F


class RandomSelectiveErasing(object):
    """
    Sparsification augmentation as described in the paper.
    """

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


class BigVisionRandAugment(v2.RandAugment):
    """
    A clean implementation of RandAugment matching big_vision's behavior.

    This implementation:
    1. Uses the same transforms as big_vision (adds Invert, SolarizeAdd, Cutout)
    2. Matches the magnitude scaling to big_vision
    3. Uses the same fill value (128,128,128) as big_vision
    4. Fixes the Contrast transform to match the corrected behavior
    """

    def __init__(self,
                 num_ops: int = 2,
                 magnitude: int = 10,
                 num_magnitude_bins: int = 31,
                 interpolation: v2.InterpolationMode = v2.InterpolationMode.BILINEAR,
                 fill: Optional[List[int]] = None):
        if fill is None:
            fill = [128, 128, 128]  # match big_vision's default

        if isinstance(fill, defaultdict) or not isinstance(fill, (list, int)) and fill is not None:
            fill = [128, 128, 128]  # force safe value

        super().__init__(num_ops=num_ops,
                         magnitude=magnitude,
                         num_magnitude_bins=num_magnitude_bins,
                         interpolation=interpolation,
                         fill=fill)

        self._transforms = [
            'Identity', 'AutoContrast', 'Equalize', 'Invert', 'Rotate',
            'Posterize', 'Solarize', 'Color', 'Contrast', 'Brightness',
            'Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY',
            'SolarizeAdd', 'Cutout'
        ]

    def _get_transforms(self) -> List[str]:
        return self._transforms

    def _apply_op(self, img: torch.Tensor, op_name: str, magnitude: float) -> torch.Tensor:
        magnitude_scale = 1.0  # default scale factor
        if hasattr(self, 'num_magnitude_bins') and self.num_magnitude_bins > 0:
            magnitude_scale = magnitude / self.num_magnitude_bins

        return self._apply_image_transform(
            img,
            op_name,
            magnitude_scale * self.magnitude,
            self.interpolation,
            self.fill
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        fill = self.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_dimensions(img)[0]
            elif fill is not None:
                fill = [float(f) for f in fill]

        ops = random.choices(self._get_transforms(), k=self.num_ops)
        for op_name in ops:
            magnitude = float(torch.empty(1).uniform_(0, self.magnitude).item())
            img = self._apply_op(img, op_name, magnitude)

        return img

    def _apply_image_transform(self,
                               img: torch.Tensor,
                               transform_id: str,
                               magnitude: float,
                               interpolation: v2.InterpolationMode,
                               fill: Any) -> torch.Tensor:
        if isinstance(fill, defaultdict) or not (isinstance(fill, (list, int)) or fill is None):
            fill = [128, 128, 128]  # use default safe value

        if transform_id == "Identity":
            return img
        elif transform_id == "ShearX":
            return F.affine(
                img,
                angle=0.0,
                translate=[0, 0],
                scale=1.0,
                shear=[math.degrees(math.atan(magnitude)), 0.0],
                interpolation=interpolation,
                fill=fill,
                center=[0, 0],
            )
        elif transform_id == "ShearY":
            return F.affine(
                img,
                angle=0.0,
                translate=[0, 0],
                scale=1.0,
                shear=[0.0, math.degrees(math.atan(magnitude))],
                interpolation=interpolation,
                fill=fill,
                center=[0, 0],
            )
        elif transform_id == "TranslateX":
            return F.affine(
                img,
                angle=0.0,
                translate=[int(magnitude), 0],
                scale=1.0,
                interpolation=interpolation,
                shear=[0.0, 0.0],
                fill=fill,
            )
        elif transform_id == "TranslateY":
            return F.affine(
                img,
                angle=0.0,
                translate=[0, int(magnitude)],
                scale=1.0,
                interpolation=interpolation,
                shear=[0.0, 0.0],
                fill=fill,
            )
        elif transform_id == "Rotate":
            return F.rotate(img, angle=magnitude, interpolation=interpolation, fill=fill)
        elif transform_id == "Brightness":
            return F.adjust_brightness(img, brightness_factor=1.0 + magnitude)
        elif transform_id == "Color":
            return F.adjust_saturation(img, saturation_factor=1.0 + magnitude)
        elif transform_id == "Contrast":
            return F.adjust_contrast(img, contrast_factor=1.0 + magnitude)
        elif transform_id == "Sharpness":
            return F.adjust_sharpness(img, sharpness_factor=1.0 + magnitude)
        elif transform_id == "Posterize":
            return F.posterize(img, bits=int(magnitude))
        elif transform_id == "Solarize":
            from torchvision.transforms import _functional_tensor as _FT
            bound = _FT._max_value(img.dtype) if isinstance(img, torch.Tensor) else 255.0
            return F.solarize(img, threshold=bound * magnitude)
        elif transform_id == "AutoContrast":
            return F.autocontrast(img)
        elif transform_id == "Equalize":
            return F.equalize(img)
        elif transform_id == "Invert":
            return F.invert(img)
        elif transform_id == "SolarizeAdd":
            return self._solarize_add(img, addition=int(magnitude))
        elif transform_id == "Cutout":
            return self._cutout(img, pad_size=int(magnitude), replace=fill)
        else:
            raise ValueError(f"No transform available for {transform_id}")

    def _solarize_add(self, img: Any, addition: int = 0, threshold: int = 128) -> Any:
        from torchvision.transforms import _functional_tensor as _FT

        if not isinstance(img, torch.Tensor):
            from torchvision.transforms import functional as F_pil
            tensor_img = F_pil.to_tensor(img)
            result = self._solarize_add(tensor_img, addition, threshold)
            return F_pil.to_pil_image(result)

        bound = _FT._max_value(img.dtype)
        added_img = img.to(torch.int64) + addition
        added_img = added_img.clip(0, bound).to(torch.uint8)

        return torch.where(img < threshold, added_img, img)

    def _cutout(self, img: Any, pad_size: int, replace: Any = None) -> Any:
        if replace is None:
            replace = [128, 128, 128]  # match big_vision's default

        is_pil = not isinstance(img, torch.Tensor)
        if is_pil:
            from torchvision.transforms import functional as F_pil
            tensor_img = F_pil.to_tensor(img)
            result = self._cutout(tensor_img, pad_size, replace)
            return F_pil.to_pil_image(result)

        _, height, width = F.get_dimensions(img)

        cutout_center_height = torch.randint(0, height, (1,)).item()
        cutout_center_width = torch.randint(0, width, (1,)).item()

        lower_pad = max(0, cutout_center_height - pad_size)
        upper_pad = max(0, height - cutout_center_height - pad_size)
        left_pad = max(0, cutout_center_width - pad_size)
        right_pad = max(0, width - cutout_center_width - pad_size)

        cutout_shape = [height - (lower_pad + upper_pad), width - (left_pad + right_pad)]

        replace_tensor = torch.tensor(replace, device=img.device, dtype=img.dtype)
        if len(replace_tensor.shape) == 1:
            replace_tensor = replace_tensor.unsqueeze(1).unsqueeze(1)

        return F.erase(img, lower_pad, left_pad, cutout_shape[0], cutout_shape[1], replace_tensor)
