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
    Patch-aware sparsification augmentation as described in the paper.

    One of several sparsification modes is sampled with equal probability and
    then parameterized to match the target sparsity ratio on the patch grid.
    """

    def __init__(
        self,
        p: float = 1.0,
        ratio: Tuple[float, float] = (0.2, 0.4),
        width: Tuple[int, int] = (16, 48),
        patch_size: int = 16,
        modes: Tuple[str, ...] = ('line', 'padding', 'blob'),
        blob_seeds: Tuple[int, int] = (1, 3),
        max_attempts: int = 200
    ):
        self.p = p
        self.ratio = ratio
        self.width = width
        self.patch_size = patch_size
        self.modes = modes
        self.blob_seeds = blob_seeds
        self.max_attempts = max_attempts

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        _, h, w = img.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError('Image size must be divisible by patch size for patch-aware sparsification!')

        grid_h = h // self.patch_size
        grid_w = w // self.patch_size
        total_patches = grid_h * grid_w

        target_ratio = random.uniform(self.ratio[0], self.ratio[1])
        target_patches = int(round(target_ratio * total_patches))
        target_patches = max(0, min(total_patches, target_patches))

        if target_patches == 0:
            return img
        if target_patches == total_patches:
            return torch.zeros_like(img)

        mode = random.choice(self.modes)
        if mode == 'padding':
            return self._apply_padding(img, target_patches)
        if mode == 'blob':
            zero_mask = self._build_blob_mask(grid_h, grid_w, target_patches)
            return self._apply_zero_mask(img, zero_mask)

        zero_mask = self._build_line_mask(grid_h, grid_w, target_patches)
        return self._apply_zero_mask(img, zero_mask)

    def _apply_zero_mask(self, img: torch.Tensor, zero_mask: torch.Tensor) -> torch.Tensor:
        pixel_mask = (~zero_mask).repeat_interleave(self.patch_size, dim=0).repeat_interleave(self.patch_size, dim=1)
        return img * pixel_mask.to(device=img.device, dtype=img.dtype).unsqueeze(0)

    def _build_line_mask(self, grid_h: int, grid_w: int, target_patches: int) -> torch.Tensor:
        zero_mask = torch.zeros((grid_h, grid_w), dtype=torch.bool)
        current = 0

        while current < target_patches:
            remaining = target_patches - current
            horizontal = random.random() < 0.5
            added = self._paint_line_segment(zero_mask, remaining, horizontal)

            if added == 0:
                added = self._paint_line_segment(zero_mask, remaining, not horizontal)

            if added == 0:
                coords = torch.nonzero(~zero_mask, as_tuple=False)
                choice = coords[torch.randperm(coords.shape[0])[:remaining]]
                zero_mask[choice[:, 0], choice[:, 1]] = True
                break

            current += added

        return zero_mask

    def _paint_line_segment(self, zero_mask: torch.Tensor, remaining: int, horizontal: bool) -> int:
        grid_h, grid_w = zero_mask.shape
        preferred_min_thickness = max(1, math.ceil(self.width[0] / self.patch_size))
        max_thickness = max(1, math.ceil(self.width[1] / self.patch_size))
        max_thickness = min(max_thickness, grid_h if horizontal else grid_w)

        best = None
        best_added = 0
        best_aspect = -1.0
        best_preferred = False

        for thickness in range(1, max_thickness + 1):
            if horizontal:
                for y in range(grid_h - thickness + 1):
                    for length in range(grid_w, 0, -1):
                        for x in range(grid_w - length + 1):
                            view = zero_mask[y:y + thickness, x:x + length]
                            added = int((~view).sum().item())
                            if added == 0 or added > remaining:
                                continue

                            aspect = length / thickness
                            preferred = thickness >= preferred_min_thickness
                            if (
                                added > best_added
                                or (added == best_added and preferred and not best_preferred)
                                or (added == best_added and preferred == best_preferred and aspect > best_aspect)
                            ):
                                best = (y, x, thickness, length)
                                best_added = added
                                best_aspect = aspect
                                best_preferred = preferred
            else:
                for x in range(grid_w - thickness + 1):
                    for length in range(grid_h, 0, -1):
                        for y in range(grid_h - length + 1):
                            view = zero_mask[y:y + length, x:x + thickness]
                            added = int((~view).sum().item())
                            if added == 0 or added > remaining:
                                continue

                            aspect = length / thickness
                            preferred = thickness >= preferred_min_thickness
                            if (
                                added > best_added
                                or (added == best_added and preferred and not best_preferred)
                                or (added == best_added and preferred == best_preferred and aspect > best_aspect)
                            ):
                                best = (y, x, length, thickness)
                                best_added = added
                                best_aspect = aspect
                                best_preferred = preferred

        if best is None:
            return 0

        y, x, height, width = best
        view = zero_mask[y:y + height, x:x + width]
        added = int((~view).sum().item())
        zero_mask[y:y + height, x:x + width] = True

        return added

    def _build_blob_mask(self, grid_h: int, grid_w: int, target_patches: int) -> torch.Tensor:
        zero_mask = torch.zeros((grid_h, grid_w), dtype=torch.bool)

        seeds_max = min(self.blob_seeds[1], target_patches)
        seeds_min = min(self.blob_seeds[0], seeds_max)
        seed_count = random.randint(seeds_min, seeds_max)

        available = [(y, x) for y in range(grid_h) for x in range(grid_w)]
        for y, x in random.sample(available, seed_count):
            zero_mask[y, x] = True

        frontier = set()
        for y, x in torch.nonzero(zero_mask, as_tuple=False).tolist():
            frontier.update(self._neighbors(y, x, grid_h, grid_w))
        frontier = {cell for cell in frontier if not zero_mask[cell[0], cell[1]]}

        while int(zero_mask.sum().item()) < target_patches:
            if frontier:
                cells = list(frontier)
                weights = []
                for y, x in cells:
                    neighbors = self._neighbors(y, x, grid_h, grid_w)
                    masked_neighbors = sum(int(zero_mask[ny, nx].item()) for ny, nx in neighbors)
                    weights.append((masked_neighbors + 1) ** 2)
                y, x = random.choices(cells, weights=weights, k=1)[0]
                frontier.discard((y, x))
            else:
                cells = torch.nonzero(~zero_mask, as_tuple=False)
                y, x = cells[random.randrange(cells.shape[0])].tolist()

            zero_mask[y, x] = True
            for ny, nx in self._neighbors(y, x, grid_h, grid_w):
                if not zero_mask[ny, nx]:
                    frontier.add((ny, nx))

        return zero_mask

    def _neighbors(self, y: int, x: int, grid_h: int, grid_w: int) -> List[Tuple[int, int]]:
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < grid_h and 0 <= nx < grid_w:
                    neighbors.append((ny, nx))
        return neighbors

    def _apply_padding(self, img: torch.Tensor, target_patches: int) -> torch.Tensor:
        _, h, w = img.shape
        grid_h = h // self.patch_size
        grid_w = w // self.patch_size

        candidates = []
        for pad_cols in range(grid_w + 1):
            masked = pad_cols * grid_h
            candidates.append(('width', pad_cols, abs(masked - target_patches)))
        for pad_rows in range(grid_h + 1):
            masked = pad_rows * grid_w
            candidates.append(('height', pad_rows, abs(masked - target_patches)))

        best_error = min(error for _, _, error in candidates)
        axis, padding_units, _ = random.choice([candidate for candidate in candidates if candidate[2] == best_error])

        canvas = torch.zeros_like(img)
        if axis == 'width':
            keep_cols = max(1, grid_w - padding_units)
            keep_w = keep_cols * self.patch_size
            resized = F.resize(img, [h, keep_w], antialias=True)
            offset = (w - keep_w) // 2
            canvas[:, :, offset:offset + keep_w] = resized
        else:
            keep_rows = max(1, grid_h - padding_units)
            keep_h = keep_rows * self.patch_size
            resized = F.resize(img, [keep_h, w], antialias=True)
            offset = (h - keep_h) // 2
            canvas[:, offset:offset + keep_h, :] = resized

        return canvas


class ThresholdBackgroundZeroing(object):
    """
    Zero out low-intensity background regions using a channel-mean threshold.

    Intended for preprocessing naturally sparse images such as MRI slices.
    """

    def __init__(self, threshold: float = 25.0):
        self.threshold = threshold

    def __call__(self, img: Any) -> Any:
        if isinstance(img, torch.Tensor):
            return self._apply_tensor(img)

        tensor = F.pil_to_tensor(img)
        tensor = self._apply_tensor(tensor)
        return F.to_pil_image(tensor)

    def _apply_tensor(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() != 3:
            raise ValueError('Expected image tensor of shape (C, H, W)!')

        result = img.clone()
        mean_map = result.to(torch.float32).mean(dim=0, keepdim=True)
        threshold = self.threshold / 255.0 if result.is_floating_point() and float(result.max()) <= 1.5 else self.threshold
        return result.masked_fill(mean_map < threshold, 0)


class NormalizePreservingZeros(object):
    """
    Apply channel-wise normalization while keeping fully zero background pixels unchanged.

    This is useful for sparse pipelines where background regions are explicitly zeroed
    before normalization and must remain zero afterwards.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if not isinstance(img, torch.Tensor):
            raise TypeError('NormalizePreservingZeros expects a tensor input.')
        if img.dim() != 3:
            raise ValueError('Expected image tensor of shape (C, H, W)!')

        zero_mask = img.eq(0).all(dim=0, keepdim=True)
        normalized = F.normalize(img, mean=self.mean, std=self.std)
        return normalized.masked_fill(zero_mask, 0)


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
