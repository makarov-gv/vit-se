
"""
Vision Transformer with Sparse Encoder (ViT/SE).

Modified from torchvision (https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py).
Originally licensed under BSD 3-Clause License.
"""
import math
from collections import OrderedDict
from functools import partial, reduce
from typing import Callable, Optional

import torch
import torch.nn as nn

from .mlp import MLPBlock


class SparseEncoderLayer(nn.Module):
    """Sparse Transformer Encoder layer with key padding mask."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, sparse_x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        torch._assert(sparse_x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {sparse_x.shape}")

        x = self.ln_1(sparse_x)
        x, _ = self.self_attention(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.dropout(x)
        x = x + sparse_x

        y = self.ln_2(x)
        y = self.mlp(y)

        return x + y


class SparseEncoder(nn.Module):
    """Sparse Transformer Encoder for sequence to sequence translation without the processing of zero embeddings."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim

        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = SparseEncoderLayer(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")
        batch_size = x.shape[0]
        sparse_mask = (x == 0).all(dim=-1)
        # sparse_mask = (x <= 0).all(dim=-1)

        if batch_size != 1:
            subs = []
            lengths = []
            for i in range(batch_size):
                sub = x[i][~sparse_mask[i]].view(-1, self.hidden_dim)
                lengths.append(sub.shape[0])
                pos = self.pos_embedding.squeeze()[~sparse_mask[i]].view(-1, self.hidden_dim)

                sub = sub + pos
                subs.append(self.dropout(sub))

            padded_x = torch.nested.nested_tensor(subs, layout=torch.jagged).to_padded_tensor(0)
            padded_length = max(lengths)

            key_padding_mask = torch.zeros((batch_size, padded_length), dtype=torch.bool, device=x.device)
            for i, length in enumerate(lengths):
                if length < padded_length:
                    key_padding_mask[i, length:] = True

            padded_x = reduce(lambda acc, layer: layer(acc, key_padding_mask), self.layers, padded_x)
            padded_x = self.ln(padded_x)

            x_list = []
            for i in range(batch_size):
                sparse_x = padded_x[i, :lengths[i], :]
                x_ = torch.zeros((self.seq_length, self.hidden_dim), dtype=x.dtype, device=x.device)
                x_[~sparse_mask[i]] = sparse_x
                x_list.append(x_)

            x = torch.stack(x_list, dim=0)
        else:
            sparse_x = x[~sparse_mask].view(1, -1, self.hidden_dim)
            sparse_x = sparse_x + self.pos_embedding[~sparse_mask].view(1, -1, self.hidden_dim)

            sparse_x = self.dropout(sparse_x)
            sparse_x = self.layers(sparse_x)
            sparse_x = self.ln(sparse_x)

            x = torch.zeros((batch_size, self.seq_length, self.hidden_dim), dtype=x.dtype, device=x.device)
            x[~sparse_mask] = sparse_x

        return x

    # def forward(self, x: torch.Tensor):  # masked encoder
    #     key_padding_mask = (x == 0).all(dim=-1)

    #     x = x + self.pos_embedding
    #     x = self.dropout(x)

    #     x = reduce(lambda acc, layer: layer(acc, key_padding_mask), self.layers, x)
    #     x = self.ln(x)
    #     x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0)

    #     return x
    
    # def forward(self, x: torch.Tensor):  # default encoder
    #     x = x + self.pos_embedding
    #     x = self.dropout(x)

    #     x = reduce(lambda acc, layer: layer(acc), self.layers, x)
    #     x = self.ln(x)

    #     return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer as per https://arxiv.org/abs/2010.11929 with only difference being the use of Sparse Encoder
    instead of default one and bias removed from convolutional embedded dimension projection.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        self.conv_proj = nn.Conv2d(3, hidden_dim, patch_size, patch_size, bias=False)  # no bias for zero preservation

        seq_length = (image_size // patch_size) ** 2

        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1  # 1 for class token

        self.encoder = SparseEncoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, h, w = x.shape
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        seq_h, seq_w = h // self.patch_size, w // self.patch_size

        x = self.conv_proj(x)
        x = x.reshape(batch_size, self.hidden_dim, seq_h * seq_w)
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self._process_input(x)

        class_token = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_token, x], dim=1)

        x = self.encoder(x)
        x = self.heads(x[:, 0])

        return x
