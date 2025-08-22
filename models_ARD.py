# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import copy
import math
from time import time

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp, PatchEmbed

## Attention Mask

try:
    from torch.nn.functional import (
        scaled_dot_product_attention as slow_attn,
    )  # q, k, v: BHLc
except ImportError:

    def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
        attn = query.mul(scale) @ key.transpose(-2, -1)  # BHLc @ BHcL => BHLL
        if attn_mask is not None:
            attn.add_(attn_mask)
        attn.softmax(dim=-1)
        return (
            F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True)
            if dropout_p > 0
            else attn.softmax(dim=-1)
        ) @ value


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None

    def kv_caching(self, enable: bool):
        self.caching, self.cached_k, self.cached_v = enable, None, None

    def kv_detach(self):
        self.cached_k = self.cached_k.detach()
        self.cached_v = self.cached_v.detach()

    def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )  # (3, B, self.num_heads, N, self.head_dim)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                k = self.cached_k = torch.cat((self.cached_k, k), dim=2)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=2)

        dropout_p = self.attn_drop if self.training else 0.0

        x = slow_attn(
            query=q,
            key=k,
            value=v,
            scale=self.scale,
            attn_mask=attn_bias,
            dropout_p=dropout_p,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)

        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, attn_bias=None):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(
            6, dim=2
        )
        result = self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_bias)
        x = x + gate_msa * result
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class LastLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, hidden_size2):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, hidden_size2, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(1152, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.0,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, 2, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        num_patches = 256
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, 2, self.out_channels)
        self.final_layer2 = FinalLayer(hidden_size, 2, self.out_channels)
        self.final_layer3 = FinalLayer(hidden_size, 2, self.out_channels)
        self.final_layer4 = FinalLayer(hidden_size, 2, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        wT = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(wT.view([wT.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        nn.init.constant_(self.final_layer2.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer2.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer2.linear.weight, 0)
        nn.init.constant_(self.final_layer2.linear.bias, 0)

        nn.init.constant_(self.final_layer3.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer3.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer3.linear.weight, 0)
        nn.init.constant_(self.final_layer3.linear.bias, 0)

        nn.init.constant_(self.final_layer4.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer4.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer4.linear.weight, 0)
        nn.init.constant_(self.final_layer4.linear.bias, 0)

    def copy_weights(self):
        self.final_layer2 = copy.deepcopy(self.final_layer)
        self.final_layer3 = copy.deepcopy(self.final_layer)
        self.final_layer4 = copy.deepcopy(self.final_layer)


    def unpatchifyT(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = 2
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs


    def forward(self, z5, z10, z15, zT, t, n, y):  ## attn_bias is applied on training
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        e_1 = self.x_embedder(zT) + self.pos_embed
        e_2 = self.x_embedder(z15) + self.pos_embed
        e_3 = self.x_embedder(z10) + self.pos_embed
        e_4 = self.x_embedder(z5) + self.pos_embed

        x = torch.cat((e_1, e_2, e_3, e_4), dim=1)  # (N, T, D)

        ## Token-wise time embedding
        t1 = self.t_embedder(t)  # (N, D)
        t1 = torch.unsqueeze(t1, 1)
        t1 = t1.repeat(1, 256, 1)

        t2 = self.t_embedder(t * (3 / 4))
        t2 = torch.unsqueeze(t2, 1)
        t2 = t2.repeat(1, 256, 1)

        t3 = self.t_embedder(t * (2 / 4))
        t3 = torch.unsqueeze(t3, 1)
        t3 = t3.repeat(1, 256, 1)

        t4 = self.t_embedder(t * (1 / 4))
        t4 = torch.unsqueeze(t4, 1)
        t4 = t4.repeat(1, 256, 1)

        t = torch.cat((t1, t2, t3, t4), dim=1)  # (N, T, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        y = torch.unsqueeze(y, 1)
        y = y.repeat(1, 256 + 256 + 256 + 256, 1)
        c = t + y

        patch_nums = (16, 16, 16, 16)
        d: torch.Tensor = torch.cat(
            [torch.full((pn * pn,), i) for i, pn in enumerate(patch_nums)]
        ).view(1, 256 + 256 + 256 + 256, 1)
        dT = d.transpose(1, 2)
        attn_bias = (
            torch.where(d >= dT, 0.0, -torch.inf)
            .reshape(1, 1, 256 + 256 + 256 + 256, 256 + 256 + 256 + 256)
            .to(zT.device)
        )
        attn_bias_top = (
            torch.where(d == dT, 0.0, -torch.inf)
            .reshape(1, 1, 256 + 256 + 256 + 256, 256 + 256 + 256 + 256)
            .to(zT.device)
        )

        cnt = 0
        for block in self.blocks:
            if cnt < n:
                x = block(x, c, attn_bias=attn_bias)
            else:
                x = block(x, c, attn_bias=attn_bias_top)
            cnt += 1

        pred_z15 = self.final_layer(x[:, :256], c[:, :256])  # (N, T, patch_size ** 2 * out_channels)
        pred_z15 = self.unpatchifyT(pred_z15)  # (N, out_channels, H, W)

        pred_z10 = self.final_layer2(x[:, 256 : 256 + 256], c[:, 256 : 256 + 256])
        pred_z10 = self.unpatchifyT(pred_z10)

        pred_z5 = self.final_layer3(x[:, 256 + 256 : 256 + 256 + 256], c[:, 256 + 256 : 256 + 256 + 256])
        pred_z5 = self.unpatchifyT(pred_z5)

        pred_z0 = self.final_layer4(x[:, 256 + 256 + 256 :], c[:, 256 + 256 + 256 :])
        pred_z0 = self.unpatchifyT(pred_z0)

        return pred_z15, pred_z10, pred_z5, pred_z0

    def infer(self, zT, t, n, y):
        with torch.no_grad():
            batch_size = zT.shape[0]
            B, C = zT.shape[:2]

            if n != 0:
                for i in range(n):
                    self.blocks[i].attn.kv_caching(True)

            t1234 = self.t_embedder(torch.cat([t, t * 3 / 4, t * 2 / 4, t * 1 / 4]))
            y = self.y_embedder(y, self.training)

            c = t1234[:batch_size] + y
            c = torch.unsqueeze(c, 1)
            c = c.repeat(1, 256, 1)

            e_1 = self.x_embedder(zT) + self.pos_embed

            for block in self.blocks:
                e_1 = block(e_1, c)

            pred_z15 = self.final_layer(e_1, c)
            pred_z15 = self.unpatchifyT(pred_z15)
            pred_z15, _ = torch.split(pred_z15, C, dim=1)

            c = t1234[batch_size : batch_size * 2] + y
            c = torch.unsqueeze(c, 1)
            c = c.repeat(1, 256, 1)
            e_2 = self.x_embedder(pred_z15) + self.pos_embed

            for block in self.blocks:
                e_2 = block(e_2, c)

            pred_z10 = self.final_layer2(e_2, c)
            pred_z10 = self.unpatchifyT(pred_z10)
            pred_z10, _ = torch.split(pred_z10, C, dim=1)

            c = t1234[batch_size * 2 : batch_size * 3] + y
            c = torch.unsqueeze(c, 1)
            c = c.repeat(1, 256, 1)
            e_3 = self.x_embedder(pred_z10) + self.pos_embed

            for block in self.blocks:
                e_3 = block(e_3, c)

            pred_z5 = self.final_layer3(e_3, c)
            pred_z5 = self.unpatchifyT(pred_z5)
            pred_z5, _ = torch.split(pred_z5, C, dim=1)

            c = t1234[batch_size * 3 : batch_size * 4] + y
            c = torch.unsqueeze(c, 1)
            c = c.repeat(1, 256, 1)
            e_4 = self.x_embedder(pred_z5) + self.pos_embed

            for block in self.blocks:
                e_4 = block(e_4, c)

            pred_z0 = self.final_layer4(e_4, c)
            pred_z0 = self.unpatchifyT(pred_z0)
            pred_z0, _ = torch.split(pred_z0, C, dim=1)

            for b in self.blocks:
                b.attn.kv_caching(False)
            return pred_z15, pred_z10, pred_z5, pred_z0



#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if grid_size == 16:
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
    elif grid_size == 8:
        grid_h = np.arange(grid_size, dtype=np.float32) * 2 + 1
        grid_w = np.arange(grid_size, dtype=np.float32) * 2 + 1
    elif grid_size == 4:
        grid_h = np.arange(grid_size, dtype=np.float32) * 4 + 2
        grid_w = np.arange(grid_size, dtype=np.float32) * 4 + 2
    elif grid_size == 2:
        grid_h = np.arange(grid_size, dtype=np.float32) * 8 + 4
        grid_w = np.arange(grid_size, dtype=np.float32) * 8 + 4

    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################


def DiT_XL_2(**kwargs):
    return DiT(
        depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs
    )  # , do_last=True


def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_B_16(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=16, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    "DiT-XL/2": DiT_XL_2,
    "DiT-XL/4": DiT_XL_4,
    "DiT-XL/8": DiT_XL_8,
    "DiT-L/2": DiT_L_2,
    "DiT-L/4": DiT_L_4,
    "DiT-L/8": DiT_L_8,
    "DiT-B/2": DiT_B_2,
    "DiT-B/4": DiT_B_4,
    "DiT-B/8": DiT_B_8,
    "DiT-S/2": DiT_S_2,
    "DiT-S/4": DiT_S_4,
    "DiT-S/8": DiT_S_8,
}
