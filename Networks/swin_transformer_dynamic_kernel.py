# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from Networks.channel_block import ChannelBlock
from Networks.layers.coordatten import CoordAtt



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class SELayer1D(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class Res_SE1D(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, use_se = True, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=4):
        super(Res_SE1D, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.use_se = use_se
        if use_se:
            self.se = SELayer1D(planes, reduction)
        self.downsample = downsample
        if inplanes != planes or stride > 1:
            self.downsample = nn.Sequential(nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm1d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



import torch
import torch.nn as nn
import scipy.fftpack as dct

# 定义 FECAM 层
class FECAMLayer(nn.Module):
    def __init__(self, channel):
        super(FECAMLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * 2, bias=False),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(channel * 2, channel, bias=False),
            nn.Sigmoid()
        )
        self.dct_norm = nn.LayerNorm(channel, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, L, C) => (B, C, L)
        b, c, l = x.size()  # (B, C, L)
        list = []

        for i in range(c):  # i represent channel
            # freq = torch.tensor(dct.dct(x[:, i, :].cpu().numpy())).to(x.device)  # dct
            freq = torch.tensor(dct.dct(x[:, i, :].detach().cpu().numpy())).to(x.device)  # dct
            list.append(freq)

        stack_dct = torch.stack(list, dim=1)
        stack_dct = self.dct_norm(stack_dct)
        f_weight = self.fc(stack_dct)
        f_weight = self.dct_norm(f_weight)

        result = x * f_weight
        return result.permute(0, 2, 1)  # (B, C, L) => (B, L, C)

# 定义基于 FECAM 的残差块
class Res_FECAM1D(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, use_fecam=True, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Res_FECAM1D, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.use_fecam = use_fecam
        if use_fecam:
            self.fecam = FECAMLayer(planes)
        self.downsample = downsample
        if inplanes != planes or stride > 1:
            self.downsample = nn.Sequential(nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm1d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_fecam:
            out = self.fecam(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



import threading
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


class SingletonMeta(type):
    """线程安全的单例元类"""
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        # 双检查锁定模式确保线程安全
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class DCT1D(nn.Module, metaclass=SingletonMeta):
    """使用元类实现的线程安全单例DCT模块"""

    def __init__(self, max_len=1024):
        super(DCT1D, self).__init__()
        self.max_len = max_len

        # 原子操作检查并注册缓冲区
        if not hasattr(self, 'dct_matrix'):
            dct_matrix = self._build_dct_matrix(max_len)
            self.register_buffer('dct_matrix', dct_matrix)
            print(f"DCT1D initialized with max_len={max_len}")

    def _build_dct_matrix(self, n):
        """构建离散余弦变换矩阵"""
        dct_m = np.zeros((n, n), dtype=np.float32)
        for k in range(n):
            for i in range(n):
                if k == 0:
                    dct_m[k, i] = 1.0 / np.sqrt(n)
                else:
                    dct_m[k, i] = np.cos(np.pi * k * (2 * i + 1) / (2 * n)) * np.sqrt(2.0 / n)
        return torch.from_numpy(dct_m)

    def forward(self, x):
        """应用DCT变换"""
        b, c, l = x.size()
        use_len = min(l, self.max_len)
        dct_matrix = self.dct_matrix[:use_len, :use_len]
        return torch.matmul(x, dct_matrix)




class FrequencyBandingReorganization(nn.Module):
    """TransMamba的频谱分带重组(SBR)实现"""

    def __init__(self, bands=4):
        super(FrequencyBandingReorganization, self).__init__()
        self.bands = bands

    def forward(self, x):
        b, c, l = x.size()
        assert c % self.bands == 0, "Channel must be divisible by bands"
        return rearrange(x, 'b (bn cbn) l -> b bn cbn l', bn=self.bands)  # b×bands×(c//bands)×l


class MultiScaleFrequencyAttention(nn.Module):
    """FMNet的多尺度频域注意力实现"""

    def __init__(self, bands=4):
        super(MultiScaleFrequencyAttention, self).__init__()
        self.conv3 = nn.Conv1d(bands, bands, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(bands, bands, kernel_size=5, padding=2)
        self.gate = nn.Sequential(
            nn.Conv1d(2 * bands, bands, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, bn, cbn, l = x.size()
        x_flat = x.reshape(b, bn, -1)  # b×bn×(cbn*l)
        x3 = self.conv3(x_flat).reshape(b, bn, cbn, l)
        x5 = self.conv5(x_flat).reshape(b, bn, cbn, l)
        gate_input = torch.cat([x3, x5], dim=1)  # b×2bn×cbn×l
        g = self.gate(gate_input.reshape(b, 2 * bn, -1)).reshape(b, bn, cbn, l)
        return g * x5 + (1 - g) * x3  # b×bn×cbn×l


class FusedFrequencySELayer1D(nn.Module):
    """使用最终版DCT1D的频域增强模块"""

    def __init__(self, channel, bands=4, reduction=4, max_dct_len=1024):
        super(FusedFrequencySELayer1D, self).__init__()
        self.channel = channel
        self.bands = bands
        self.dct = DCT1D(max_len=max_dct_len)  # 自动获取单例实例
        self.sbr = FrequencyBandingReorganization(bands=bands)
        self.ms_fa = MultiScaleFrequencyAttention(bands=bands)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, l = x.size()
        residual = x
        x_dct = self.dct(x)
        x_bands = self.sbr(x_dct)
        x_fused = self.ms_fa(x_bands)
        x_freq = rearrange(x_fused, 'b bn cbn l -> b (bn cbn) l')
        y = self.avg_pool(x_freq).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return residual * y.expand_as(residual)


class FusedRes_SE1D(nn.Module):
    """融合频域分带的残差模块"""

    # def __init__(self, inplanes, planes, stride=1, use_se=True, bands=4, reduction=4):
    def __init__(self, inplanes, planes, stride=1, use_se=True, bands=2, reduction=2):
        super(FusedRes_SE1D, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.use_se = use_se
        if use_se:
            self.se = FusedFrequencySELayer1D(planes, bands=bands, reduction=reduction)
        self.downsample = nn.Sequential(
            nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(planes)
        ) if inplanes != planes or stride > 1 else lambda x: x

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)
        out += residual
        out = self.relu(out)
        return out

# ----------------------------
# MixFFN模块（新增完整实现）
# ----------------------------
class MixFFN(nn.Module):
    def __init__(self, dim, num_kernels):
        super().__init__()
        self.proj_in = nn.Conv2d(dim, dim * 2, 1)
        self.conv1 = DyConv(dim, kernel_size=5, groups=dim, num_kernels=num_kernels)
        self.conv2 = DyConv(dim, kernel_size=7, groups=dim, num_kernels=num_kernels)
        self.proj_out = nn.Conv2d(dim * 2, dim, 1)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.act(self.proj_in(x))
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.act(self.conv1(x1)).unsqueeze(dim=2)
        x2 = self.act(self.conv2(x2)).unsqueeze(dim=2)
        x = torch.cat([x1, x2], dim=2)
        x = rearrange(x, 'b c g h w -> b (c g) h w')
        x = self.proj_out(x)
        x = x + shortcut
        return x


class DyConv(nn.Module):
    def __init__(self, dim, kernel_size, groups, num_kernels=1):
        super().__init__()
        if num_kernels > 1:
            self.conv = DynamicKernelAggregation(dim, kernel_size=kernel_size, groups=groups,
                                                 num_kernels=num_kernels)
        else:
            self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=groups, padding=kernel_size // 2)

    def forward(self, x):
        x = self.conv(x)
        return x


class DynamicKernelAggregation(nn.Module):
    def __init__(self, dim, kernel_size, groups=1, num_kernels=4):
        super().__init__()
        assert dim % groups == 0
        self.attention = KernelAttention(dim, num_kernels=num_kernels)
        self.aggregation = KernelAggregation(dim, kernel_size=kernel_size, groups=groups, num_kernels=num_kernels)

    def forward(self, x):
        attention = self.attention(x)
        x = self.aggregation(x, attention)
        return x


class KernelAttention(nn.Module):
    def __init__(self, dim, reduction=8, num_kernels=8):
        super().__init__()
        if dim != 3:
            mid_channels = dim // reduction
        else:
            mid_channels = num_kernels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(dim, mid_channels, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(mid_channels, num_kernels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.sigmoid(x)
        return x


class KernelAggregation(nn.Module):
    def __init__(self, dim, kernel_size, groups, num_kernels, bias=True, init_weight=True):
        super().__init__()
        self.groups = groups
        self.bias = bias
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.dim = dim
        self.weight = nn.Parameter(torch.randn(num_kernels, dim, dim // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_kernels, dim))
        else:
            self.bias = None

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.num_kernels):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x, attention):
        B, C, H, W = x.shape
        x = x.contiguous().view(1, B * self.dim, H, W)

        weight = self.weight.contiguous().view(self.num_kernels, -1)
        weight = torch.mm(attention, weight).contiguous().view(B * self.dim, self.dim // self.groups,
                                                               self.kernel_size, self.kernel_size)
        if self.bias is not None:
            bias = torch.mm(attention, self.bias).contiguous().view(-1)
            x = F.conv2d(x, weight=weight, bias=bias, stride=1, padding=self.kernel_size // 2,
                         groups=self.groups * B)
        else:
            x = F.conv2d(x, weight=weight, bias=None, stride=1, padding=self.kernel_size // 2,
                         groups=self.groups * B)
        x = x.contiguous().view(B, self.dim, x.shape[-2], x.shape[-1])

        return x


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# ----------------------------
# 修改后的Swin Transformer
# ----------------------------
class SwinTransformer_DK(nn.Module):
    r""" Swin Transformer with MixFFN integration for mode=3
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, mode=0, num_kernels=4,  # 新增num_kernels参数
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.mode = mode
        self.num_kernels = num_kernels  # MixFFN的kernel数量

        # 图像分块嵌入
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # 绝对位置嵌入
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 构建Stage层
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        # 针对mode=3新增MixFFN模块（位于最后一个Stage之后）
        if self.mode == 3:
            self.mixffn = MixFFN(dim=self.num_features, num_kernels=num_kernels)

        # 输出头配置
        if mode == 1 or mode == 2 or mode == 3:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.output1 = nn.Sequential(
                nn.Linear(1024, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, 1)
            )
        elif mode == 0:
            self.avgpool = nn.AdaptiveAvgPool1d(48)
            self.output1 = nn.Sequential(
                nn.Linear(6912, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, 1)
            )

        # mode=3的FusedRes_SE1D配置
        if self.mode == 3:
            self.post_conv = nn.Sequential(
                FusedRes_SE1D(self.num_features, 1024, stride=1),  # 使用self.num_features动态匹配通道数
            )
        elif self.mode == 2:
            self.post_conv = nn.Sequential(
                Res_SE1D(1024, 1024, stride=1, use_se=False),
            )

        if self.mode == 4:
            # 保持原有mode=4配置
            self.post_conv1 = nn.Sequential(Res_SE1D(1024, 1024, stride=1, use_se=False))
            self.avgpool1 = nn.AdaptiveAvgPool1d(1)
            self.output1 = nn.Sequential(nn.Linear(1024, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 12))
            self.post_conv2 = nn.Sequential(Res_SE1D(1024, 1024, stride=1, use_se=False))
            self.avgpool2 = nn.AdaptiveAvgPool1d(1)
            self.output2 = nn.Sequential(nn.Linear(1024, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        B = x.shape[0]  # 获取批次大小
        x = self.patch_embed(x)  # (B, L, C) L=H*W/P^2

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # 经过所有Stage
        for layer in self.layers:
            x = layer(x)

        # 标准化处理
        x = self.norm(x)  # (B, L, C) 其中C=self.num_features

        # 对mode=3: 插入MixFFN处理（最后一个Stage之后）
        if self.mode == 3:
            # 计算最后一个Stage输出的空间分辨率
            H_final = self.patches_resolution[0] // (2 ** (self.num_layers - 1))
            W_final = self.patches_resolution[1] // (2 ** (self.num_layers - 1))

            # (B, L, C) -> (B, H_final, W_final, C) -> (B, C, H_final, W_final) 适配MixFFN的2D输入
            x = x.view(B, H_final, W_final, -1)  # 重塑为空间维度
            x = x.permute(0, 3, 1, 2).contiguous()  # 转换为通道优先格式

            # 通过MixFFN增强特征
            x = self.mixffn(x)  # (B, C, H_final, W_final)

            # 转换回1D序列格式 (B, C, L) 其中L=H_final*W_final
            x = x.flatten(2)  # (B, C, H_final*W_final)

        # 不同mode的后续处理
        if self.mode == 0:
            x = self.avgpool(x)  # (B, L, 48)
        elif self.mode == 1:
            x = self.avgpool(x.transpose(1, 2))  # (B, C, 1)
        elif self.mode == 3:
            # MixFFN输出已为(B, C, L)，直接输入FusedRes_SE1D
            x = self.post_conv(x)  # (B, C, L)
            x = self.avgpool(x)  # (B, C, 1)
        elif self.mode == 2:
            x = self.post_conv(x.transpose(1, 2))  # (B, C, L) -> 处理
            x = self.avgpool(x)  # (B, C, 1)
        elif self.mode == 4:
            x1 = self.post_conv1(x.transpose(1, 2))
            x1 = self.avgpool1(x1)
            x1 = torch.flatten(x1, 1)
            x2 = self.post_conv2(x.transpose(1, 2))
            x2 = self.avgpool2(x2)
            x2 = torch.flatten(x2, 1)
            return x1, x2

        # 展平特征
        x = torch.flatten(x, 1)  # (B, C) 或 (B, L*48)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.mode == 4:
            x_cls = self.output1(x[0])
            x_cnt = self.output2(x[1] - x[0])
            return x_cnt, x_cls
        else:
            x_cnt = self.output1(x)
            return x_cnt

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        # 增加MixFFN的计算量（粗略估计）
        if self.mode == 3:
            H_final = self.patches_resolution[0] // (2 ** (self.num_layers - 1))
            W_final = self.patches_resolution[1] // (2 ** (self.num_layers - 1))
            flops += self.num_features * H_final * W_final * self.num_kernels * 2  # 近似值
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

