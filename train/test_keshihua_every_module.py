import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torchvision.transforms as transforms
import torch.nn as nn
import threading
from timm.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torch import nn
import threading
from einops import rearrange

# -------------------------- 模型定义（复用原模型结构，修正SingletonMeta） --------------------------
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
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size  # 现在是元组 (H, W)
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # 关键修改：将 window_size 转换为最小值进行比较
        if min(self.input_resolution) <= min(self.window_size):  # 使用 min() 确保两边都是整数
            self.shift_size = 0
            self.window_size = (min(self.input_resolution), min(self.input_resolution))  # 保持元组类型

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
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

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "输入特征尺寸不匹配"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "输入特征尺寸不匹配"
        assert H % 2 == 0 and W % 2 == 0, "特征尺寸需为偶数"

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

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

        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x1 = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, x1


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], "输入图像尺寸不匹配"
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


# 修正后的SingletonMeta类，使用threading.Lock
class SingletonMeta(type):
    """线程安全的单例元类"""
    _instances = {}
    _lock = threading.Lock()  # 使用Python标准库的线程锁

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class DCT1D(nn.Module):
    def __init__(self):
        super(DCT1D, self).__init__()
        self.cache = {}

    def _build_dct_matrix(self, n):
        dct_m = np.zeros((n, n), dtype=np.float32)
        for k in range(n):
            for i in range(n):
                if k == 0:
                    dct_m[k, i] = 1.0 / np.sqrt(n)
                else:
                    dct_m[k, i] = np.cos(np.pi * k * (2 * i + 1) / (2 * n)) * np.sqrt(2.0 / n)
        return torch.from_numpy(dct_m)

    def forward(self, x):
        b, c, l = x.size()
        if l not in self.cache:
            dct_matrix = self._build_dct_matrix(l)
            self.cache[l] = dct_matrix.to(x.device)
        dct_matrix = self.cache[l]
        return torch.matmul(x, dct_matrix)


class FrequencyBandingReorganization(nn.Module):
    def __init__(self, bands=4):
        super(FrequencyBandingReorganization, self).__init__()
        self.bands = bands

    def forward(self, x):
        b, c, l = x.size()
        assert c % self.bands == 0, "通道数需能被分带数整除"
        return rearrange(x, 'b (bn cbn) l -> b bn cbn l', bn=self.bands)


class MultiScaleFrequencyAttention(nn.Module):
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
        x_flat = x.reshape(b, bn, -1)
        x3 = self.conv3(x_flat).reshape(b, bn, cbn, l)
        x5 = self.conv5(x_flat).reshape(b, bn, cbn, l)
        gate_input = torch.cat([x3, x5], dim=1)
        g = self.gate(gate_input.reshape(b, 2 * bn, -1)).reshape(b, bn, cbn, l)
        return g * x5 + (1 - g) * x3


class FusedFrequencySELayer1D(nn.Module):
    def __init__(self, channel, bands=4, reduction=4):
        super(FusedFrequencySELayer1D, self).__init__()
        self.channel = channel
        self.bands = bands
        self.dct = DCT1D()
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


# 主模型
class SwinTransformer_CDPNET3(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, mode=3, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.mode = mode

        if norm_layer is None:
            norm_layer = nn.LayerNorm

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

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

        # Regression模块定义
        class DWConv(nn.Module):
            def __init__(self, in_channels, kernel_size=3):
                super().__init__()
                self.conv = nn.Conv2d(
                    in_channels, in_channels, kernel_size,
                    padding=kernel_size // 2, groups=in_channels, bias=False
                )

            def forward(self, x):
                return self.conv(x)

        class ChannelShuffle(nn.Module):
            def __init__(self, groups):
                super().__init__()
                self.groups = groups

            def forward(self, x):
                batch, ch, h, w = x.shape
                ch_per_group = ch // self.groups
                x = x.view(batch, self.groups, ch_per_group, h, w)
                x = x.transpose(1, 2).contiguous()
                return x.view(batch, ch, h, w)

        class PWConv(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

            def forward(self, x):
                return self.conv(x)

        class MKMM(nn.Module):
            def __init__(self, in_channels, kernel_sizes=[3, 5], groups=2, norm_layer=nn.LayerNorm):
                super().__init__()
                self.split_ch = in_channels // 2
                self.dwconv1 = DWConv(self.split_ch, kernel_sizes[0])
                self.dwconv2 = DWConv(self.split_ch, kernel_sizes[1])
                self.shuffle = ChannelShuffle(groups)
                self.pwconv1 = PWConv(self.split_ch, self.split_ch)
                self.pwconv2 = PWConv(self.split_ch, self.split_ch)
                self.norm = norm_layer(in_channels) if norm_layer is not None else nn.Identity()

            def forward(self, x):
                x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                x1, x2 = torch.chunk(x, 2, dim=1)
                x1 = self.dwconv1(x1)
                x1 = self.shuffle(x1)
                x1 = self.pwconv1(x1)
                x2 = self.dwconv2(x2)
                x2 = self.shuffle(x2)
                x2 = self.pwconv2(x2)
                return torch.cat([x1, x2], dim=1)

        class MFEM(nn.Module):
            def __init__(self, in_channels):
                super().__init__()
                self.lp = nn.Conv2d(in_channels, in_channels, 1, bias=False)
                self.dwconv_q = DWConv(in_channels, kernel_size=3)
                self.dwconv_k = DWConv(in_channels, kernel_size=3)
                self.dwconv_v = DWConv(in_channels, kernel_size=3)
                self.dwconv_a = DWConv(in_channels, kernel_size=3)
                self.softmax = nn.Softmax(dim=1)
                self.lp_out = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)

            def forward(self, m):
                m_proj = self.lp(m)
                q = self.dwconv_q(m_proj)
                k = self.dwconv_k(m_proj)
                v = self.dwconv_v(m)
                qk = q * k
                attn = torch.tanh(self.dwconv_a(qk))
                attn = self.softmax(attn)
                attn_v = attn * v
                fused = torch.cat([attn_v, m], dim=1)
                return self.lp_out(fused)

        class CGFIM_v2(nn.Module):
            def __init__(self, in_channels_list, norm_layer=nn.LayerNorm):
                super().__init__()
                self.mkmm = nn.ModuleList([MKMM(ch, norm_layer=norm_layer) for ch in in_channels_list])
                self.downsample_x1 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
                self.upsample_x2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.upsample_x4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
                self.conv_adjust = nn.ModuleList([nn.Conv2d(ch, 128, 1, bias=False) for ch in in_channels_list])

            def forward(self, x0, x1, x2, x3):
                s0 = self.mkmm[0](x0)
                s1 = self.mkmm[1](x1)
                s2 = self.mkmm[2](x2)
                s3 = self.mkmm[3](x3)

                s0_down = self.downsample_x1(s0)
                s2_up = self.upsample_x2(s2)
                s3_up = self.upsample_x4(s3)

                s0_adj = self.conv_adjust[0](s0_down)
                s1_adj = self.conv_adjust[1](s1)
                s2_adj = self.conv_adjust[2](s2_up)
                s3_adj = self.conv_adjust[3](s3_up)

                return torch.cat([s0_adj, s1_adj, s2_adj, s3_adj], dim=1)

        class Regression1(nn.Module):
            def __init__(self, norm_layer=nn.LayerNorm):
                super(Regression1, self).__init__()
                self.cgfim = CGFIM_v2(in_channels_list=[128, 256, 512, 1024], norm_layer=norm_layer)
                self.mfem = MFEM(in_channels=512)
                self.stage1 = nn.Sequential(
                    nn.Conv2d(512, 256, 3, padding=1, dilation=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                )
                self.stage2 = nn.Sequential(
                    nn.Conv2d(512, 256, 3, padding=2, dilation=2),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                )
                self.stage3 = nn.Sequential(
                    nn.Conv2d(512, 256, 3, padding=3, dilation=3),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                )
                self.stage4 = nn.Sequential(
                    nn.Conv2d(512, 256 * 3, 1),
                    nn.BatchNorm2d(256 * 3),
                    nn.ReLU(inplace=True)
                )
                self.out_adjust = nn.Sequential(
                    nn.Conv2d(768, 1024, 1, bias=False),
                    nn.BatchNorm2d(1024),
                    nn.ReLU(inplace=True)
                )
                self.init_param()

            def forward(self, x0, x1, x2, x3):
                fused_cgfim = self.cgfim(x0, x1, x2, x3)
                enhanced = self.mfem(fused_cgfim)
                x = enhanced
                y1 = self.stage1(x)
                y2 = self.stage2(x)
                y3 = self.stage3(x)
                y4 = self.stage4(x)
                y = torch.cat((y1, y2, y3), dim=1) + y4
                output = self.out_adjust(y)
                return output

            def init_param(self):
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.normal_(m.weight, std=0.01)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

        self.regression = Regression1(norm_layer=norm_layer)
        self.norm = norm_layer(self.num_features)
        self.post_conv = nn.Sequential(
            FusedRes_SE1D(1024, 1024, stride=1),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.output1 = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features_with_features(self, x):
        """提取并返回中间特征用于可视化"""
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # 收集各阶段特征
        x1 = x2 = x3 = x4 = None
        for i, layer in enumerate(self.layers):
            x, x11 = layer(x)  # x11是下采样前特征
            B1, L1, C1 = x11.shape
            H1 = W1 = int(L1 ** 0.5)
            xf1 = x11.view(B1, C1, H1, W1)  # 转换为(B, C, H, W)
            if i == 0:
                x1 = xf1
            elif i == 1:
                x2 = xf1
            elif i == 2:
                x3 = xf1
            elif i == 3:
                x4 = xf1

        # 确保所有特征都被正确初始化
        assert x1 is not None and x2 is not None and x3 is not None and x4 is not None, "特征提取失败"

        # 回归模块输出特征
        reg_feat = self.regression(x1, x2, x3, x4)

        # post_conv输出特征
        B, C, H, W = reg_feat.shape
        x = reg_feat.flatten(2)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        post_conv_feat = self.post_conv(x)
        post_conv_feat = post_conv_feat.view(B, C, H, W)

        return x4, reg_feat, post_conv_feat

    def forward(self, x, return_features=False):
        if return_features:
            return self.forward_features_with_features(x)
        else:
            x = self.forward_features(x)
            x_cnt = self.output1(x)
            return x_cnt


# 修改 base_patch16_384_swin_cdpnet3 函数中的 window_size 参数
def base_patch16_384_swin_cdpnet3(pretrained=False, mode=3, **kwargs):
    """创建模型实例"""
    model = SwinTransformer_CDPNET3(
        img_size=384,
        patch_size=4,
        in_chans=3,
        num_classes=1,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=(12, 12),  # 将 12 改为 (12, 12) 元组形式
        mlp_ratio=4.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        mode=mode
    )
    if pretrained:
        try:
            checkpoint = torch.load('swin_base_patch4_window12_384.pth', map_location='cpu')
            model.load_state_dict(checkpoint["model"], strict=False)
            print("成功加载预训练权重")
        except:
            print("未找到预训练权重，使用随机初始化")
    return model


# -------------------------- 2. 图像预处理 --------------------------
def preprocess_image(img_path, target_size=(384, 384)):
    """预处理图像并保留原始图像"""
    # 读取原始图像
    original_img = cv2.imread(img_path)
    if original_img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

    # 预处理为模型输入
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet均值
            std=[0.229, 0.224, 0.225]  # ImageNet标准差
        )
    ])
    img_tensor = transform(original_img).unsqueeze(0)  # 添加batch维度
    return original_img, img_tensor


# -------------------------- 3. 热力图可视化核心函数 --------------------------
def visualize_heatmap(
        original_img,
        feature_map,
        title,
        target_size=(384, 384),
        colormap="jet",  # 热力图颜色方案
        alpha=0.6,  # 热力图透明度
        save_path=None,
        show_colorbar=True  # 显示颜色条
):
    """
    生成高质量特征热力图
    """
    # 1. 处理特征图 - 计算通道平均值作为热力图基础
    feat = feature_map[0].detach().cpu().numpy()  # 形状: (C, H, W)
    heatmap = np.mean(feat, axis=0)  # 通道平均: (H, W)

    # 2. 归一化到[0, 1]范围
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # 3. 上采样到目标尺寸，使用高质量插值
    heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_CUBIC)

    # 4. 调整原图尺寸
    original_resized = cv2.resize(original_img, target_size)

    # 5. 创建可视化图像
    plt.figure(figsize=(14, 6))

    # 左侧：原图
    plt.subplot(131)
    plt.imshow(original_resized)
    plt.title("Original Image", fontsize=14)
    plt.axis("off")

    # 中间：纯热力图
    plt.subplot(132)
    hm = plt.imshow(heatmap, cmap=colormap)
    plt.title("Feature Heatmap", fontsize=14)
    plt.axis("off")
    if show_colorbar:
        plt.colorbar(hm, fraction=0.046, pad=0.04)  # 添加颜色条

    # 右侧：叠加热力图的原图
    plt.subplot(133)
    plt.imshow(original_resized)
    plt.imshow(heatmap, cmap=colormap, alpha=alpha)  # 半透明叠加
    plt.title(title, fontsize=14)
    plt.axis("off")

    plt.tight_layout()

    # 保存图像（可选）
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


# -------------------------- 4. 多通道热力图对比函数 --------------------------
def visualize_multi_channel_heatmaps(
        original_img,
        feature_map,
        title,
        target_size=(384, 384),
        num_channels=5,  # 显示的通道数
        colormap="jet",
        save_path=None
):
    """可视化多个通道的热力图对比"""
    feat = feature_map[0].detach().cpu().numpy()  # (C, H, W)

    # 随机选择num_channels个通道（或取前num_channels个）
    selected_channels = min(num_channels, feat.shape[0])
    channels = np.linspace(0, feat.shape[0] - 1, selected_channels, dtype=int)

    # 创建子图
    fig, axes = plt.subplots(1, selected_channels + 1, figsize=(4 * (selected_channels + 1), 4))

    # 显示原图
    axes[0].imshow(cv2.resize(original_img, target_size))
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    # 显示每个通道的热力图
    for i, ch in enumerate(channels, 1):
        heatmap = feat[ch]
        # 归一化
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        # 上采样
        heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_CUBIC)
        # 显示
        hm = axes[i].imshow(heatmap, cmap=colormap)
        axes[i].set_title(f"Channel {ch}", fontsize=12)
        axes[i].axis("off")

    # 添加一个共用的颜色条
    cbar = fig.colorbar(hm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05)
    cbar.set_label('Feature Activation', fontsize=12)

    plt.suptitle(title, fontsize=16, y=0.95)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


# -------------------------- 5. 主程序 --------------------------
def main(img_path, checkpoint_path=None, save_figures=False):
    # 1. 配置参数
    target_size = (384, 384)  # 与模型输入尺寸一致

    # 2. 加载模型
    model = base_patch16_384_swin_cdpnet3(pretrained=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 3. 加载权重（如果提供）
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"成功加载模型权重: {checkpoint_path}")

    model.eval()  # 切换到评估模式

    # 4. 预处理图像
    original_img, img_tensor = preprocess_image(img_path, target_size)
    img_tensor = img_tensor.to(device)

    # 5. 提取特征
    with torch.no_grad():  # 关闭梯度计算
        x4_feat, reg_feat, post_conv_feat = model(img_tensor, return_features=True)

    # 6. 可视化热力图
    # 定义不同的颜色方案以区分不同层
    colormaps = {
        "4th Stage Features": "jet",
        "Regression Module Output": "viridis",
        "Post Conv Features": "plasma"
    }

    # 单通道聚合热力图
    features = [
        (x4_feat, "4th Stage Features", "stage4_heatmap.png"),
        (reg_feat, "Regression Module Output", "regression_heatmap.png"),
        (post_conv_feat, "Post Conv Features", "post_conv_heatmap.png")
    ]

    for feat, title, save_name in features:
        save_path = save_name if save_figures else None
        visualize_heatmap(
            original_img=original_img,
            feature_map=feat,
            title=title,
            target_size=target_size,
            colormap=colormaps[title],
            alpha=0.6,
            save_path=save_path
        )

    # 多通道热力图对比（仅对第一个特征层）
    visualize_multi_channel_heatmaps(
        original_img=original_img,
        feature_map=x4_feat,
        title="4th Stage - Multi-channel Heatmaps",
        target_size=target_size,
        num_channels=6,
        save_path="multi_channel_heatmap.png" if save_figures else None
    )


if __name__ == '__main__':
    # 配置路径
    IMAGE_PATH = "/amax/zs/code/WSCC_TAF-main/datasets/ShanghaiTech/part_A_final/test_data/images/IMG_156.jpg"
    CHECKPOINT_PATH = "/amax/zs/code/WSCC_TAF-main/save_file/ShanghaiA_swincdpnet3/model_best.pth"
    SAVE_FIGURES = False  # 是否保存可视化结果

    main(IMAGE_PATH, CHECKPOINT_PATH, SAVE_FIGURES)
