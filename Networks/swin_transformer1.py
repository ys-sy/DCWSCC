# --------------------------------------------------------
# Swin Transformer with MCFEA
# --------------------------------------------------------
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F

import threading
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

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

        trunc_normal_(self.relative_position_bias_table, std=.02)
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
        else:
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
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
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
        assert L == H * W, "input feature has wrong size"

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
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

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


class PatchEmbed(nn.Module):
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
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        B, C, H, W = x.shape
        assert C % self.groups == 0, "通道数必须能被分组数整除"
        x = x.view(B, self.groups, C // self.groups, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.view(B, -1, H, W)


class ParallelDepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups=2):
        super().__init__()
        self.groups = groups
        self.in_channels = in_channels  # 新增输入通道记录
        self.out_channels = out_channels

        self.conv3 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels
        )
        self.conv5 = nn.Conv2d(
            in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels
        )

        self.channel_shuffle = ChannelShuffle(groups)

        self.pointwise = nn.Conv2d(
            in_channels * 2, out_channels, kernel_size=1
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 新增输入通道验证
        assert x.shape[1] == self.in_channels, f"输入通道应为{self.in_channels}，实际为{x.shape[1]}"
        x3 = self.conv3(x)
        x5 = self.conv5(x)

        x3 = self.channel_shuffle(x3)
        x5 = self.channel_shuffle(x5)

        x_concat = torch.cat([x3, x5], dim=1)

        x_out = self.pointwise(x_concat)
        x_out = self.bn(x_out)
        x_out = self.relu(x_out)

        return x_out


class CrowdFeatureEnhancedModule(nn.Module):
    def __init__(self, dim, groups=2):
        super().__init__()
        self.dim = dim
        self.groups = groups

        self.fc = nn.Linear(dim // 2, dim // 2)
        self.pdcr = ParallelDepthwiseConv(dim // 2, dim // 2, groups)

        self.dwconv = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, groups=dim // 2)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

        x_split = torch.split(x, self.dim // 2, dim=1)
        X, Z = x_split[0], x_split[1]

        X_flatten = X.flatten(2).transpose(1, 2)
        X_linear = self.fc(X_flatten)

        X_linear_2d = X_linear.permute(0, 2, 1).reshape(B, self.dim // 2, H, W)
        Qm = self.pdcr(X_linear_2d)
        Km = self.pdcr(X_linear_2d)
        Vm = self.pdcr(X)

        attn = Qm * Km
        attn = self.dwconv(attn)
        attn = self.tanh(attn)

        attn = self.softmax(attn / math.sqrt(self.dim // 2))
        enhanced = attn * Vm
        enhanced = enhanced + Z

        output = torch.cat([enhanced, Z], dim=1)
        output = output.permute(0, 2, 3, 1)
        return output


import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleCrowdAggregator(nn.Module):
    def __init__(self, dims):
        """
        dims: list，表示各阶段的特征通道数（例如：[256, 512, 1024, 1024]）
        聚合时按从低分辨率（最后一层）向高分辨率（第一层）逐层上采样并融合。
        """
        super(MultiScaleCrowdAggregator, self).__init__()
        self.convs = nn.ModuleList()
        # 我们按照倒序来聚合：先聚合最后两层，再逐层与更高分辨率特征融合。
        for i in range(len(dims) - 1):
            # 假设要融合的两个特征分别来自高分辨率阶段 (dims[-2 - i]) 和低分辨率阶段 (dims[-1 - i])
            in_channels = dims[-2 - i] + dims[-1 - i]
            out_channels = dims[-2 - i]  # 输出通道保持较高分辨率阶段的通道数
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

    def forward(self, features):
        """
        features: list of tensor, 每个元素形状为 [B, C, H, W]，
                  顺序为：stage0（高分辨率） ... stageN（低分辨率）
        聚合时从最低分辨率开始往上融合，最后返回一个列表，顺序与 features 相同
        """
        aggregated = [features[-1]]  # 从最低分辨率开始（如 stage3）
        for i in range(len(features) - 1):
            target = features[-2 - i]  # 要融合的更高分辨率特征
            upsampled = F.interpolate(aggregated[-1], size=target.shape[2:], mode='bilinear', align_corners=True)
            if upsampled.shape[2:] != target.shape[2:]:
                upsampled = F.interpolate(upsampled, size=target.shape[2:], mode='bilinear', align_corners=True)
            combined = torch.cat([target, upsampled], dim=1)
            # 调试打印（可选）
            # print(f"[MSCA] 拼接通道数: {target.shape[1]} + {upsampled.shape[1]} = {combined.shape[1]}")
            agg = self.convs[i](combined)
            aggregated.append(agg)
        # 返回聚合结果列表，逆序还原到原始顺序（从 stage0 到 stageN）
        return aggregated[::-1]


import torch
import torch.nn as nn
import torch.nn.functional as F


class MCFEA(nn.Module):
    def __init__(self, embed_dims=[128, 256, 512, 512], groups=2):
        """
        embed_dims: 初始各阶段预设的通道数（例如：[128, 256, 512, 512]），
                    但加载预训练后会被更新为实际的输出通道（如 [256, 512, 1024, 1024]）
        groups: 用于 CFEM 内部的分组参数
        """
        super(MCFEA, self).__init__()
        self.embed_dims = embed_dims  # 此处记录初始配置（后面会根据实际情况更新）
        self.groups = groups
        self.cfem = nn.ModuleList([
            CrowdFeatureEnhancedModule(dim, groups) for dim in embed_dims
        ])
        self.msca = MultiScaleCrowdAggregator(embed_dims)

    def forward(self, features):
        """
        features: list of tensors，来自 Swin Transformer 各阶段特征，
                  每个元素 shape 为 [B, H, W, C]
        输出: 一个列表，列表内每个 tensor 的形状为 [B, C, H, W]，
              用于下游处理（例如 mode==3 分支中会取第一个尺度进行 permute）
        """
        enhanced_features = []
        rebuild_msca = False
        for i, feat in enumerate(features):
            in_channels = feat.shape[3]
            if in_channels != self.cfem[i].dim:
                print(f"[MCFEA] 阶段{i}通道数不一致: 原为{self.cfem[i].dim}, 实际为{in_channels}，已自动替换CFEM模块")
                # 动态替换为新的 CFEM 模块，确保内部各层按实际 in_channels 构造
                self.cfem[i] = CrowdFeatureEnhancedModule(in_channels, self.groups).to(feat.device)
                self.embed_dims[i] = in_channels
                rebuild_msca = True
            # CFEM 输出保持原有 [B, H, W, C]，这里再转为 [B, C, H, W]
            enhanced = self.cfem[i](feat)
            enhanced_features.append(enhanced.permute(0, 3, 1, 2))

        if rebuild_msca:
            # 重新构造 MSCA，使其各层的卷积输入输出通道依照最新 embed_dims
            self.msca = MultiScaleCrowdAggregator(self.embed_dims).to(features[0].device)
        # 将 enhanced_features 传入 MSCA，MSCA 返回的是一个列表，顺序为从 stage0（高分辨率）到最后
        aggregated_features = self.msca(enhanced_features)
        return aggregated_features


class SwinTransformer1(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1,
                 embed_dim=128, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, mode=0):
        print(f"[DEBUG] 当前 embed_dim: {embed_dim}")
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.mode = mode

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

        self.norm = norm_layer(self.num_features)

        # SwinTransformer1 中
        stage_dims = [embed_dim * (2 ** i) for i in range(len(depths))]
        # 若最后两层通道一样，确保保持一致
        if len(stage_dims) >= 4 and stage_dims[3] != stage_dims[2]:
            stage_dims[3] = stage_dims[2]
        print(f"[MCFEA] 正确通道配置: {stage_dims}")  # 应输出 [256, 512, 1024, 1024]

        self.mcfea = MCFEA(embed_dims=stage_dims, groups=2)

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


        if mode == 3:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.post_conv = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, 1)
            )

        elif self.mode == 2:
            self.post_conv = nn.Sequential(
                Res_SE1D(1024, 1024, stride=1, use_se=False),
            )

        if self.mode == 4:
            self.post_conv1 = nn.Sequential(
                Res_SE1D(1024, 1024, stride=1, use_se=False),
            )
            self.avgpool1 = nn.AdaptiveAvgPool1d(1)
            self.output1 = nn.Sequential(
                nn.Linear(1024, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, 12)
            )
            self.post_conv2 = nn.Sequential(
                Res_SE1D(1024, 1024, stride=1, use_se=False),
            )
            self.avgpool2 = nn.AdaptiveAvgPool1d(1)
            self.output2 = nn.Sequential(
                nn.Linear(1024, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, 1)
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

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        stage_features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            H = W = int(math.sqrt(x.shape[1]))
            stage_feat = x.view(x.shape[0], H, W, -1)
            stage_features.append(stage_feat)
            # 新增维度打印
            # print(f"[Swin Transformer] 阶段{i}特征形状: {stage_feat.shape}")

        # MCFEA前向传播
        if self.mode == 3:
            mcfea_features = self.mcfea(stage_features)
            x = mcfea_features[0]
            B, C, H, W = x.shape
            x = x.view(B, C, H * W)  # [24, 256, 2304]

            # 先池化展平
            x = self.avgpool(x)  # 输出 [24, 256, 1]
            x = torch.flatten(x, 1)  # 输出 [24, 256]

            # 再应用分类头
            x = self.post_conv(x)  # 输入形状正确

            # print(f"[DEBUG] 最终输出形状: {x.shape}")
            return x

        else:
            x = self.norm(x)
            if self.mode == 0:
                x = self.avgpool(x)
            elif self.mode == 1:
                x = self.avgpool(x.transpose(1, 2))
            elif self.mode == 2:
                x = self.post_conv(x.transpose(1, 2))
                x = self.avgpool(x)
            elif self.mode == 4:
                x1 = self.post_conv1(x.transpose(1, 2))
                x1 = self.avgpool1(x1)
                x1 = torch.flatten(x1, 1)
                x2 = self.post_conv2(x.transpose(1, 2))
                x2 = self.avgpool2(x2)
                x2 = torch.flatten(x2, 1)
                return x1, x2
            x = torch.flatten(x, 1)

        if self.mode == 4:
            x_cls = self.output1(x[0])
            x_cnt = self.output2(x[1] - x[0])
            return x_cnt, x_cls
        else:
            x_cnt = self.output1(x)
            return x_cnt


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

    def __init__(self, inplanes, planes, stride=1, use_se=True, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=4):
        super(Res_SE1D, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
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


# 原DCT1D类修改为动态版本
class DCT1D(nn.Module):
    def __init__(self):
        super(DCT1D, self).__init__()

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
        b, l, c = x.size()
        dct_matrix = self._build_dct_matrix(l).to(x.device)
        x = x.transpose(1, 2)  # [B, C, L]
        return torch.matmul(x, dct_matrix)  # [B, C, L]


class FrequencyBandingReorganization(nn.Module):
    def __init__(self, bands=4):
        super(FrequencyBandingReorganization, self).__init__()
        self.bands = bands

    def forward(self, x):
        b, c, l = x.size()
        assert c % self.bands == 0, "Channel must be divisible by bands"
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
        x_transposed = x.transpose(1, 2)  # [B, L, C]
        x_dct = self.dct(x_transposed)  # [B, C, L]
        x_bands = self.sbr(x_dct)
        x_fused = self.ms_fa(x_bands)

        # 修正后的维度重组
        x_freq = rearrange(x_fused, 'b bn cbn l -> b (bn cbn) l')
        print(f"[DEBUG] 重组后x_freq形状: {x_freq.shape}")  # 应输出 [24, 256, 2304]

        # 移除permute操作
        x_freq = self.avg_pool(x_freq)  # [B, 256, 1]
        print(f"[DEBUG] 池化后x_freq形状: {x_freq.shape}")  # 应输出 [24, 256, 1]

        y = x_freq.view(b, c)
        y = self.fc(y).view(b, c, 1)
        return residual * y.expand_as(residual)


class FusedRes_SE1D(nn.Module):
    def __init__(self, inplanes, planes, stride=1, use_se=True, bands=4, reduction=4):
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