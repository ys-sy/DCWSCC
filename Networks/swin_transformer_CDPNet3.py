import math
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
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
    """将特征图分割为窗口"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """将窗口合并回特征图"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """窗口注意力模块"""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # 计算相对位置索引
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

        # 添加相对位置偏置
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
    """Swin Transformer基础块"""

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

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 计算移位窗口的注意力掩码
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

        # 循环移位
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 分割窗口
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # 窗口注意力
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # 逆移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """Patch合并层"""

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
    """Swin Transformer阶段层"""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # 构建Transformer块
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

        # 下采样层
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x1 = x  # 保存下采样前的特征用于中间特征提取
        if self.downsample is not None:
            x = self.downsample(x)
        return x, x1  # 返回下采样后特征和下采样前特征


class PatchEmbed(nn.Module):
    """图像到Patch的嵌入层"""

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
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], "输入图像尺寸不匹配"
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


# -------------------------- mode=3 依赖的频域增强模块 --------------------------
class SingletonMeta(type):
    """线程安全的单例元类"""
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class DCT1D(nn.Module):
    """动态DCT模块，根据输入长度生成对应矩阵"""

    def __init__(self):
        super(DCT1D, self).__init__()
        self.cache = {}  # 缓存不同长度的DCT矩阵，避免重复计算

    def _build_dct_matrix(self, n):
        """构建n×n的离散余弦变换矩阵"""
        dct_m = np.zeros((n, n), dtype=np.float32)
        for k in range(n):
            for i in range(n):
                if k == 0:
                    dct_m[k, i] = 1.0 / np.sqrt(n)
                else:
                    dct_m[k, i] = np.cos(np.pi * k * (2 * i + 1) / (2 * n)) * np.sqrt(2.0 / n)
        return torch.from_numpy(dct_m)

    def forward(self, x):
        """应用DCT变换，根据输入长度动态调整矩阵"""
        b, c, l = x.size()

        # 从缓存获取或生成对应长度的DCT矩阵
        if l not in self.cache:
            dct_matrix = self._build_dct_matrix(l)
            self.cache[l] = dct_matrix.to(x.device)  # 移到与输入相同的设备
        dct_matrix = self.cache[l]

        # 执行矩阵乘法（确保维度匹配：[b, c, l] × [l, l] → [b, c, l]）
        return torch.matmul(x, dct_matrix)


class FrequencyBandingReorganization(nn.Module):
    """频谱分带重组"""

    def __init__(self, bands=4):
        super(FrequencyBandingReorganization, self).__init__()
        self.bands = bands

    def forward(self, x):
        b, c, l = x.size()
        assert c % self.bands == 0, "通道数需能被分带数整除"
        return rearrange(x, 'b (bn cbn) l -> b bn cbn l', bn=self.bands)  # b×bands×(c//bands)×l


class MultiScaleFrequencyAttention(nn.Module):
    """多尺度频域注意力"""

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
        return g * x5 + (1 - g) * x3  # 融合多尺度特征


class FusedFrequencySELayer1D(nn.Module):
    """频域增强SE层（使用修改后的DCT1D）"""
    def __init__(self, channel, bands=4, reduction=4):  # 移除max_dct_len参数
        super(FusedFrequencySELayer1D, self).__init__()
        self.channel = channel
        self.bands = bands
        self.dct = DCT1D()  # 使用修改后的DCT1D，无需预设max_len
        self.sbr = FrequencyBandingReorganization(bands=bands)
        self.ms_fa = MultiScaleFrequencyAttention(bands=bands)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    # forward方法保持不变
    def forward(self, x):
        b, c, l = x.size()
        residual = x
        x_dct = self.dct(x)  # 频域变换（现在维度会自动匹配）
        x_bands = self.sbr(x_dct)  # 分带重组
        x_fused = self.ms_fa(x_bands)  # 多尺度频域注意力
        x_freq = rearrange(x_fused, 'b bn cbn l -> b (bn cbn) l')  # 重组为原始通道维度
        y = self.avg_pool(x_freq).view(b, c)  # 通道注意力权重
        y = self.fc(y).view(b, c, 1)
        return residual * y.expand_as(residual)  # 应用注意力权重


class FusedRes_SE1D(nn.Module):
    """融合频域分带的残差模块 (mode=3核心模块)"""

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
        # 下采样层
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
            out = self.se(out)  # 应用频域增强注意力
        out += residual
        out = self.relu(out)
        return out


# -------------------------- Swin Transformer 主网络 (保留mode=3) --------------------------
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
        self.mode = mode  # 固定为mode=3

        # Patch嵌入
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

        # 随机深度衰减
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 构建Swin Transformer层
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

        # -------------------------- Regression1 相关模块 --------------------------
        class DWConv(nn.Module):
            """深度可分离卷积"""

            def __init__(self, in_channels, kernel_size=3):
                super().__init__()
                self.conv = nn.Conv2d(
                    in_channels, in_channels, kernel_size,
                    padding=kernel_size // 2, groups=in_channels, bias=False
                )

            def forward(self, x):
                return self.conv(x)

        class ChannelShuffle(nn.Module):
            """通道打乱"""

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
            """逐点卷积（1x1卷积）"""

            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

            def forward(self, x):
                return self.conv(x)

        class MKMM(nn.Module):
            """多核混合模块"""

            def __init__(self, in_channels, kernel_sizes=[3, 5], groups=2, norm_layer=nn.LayerNorm):
                super().__init__()
                self.split_ch = in_channels // 2
                self.dwconv1 = DWConv(self.split_ch, kernel_sizes[0])
                self.dwconv2 = DWConv(self.split_ch, kernel_sizes[1])
                self.shuffle = ChannelShuffle(groups)
                self.pwconv1 = PWConv(self.split_ch, self.split_ch)
                self.pwconv2 = PWConv(self.split_ch, self.split_ch)
                # 在初始化时创建LayerNorm层，归一化通道维度（输入permute后最后一维是通道）
                self.norm = norm_layer(in_channels)  # 关键修改：初始化LayerNorm

            def forward(self, x):
                # 使用初始化的self.norm替代动态创建的nn.LayerNorm
                x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # 关键修改
                x1, x2 = torch.chunk(x, 2, dim=1)
                x1 = self.dwconv1(x1)
                x1 = self.shuffle(x1)
                x1 = self.pwconv1(x1)
                x2 = self.dwconv2(x2)
                x2 = self.shuffle(x2)
                x2 = self.pwconv2(x2)
                return torch.cat([x1, x2], dim=1)

        class MFEM(nn.Module):
            """多感受野增强模块"""

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
            """适配4阶段特征的粗粒度交互模块"""

            def __init__(self, in_channels_list, norm_layer=nn.LayerNorm):  # 添加norm_layer参数
                super().__init__()
                # 创建MKMM时传入norm_layer
                self.mkmm = nn.ModuleList([MKMM(ch, norm_layer=norm_layer) for ch in in_channels_list])
                self.downsample_x1 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
                self.upsample_x2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.upsample_x4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
                self.conv_adjust = nn.ModuleList([nn.Conv2d(ch, 128, 1, bias=False) for ch in in_channels_list])

            def forward(self, x0, x1, x2, x3):
                s0 = self.mkmm[0](x0)  # (B, 128, 96, 96)
                s1 = self.mkmm[1](x1)  # (B, 256, 48, 48)
                s2 = self.mkmm[2](x2)  # (B, 512, 24, 24)
                s3 = self.mkmm[3](x3)  # (B, 1024, 12, 12)

                s0_down = self.downsample_x1(s0)  # (B, 128, 48, 48)
                s2_up = self.upsample_x2(s2)  # (B, 512, 48, 48)
                s3_up = self.upsample_x4(s3)  # (B, 1024, 48, 48)

                s0_adj = self.conv_adjust[0](s0_down)  # (B, 128, 48, 48)
                s1_adj = self.conv_adjust[1](s1)  # (B, 128, 48, 48)
                s2_adj = self.conv_adjust[2](s2_up)  # (B, 128, 48, 48)
                s3_adj = self.conv_adjust[3](s3_up)  # (B, 128, 48, 48)

                return torch.cat([s0_adj, s1_adj, s2_adj, s3_adj], dim=1)  # (B, 512, 48, 48)

        class Regression1(nn.Module):
            def __init__(self, norm_layer=nn.LayerNorm):  # 添加norm_layer参数
                super(Regression1, self).__init__()
                # 粗粒度特征交互模块（适配4阶段特征），传入norm_layer
                self.cgfim = CGFIM_v2(in_channels_list=[128, 256, 512, 1024], norm_layer=norm_layer)

                # 多感受野增强模块（处理融合后的特征）
                self.mfem = MFEM(in_channels=512)

                # 特征预处理层（对应原v0-v3，统一特征尺度）
                self.v0 = nn.Sequential(
                    nn.Identity()  # 已通过CGFIM处理x0，此处保留格式
                )
                self.v1 = nn.Sequential(
                    nn.Identity()  # 已通过CGFIM处理x1，此处保留格式
                )
                self.v2 = nn.Sequential(
                    nn.Identity()  # 已通过CGFIM处理x2，此处保留格式
                )
                self.v3 = nn.Sequential(
                    nn.Identity()  # 已通过CGFIM处理x3，此处保留格式
                )

                # 多尺度卷积融合层（对应原stage1-stage4）
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

                # 输出通道调整（适配后续处理）
                self.out_adjust = nn.Sequential(
                    nn.Conv2d(768, 1024, 1, bias=False),
                    nn.BatchNorm2d(1024),
                    nn.ReLU(inplace=True)
                )

                self.init_param()

            def forward(self, x0, x1, x2, x3):
                # 1. 粗粒度特征交互（跨阶段融合4个特征）
                fused_cgfim = self.cgfim(x0, x1, x2, x3)  # 输出: (B, 512, 48, 48)

                # 2. 多感受野增强（细化融合特征）
                enhanced = self.mfem(fused_cgfim)  # 输出: (B, 512, 48, 48)

                # 3. 特征融合（对应原x1+x2+x3+x0格式）
                x = enhanced  # 融合后特征作为基础输入

                # 4. 多尺度卷积分支处理
                y1 = self.stage1(x)
                y2 = self.stage2(x)
                y3 = self.stage3(x)
                y4 = self.stage4(x)

                # 5. 特征拼接与残差融合
                y = torch.cat((y1, y2, y3), dim=1) + y4  # 输出: (B, 768, 48, 48)

                # 6. 通道调整适配后续网络
                output = self.out_adjust(y)  # 输出: (B, 384, 48, 48)
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

        # 实例化Regression1模块
        self.regression = Regression1(norm_layer=norm_layer)

        # -------------------------- 原有模块继续 --------------------------
        self.norm = norm_layer(self.num_features)

        # 频域增强后处理模块
        self.post_conv = nn.Sequential(
            FusedRes_SE1D(1024, 1024, stride=1),  # 使用融合频域的残差模块
        )

        # mode=3 特定结构
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

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # 收集各阶段中间特征
        i = 0
        for layer in self.layers:
            x, x11 = layer(x)  # 获取下采样后和下采样前特征
            B1, L1, C1 = x11.shape
            H1 = W1 = int(L1 ** 0.5)
            xf1 = x11.view(B1, C1, H1, W1)  # 转换为(B, C, H, W)格式
            if i == 0:
                x1 = xf1  # 第一阶段特征
            elif i == 1:
                x2 = xf1  # 第二阶段特征
            elif i == 2:
                x3 = xf1  # 第三阶段特征
            elif i == 3:
                x4 = xf1  # 第四阶段特征
            i += 1

        # 通过Regression1处理中间特征
        x = self.regression(x1, x2, x3, x4)  # 输出形状: [B, 1024, 48, 48]

        # 调整形状以适配LayerNorm
        B, C, H, W = x.shape
        x = x.flatten(2)  # 展平空间维度: [B, 1024, H*W] = [B, 1024, 2304]
        x = x.transpose(1, 2)  # 转置为 [B, 2304, 1024] (最后一维为通道数1024)

        # 应用LayerNorm
        x = self.norm(x)  # 输入形状: [B, 2304, 1024]，符合LayerNorm要求

        # 恢复维度顺序以适配后续频域模块
        x = x.transpose(1, 2)  # 转回 [B, 1024, 2304]

        # 频域增强后处理（现在DCT矩阵会自动匹配2304的长度）
        x = self.post_conv(x)  # FusedRes_SE1D处理[B, C, L]格式

        # 池化和展平
        x = self.avgpool(x)  # 自适应池化到[B, 1024, 1]
        x = torch.flatten(x, 1)  # 展平为[B, 1024]

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x_cnt = self.output1(x)  # mode=3 输出
        return x_cnt

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops