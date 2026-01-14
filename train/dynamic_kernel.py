import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import os

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
            self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=groups)

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
        attention = x
        attention = self.attention(attention)
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



# import torch
# from your_module import MixFFN  # 请替换为实际的模块名

def test_mixffn():
    # 设置测试参数
    test_configs = [
        {"dim": 64, "num_kernels": 4, "input_size": (1, 64, 32, 32)},
        {"dim": 128, "num_kernels": 4, "input_size": (1, 128, 16, 16)},
        {"dim": 256, "num_kernels": 8, "input_size": (1, 256, 8, 8)},
        {"dim": 512, "num_kernels": 8, "input_size": (1, 512, 4, 4)}
    ]

    # 对每种配置进行测试
    for config in test_configs:
        print(f"\n测试配置: dim={config['dim']}, num_kernels={config['num_kernels']}, "
              f"input_size={config['input_size']}")

        # 创建MixFFN实例
        mixffn = MixFFN(dim=config["dim"], num_kernels=config["num_kernels"])

        # 生成随机输入数据
        x = torch.randn(*config["input_size"])

        # 前向传播
        try:
            output = mixffn(x)
            print(f"输入尺寸: {x.shape}")
            print(f"输出尺寸: {output.shape}")
            print("测试通过!")
        except Exception as e:
            print(f"测试失败: {e}")


if __name__ == "__main__":
    test_mixffn()