import threading
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os

# 创建保存图像的目录
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')


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

    def __init__(self, channel, bands=4, reduction=4, max_dct_len=1024, visualize=False):
        super(FusedFrequencySELayer1D, self).__init__()
        self.channel = channel
        self.bands = bands
        self.visualize = visualize
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

        if visualize:
            self.visualizer = FeatureVisualizer()

    def forward(self, x):
        b, c, l = x.size()
        residual = x
        x_dct = self.dct(x)

        if self.visualize:
            self.visualizer.visualize_feature(x_dct, 'dct_transform', b)

        x_bands = self.sbr(x_dct)
        x_fused = self.ms_fa(x_bands)
        x_freq = rearrange(x_fused, 'b bn cbn l -> b (bn cbn) l')

        if self.visualize:
            self.visualizer.visualize_feature(x_freq, 'frequency_bands', b)

        y = self.avg_pool(x_freq).view(b, c)
        y = self.fc(y).view(b, c, 1)

        if self.visualize:
            self.visualizer.visualize_feature(y, 'attention_weights', b)

        return residual * y.expand_as(residual)


class FusedRes_SE1D(nn.Module):
    """融合频域分带的残差模块"""

    def __init__(self, inplanes, planes, stride=1, use_se=True, bands=4, reduction=4, visualize=False):
        super(FusedRes_SE1D, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.use_se = use_se
        self.visualize = visualize

        if use_se:
            self.se = FusedFrequencySELayer1D(planes, bands=bands, reduction=reduction, visualize=visualize)

        self.downsample = nn.Sequential(
            nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(planes)
        ) if inplanes != planes or stride > 1 else lambda x: x

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.visualize:
            FeatureVisualizer.visualize_feature(out, 'conv1_output', x.size(0))

        out = self.conv2(out)
        out = self.bn2(out)

        if self.visualize:
            FeatureVisualizer.visualize_feature(out, 'conv2_output', x.size(0))

        if self.use_se:
            out = self.se(out)

        out += residual

        if self.visualize:
            FeatureVisualizer.visualize_feature(out, 'residual_output', x.size(0))

        out = self.relu(out)
        return out


class FeatureVisualizer:
    """特征可视化工具类"""

    @staticmethod
    def visualize_feature(feature, name, batch_size):
        """
        可视化特征图

        Args:
            feature: 要可视化的特征张量 (B, C, L)
            name: 可视化图像的名称
            batch_size: 批次大小
        """
        # 选择批次中的第一个样本
        feature = feature[0].detach().cpu().numpy()  # 固定使用第一个样本

        # 创建图像
        plt.figure(figsize=(10, 6))

        if feature.shape[0] > 1:  # 如果有多个通道，绘制热力图
            plt.imshow(feature, aspect='auto', cmap='viridis')
            plt.colorbar(label='Magnitude')
            plt.xlabel('Position')
            plt.ylabel('Channel')
        else:  # 如果只有一个通道，绘制曲线图
            plt.plot(feature[0])
            plt.xlabel('Position')
            plt.ylabel('Magnitude')

        plt.title(f'{name} Feature Visualization')
        plt.tight_layout()

        # 保存图像
        plt.savefig(f'visualizations/{name}.png')
        plt.close()
        print(f"Visualization saved to 'visualizations/{name}.png'")


class ImageTo1DFeature(nn.Module):
    """将2D图像转换为1D特征的模块"""

    def __init__(self, image_size=224, patch_size=16):
        super(ImageTo1DFeature, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # 用于将图像块投影到特征空间的线性层
        self.projection = nn.Conv2d(3, 64, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        将2D图像转换为1D特征序列

        Args:
            x: 输入图像张量 (B, 3, H, W)

        Returns:
            1D特征序列 (B, C, L)
        """
        # 应用投影并展平
        x = self.projection(x)  # (B, C, H', W')
        x = x.flatten(2)  # (B, C, H'*W')
        return x


# 简单的测试函数
def test_visualization():
    # 创建一个简单的模型
    model = FusedRes_SE1D(64, 64, use_se=True, visualize=True)

    # 创建一个随机图像
    img = torch.randn(1, 3, 224, 224)

    # 转换为1D特征
    img_to_1d = ImageTo1DFeature()
    features = img_to_1d(img)

    # 通过模型并可视化
    output = model(features)
    print("Model output shape:", output.shape)


if __name__ == "__main__":
    test_visualization()