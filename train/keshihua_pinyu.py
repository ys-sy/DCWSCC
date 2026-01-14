import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from einops import rearrange
from PIL import Image
import torchvision.transforms as transforms
import threading


# ---------------------- 1. 模型类 (保持不变) ----------------------
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


class DCT1D(nn.Module, metaclass=SingletonMeta):
    """使用元类实现的线程安全单例DCT模块"""

    def __init__(self, max_len=1024):
        super(DCT1D, self).__init__()
        self.max_len = max_len

        if not hasattr(self, 'dct_matrix'):
            dct_matrix = self._build_dct_matrix(max_len)
            self.register_buffer('dct_matrix', dct_matrix)
            idct_matrix = dct_matrix.T * max_len
            self.register_buffer('idct_matrix', idct_matrix)
            print(f"DCT1D initialized with max_len={max_len}")

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
        use_len = min(l, self.max_len)
        dct_matrix = self.dct_matrix[:use_len, :use_len]
        return torch.matmul(x, dct_matrix)

    def inverse(self, x):
        b, c, l = x.size()
        use_len = min(l, self.max_len)
        idct_matrix = self.idct_matrix[:use_len, :use_len]
        return torch.matmul(x, idct_matrix)


class FrequencyBandingReorganization(nn.Module):
    """TransMamba的频谱分带重组(SBR)实现"""

    def __init__(self, bands=4):
        super(FrequencyBandingReorganization, self).__init__()
        self.bands = bands

    def forward(self, x):
        b, c, l = x.size()
        assert c % self.bands == 0, "Channel must be divisible by bands"
        return rearrange(x, 'b (bn cbn) l -> b bn cbn l', bn=self.bands)


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
        x_flat = x.reshape(b, bn, -1)
        x3 = self.conv3(x_flat).reshape(b, bn, cbn, l)
        x5 = self.conv5(x_flat).reshape(b, bn, cbn, l)
        gate_input = torch.cat([x3, x5], dim=1)
        g = self.gate(gate_input.reshape(b, 2 * bn, -1)).reshape(b, bn, cbn, l)
        return g * x5 + (1 - g) * x3


class FusedFrequencySELayer1D(nn.Module):
    """使用最终版DCT1D的频域增强模块"""

    def __init__(self, channel, bands=4, reduction=4, max_dct_len=1024):
        super(FusedFrequencySELayer1D, self).__init__()
        self.channel = channel
        self.bands = bands
        self.dct = DCT1D(max_len=max_dct_len)
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

    def __init__(self, inplanes, planes, stride=1, use_se=True, bands=2, reduction=2):
        super(FusedRes_SE1D, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.use_se = use_se
        if use_se:
            self.se = FusedFrequencySELayer1D(planes, bands=bands, reduction=reduction, max_dct_len=80)
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


# ---------------------- 2. 图片加载和预处理 ----------------------
def load_and_preprocess_image(image_path, target_size=(80, 20), display_size=(413, 295)):
    """加载图片并预处理，返回调整大小后的原图用于显示"""
    original_image = Image.open(image_path).convert('RGB')
    resized_original = original_image.resize(display_size, Image.LANCZOS)  # 高质量缩放

    # 模型输入预处理
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image_tensor = transform(original_image)
    if len(image_tensor.shape) != 3:
        raise ValueError(f"处理后的图片张量维度不正确，期望3维，实际{len(image_tensor.shape)}维")

    batch_image = image_tensor.unsqueeze(0)
    batch_image = batch_image.flatten(2)
    batch_image = batch_image[:, :, :80]
    multi_channel = batch_image.repeat(1, 20, 1)

    return multi_channel, resized_original


def load_original_image(image_path, display_size=(413, 295)):
    """加载原始图片用于显示，尺寸为413×295（宽×高）"""
    original_image = Image.open(image_path).convert('RGB')
    resized_image = original_image.resize(display_size, Image.LANCZOS)  # 高质量缩放
    return resized_image


# ---------------------- 3. 主流程 ----------------------
if __name__ == "__main__":
    # 设置全局字体（将坐标轴刻度字号扩大1.5倍）
    base_tick_size = 32
    enlarged_tick_size = int(base_tick_size * 1.5)  # 32×1.5=48

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Georgia', 'Palatino', 'Bookman Old Style', 'serif'],
        'font.size': 32,  # 全局默认字体
        'axes.titlesize': 36,  # 标题字体
        'axes.labelsize': 36,  # 坐标轴标签字体
        'xtick.labelsize': enlarged_tick_size,  # 扩大后的x轴刻度数字
        'ytick.labelsize': enlarged_tick_size,  # 扩大后的y轴刻度数字
        'axes.labelpad': 8,  # 标签与坐标轴间距
        'xtick.major.pad': 6,  # x刻度与轴间距
        'ytick.major.pad': 6,  # y刻度与轴间距
    })

    # 图片路径
    feature_map_path = "/amax/zs/code/WSCC_TAF-main/merged_feature_custom167.jpg"
    original_image_path = "/amax/zs/code/WSCC_TAF-main/datasets/ShanghaiTech/part_A_final/test_data/images/IMG_167.jpg"

    # 加载图片（显示尺寸413×295）
    x, resized_feature_map = load_and_preprocess_image(
        feature_map_path,
        target_size=(80, 20),
        display_size=(413, 295)
    )
    resized_original_image = load_original_image(
        original_image_path,
        display_size=(413, 295)
    )
    print(f"输入图片转换后的形状: {x.shape}")

    # 模型前向计算
    dct_layer = DCT1D(max_len=80)
    res_block = FusedRes_SE1D(inplanes=20, planes=20, use_se=True, bands=4, reduction=4)
    x_res = res_block(x)
    x_dct = dct_layer(x)
    x_res_dct = dct_layer(x_res)

    sbr = FrequencyBandingReorganization(bands=4)
    ms_fa = MultiScaleFrequencyAttention(bands=4)
    x_bands = sbr(x_dct)
    x_fused = ms_fa(x_bands)
    x_freq = x_fused.reshape(1, 20, 80)

    # GAP处理
    num_low_freq = int(0.8 * x_res.size(2))
    x_low_freq = x_res_dct.clone()
    x_low_freq[:, :, num_low_freq:] = 0
    x_low_time = dct_layer.inverse(x_low_freq)
    gap = nn.AdaptiveAvgPool1d(1)
    x_gap = gap(x_low_time)
    x_gap_expanded = x_gap.repeat(1, 1, 80)

    # 数据后处理
    left_mag = x_dct[0].permute(1, 0).abs().detach().numpy()
    middle_mag = x_res_dct[0].permute(1, 0).abs().detach().numpy()
    right_mag = x_gap_expanded[0].permute(1, 0).abs().detach().numpy()


    def print_stats(data, name):
        print(f"\n{name} 统计量:")
        print(f"min: {data.min():.4f}, max: {data.max():.4f}")
        print(f"25%分位: {np.percentile(data, 25):.4f}, 75%分位: {np.percentile(data, 75):.4f}")
        print(f"95%分位: {np.percentile(data, 95):.4f}")


    print_stats(left_mag, "原始DCT幅度")
    print_stats(middle_mag, "残差DCT幅度")
    print_stats(right_mag, "GAP结果幅度")


    def normalize_contrast(data, lower_percent=5, upper_percent=95):
        p_low = np.percentile(data, lower_percent)
        p_high = np.percentile(data, upper_percent)
        data_clipped = np.clip(data, p_low, p_high)
        return (data_clipped - p_low) / (p_high - p_low + 1e-8)


    def enhance_low_values(data, epsilon=1e-8):
        return np.log(data + epsilon)


    left_enhanced = enhance_low_values(left_mag) if left_mag.min() >= 0 else left_mag
    middle_enhanced = enhance_low_values(middle_mag) if middle_mag.min() >= 0 else middle_mag
    right_enhanced = enhance_low_values(right_mag) if right_mag.min() >= 0 else right_mag

    left_norm = normalize_contrast(left_enhanced)
    middle_norm = normalize_contrast(middle_enhanced)
    right_norm = normalize_contrast(right_enhanced)

    # 可视化（间距缩小一倍，优化布局）
    cmap = 'Blues'

    # 保持画布尺寸不变，通过减小wspace缩小子图间距
    fig, axes = plt.subplots(1, 5, figsize=(54, 15), sharey=False)
    plt.subplots_adjust(wspace=0.025)  # 水平间距缩小一倍

    # 第1个子图：原图
    ax0 = axes[0]
    ax0.imshow(resized_original_image)
    # ax0.set_title('Original Image', fontsize=36, fontfamily='serif')
    ax0.axis('off')

    # 第2个子图：特征图
    ax1 = axes[1]
    ax1.imshow(resized_feature_map)
    # ax1.set_title('Feature Map', fontsize=36, fontfamily='serif')
    ax1.axis('off')

    # 第3个子图：原始DCT结果
    ax2 = axes[2]
    im2 = ax2.imshow(
        left_norm,
        cmap=cmap,
        aspect=0.45,
        vmin=0, vmax=1,
        extent=[0, 20, 80, 0]
    )
    ax2.set_xlabel('Channel', fontsize=36, fontfamily='serif')
    ax2.set_ylabel('Frequency', fontsize=36, fontfamily='serif')
    ax2.set_xticks([0, 10, 20])
    ax2.set_yticks([0, 40, 80])
    ax2.tick_params(axis='both', labelsize=enlarged_tick_size)  # 应用扩大后的刻度大小
    # ax2.set_title('DCT', fontsize=36, fontfamily='serif')
    cbar2 = fig.colorbar(im2, ax=ax2, orientation='vertical', pad=0.005, aspect=20, shrink=0.75)
    cbar2.set_label('Normalized Magnitude', fontsize=32, fontfamily='serif')

    # 第4个子图：残差模块处理后的DCT结果
    ax3 = axes[3]
    im3 = ax3.imshow(
        middle_norm,
        cmap=cmap,
        aspect=0.45,
        vmin=0, vmax=1,
        extent=[0, 20, 80, 0]
    )
    ax3.set_xlabel('Channel', fontsize=36, fontfamily='serif')
    ax3.set_ylabel('Frequency', fontsize=36, fontfamily='serif')
    ax3.set_xticks([0, 10, 20])
    ax3.set_yticks([0, 40, 80])
    ax3.tick_params(axis='both', labelsize=enlarged_tick_size)  # 应用扩大后的刻度大小
    # ax3.set_title('TDPR', fontsize=36, fontfamily='serif')
    cbar3 = fig.colorbar(im3, ax=ax3, orientation='vertical', pad=0.005, aspect=20, shrink=0.75)
    cbar3.set_label('Normalized Magnitude', fontsize=32, fontfamily='serif')

    # 第5个子图：GAP结果
    ax4 = axes[4]
    im4 = ax4.imshow(
        right_norm,
        cmap=cmap,
        aspect=0.45,
        vmin=0, vmax=1,
        extent=[0, 20, 80, 0]
    )
    ax4.set_xlabel('Channel', fontsize=36, fontfamily='serif')
    ax4.set_ylabel('Frequency', fontsize=36, fontfamily='serif')
    ax4.set_xticks([0, 10, 20])
    ax4.set_yticks([0, 40, 80])
    ax4.tick_params(axis='both', labelsize=enlarged_tick_size)  # 应用扩大后的刻度大小
    # ax4.set_title('GAP Result', fontsize=36, fontfamily='serif')
    cbar4 = fig.colorbar(im4, ax=ax4, orientation='vertical', pad=0.005, aspect=20, shrink=0.75)
    cbar4.set_label('Normalized Magnitude', fontsize=32, fontfamily='serif')

    plt.tight_layout()
    plt.show()