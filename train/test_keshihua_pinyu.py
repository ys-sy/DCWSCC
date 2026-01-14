import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from PIL import Image
import torchvision.transforms as transforms


# 动态DCT模块
class DCT1D(nn.Module):
    def __init__(self):
        super(DCT1D, self).__init__()
        self.cache = {}  # 缓存不同长度的DCT矩阵

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
        b, c, l = x.size()
        if l not in self.cache:
            dct_matrix = self._build_dct_matrix(l)
            self.cache[l] = dct_matrix.to(x.device)
        return torch.matmul(x, self.cache[l])


# 频谱分带重组模块
class FrequencyBandingReorganization(nn.Module):
    def __init__(self, bands):
        super(FrequencyBandingReorganization, self).__init__()
        self.bands = bands

    def forward(self, x):
        b, c, l = x.size()
        assert c % self.bands == 0, "通道数需能被分带数整除"
        return rearrange(x, 'b (bn cbn) l -> b bn cbn l', bn=self.bands)


# 多尺度频域卷积模块
class MultiScaleFrequencyConv(nn.Module):
    def __init__(self, cbn):
        super().__init__()
        self.conv3 = nn.Conv1d(cbn, cbn, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(cbn, cbn, kernel_size=5, padding=2)

    def forward(self, x):
        # x: (b, bn, cbn, l)
        b, bn, cbn, l = x.shape
        x_reshaped = x.reshape(b * bn, cbn, l)
        x3 = self.conv3(x_reshaped).reshape(b, bn, cbn, l)
        x5 = self.conv5(x_reshaped).reshape(b, bn, cbn, l)
        return x3, x5


# 频域门控模块
class FrequencyGate(nn.Module):
    def __init__(self, cbn):
        super().__init__()
        self.conv = nn.Conv1d(2 * cbn, cbn, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x3, x5):
        # x3, x5: (b, bn, cbn, l)
        b, bn, cbn, l = x3.shape
        concat = torch.cat([x3, x5], dim=2)  # (b, bn, 2cbn, l)
        concat_reshaped = concat.permute(0, 2, 1, 3).reshape(b, 2 * cbn, bn * l)
        gate = self.conv(concat_reshaped).reshape(b, cbn, bn, l).permute(0, 2, 1, 3)
        return self.sigmoid(gate)


# 空间门控模块
class SpatialGate(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (b, in_channels, l)
        gate = self.conv(x)
        return self.sigmoid(gate)


# 门控融合模块
class GateFusion(nn.Module):
    def __init__(self, bn, cbn):
        super().__init__()
        self.bn = bn
        self.cbn = cbn
        self.mlp = nn.Sequential(
            nn.Conv1d(2 * cbn, cbn, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(cbn, cbn, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, g_spa, g_freq):
        # g_spa: (b, bn*cbn, l) -> reshape to (b, bn, cbn, l)
        b, c, l = g_spa.shape
        assert c == self.bn * self.cbn, "空间门控通道数不匹配"
        g_spa_reshaped = g_spa.reshape(b, self.bn, self.cbn, l)
        # g_freq: (b, self.bn, self.cbn, l)
        concat = torch.cat([g_spa_reshaped, g_freq], dim=2)  # (b, bn, 2cbn, l)
        concat_reshaped = concat.permute(0, 2, 1, 3).reshape(b, 2 * self.cbn, self.bn * l)
        gate = self.mlp(concat_reshaped).reshape(b, self.cbn, self.bn, l).permute(0, 2, 1, 3)
        return gate


# 双层MLP注意力模块
class DoubleMLPAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mlp1 = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.mlp2 = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (b, in_channels, l)
        x = self.mlp1(x)
        x = self.relu(x)
        x = self.mlp2(x)
        return self.sigmoid(x)


# 热力图绘制函数
def plot_heatmap(data, ax, title, vmin=None, vmax=None):
    """绘制频域特征热力图，横坐标0、5、10，纵坐标0到60"""
    # 处理数据维度并确保非负
    data_np = data.squeeze(0).detach().numpy().T
    data_np = np.maximum(data_np, 0)  # 双重保障非负

    # 绘制热力图
    im = ax.imshow(data_np, cmap='jet', vmin=vmin, vmax=vmax, aspect='auto')

    # 设置坐标刻度和范围
    ax.set_xticks([0, 5, 10])
    ax.set_xticklabels(['0', '5', '10'], fontsize=8)
    ax.set_ylim(0, 60)
    ax.set_xlim(0, 10)

    # 设置坐标轴标签
    ax.set_xlabel('Channel', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(title, fontsize=12)

    return im


# 加载图片并提取特征
def load_image_features(image_path):
    """加载图片并提取空间特征，输出形状为(1, 10, 90)"""
    # 加载图片
    img = Image.open(image_path).convert('RGB')

    # 预处理（不使用标准化，避免引入负值）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # 保持值在0-1范围
    ])

    # 处理图片并添加batch维度
    img_tensor = transform(img).unsqueeze(0)  # 形状: (1, 3, 224, 224)

    # 特征提取器（输出非负特征）
    feature_extractor = nn.Sequential(
        nn.Conv2d(3, 10, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),  # 确保空间特征非负
        nn.AdaptiveAvgPool2d((1, 90))
    )

    # 提取特征
    with torch.no_grad():
        features = feature_extractor(img_tensor)
        features = features.squeeze(2)  # 形状: (1, 10, 90)

    return features


# 主程序
if __name__ == "__main__":
    # 配置参数
    image_path = "/amax/zs/code/WSCC_TAF-main/datasets/ShanghaiTech/part_A_final/test_data/images/IMG_11.jpg"
    bands = 5  # 分带数量，需满足10 % bands == 0 → bands=5，cbn=2
    cbn = 10 // bands  # 每个子带的通道数

    # 加载空间特征
    spatial_feat = load_image_features(image_path)  # (1, 10, 90)

    # 初始化模块
    dct_layer = DCT1D()
    sbr_layer = FrequencyBandingReorganization(bands=bands)
    ms_conv = MultiScaleFrequencyConv(cbn=cbn)
    freq_gate = FrequencyGate(cbn=cbn)
    spatial_gate = SpatialGate(in_channels=10)
    gate_fusion = GateFusion(bn=bands, cbn=cbn)
    mlp_att = DoubleMLPAttention(in_channels=bands)  # in_channels=bands=5

    # 特征处理流程
    # 1. 频域转换
    F_dct = dct_layer(spatial_feat)
    F_dct = torch.relu(F_dct)  # (1, 10, 90)

    # 2. 频率分带重组
    F_bandk = sbr_layer(F_dct)  # (1, 5, 2, 90)

    # 3. 多尺度频域卷积
    ms3, ms5 = ms_conv(F_bandk)  # (1,5,2,90), (1,5,2,90)

    # 4. 空间门控
    G_spa = spatial_gate(spatial_feat)  # (1,10,90)

    # 5. 频域门控
    G_freq = freq_gate(ms3, ms5)  # (1,5,2,90)

    # 6. 门控融合
    G = gate_fusion(G_spa, G_freq)  # (1,5,2,90)

    # 7. 融合特征
    F_fusion = G * ms5 + (1 - G) * ms3  # (1,5,2,90)

    # 8. 频带平均池化
    Z = F_fusion.mean(dim=2)  # 对cbn维度（dim=2）平均 → (1,5,90)

    # 9. 双层MLP注意力
    att = mlp_att(Z)  # (1,5,90)

    # 10. 残差连接
    # 重塑F_fusion到(1,10,90)
    F_fusion_reshaped = F_fusion.reshape(1, bands * cbn, 90)  # 5*2=10 → (1,10,90)
    # 扩展att到(1,10,90)：每个bands通道重复cbn次
    att_reshaped = att.unsqueeze(2).repeat(1, 1, cbn, 1).reshape(1, bands * cbn, 90)
    # 残差连接
    F_out = att_reshaped * F_fusion_reshaped + F_dct

    # 全局平均池化对比（原代码中的gap_feat）
    gap_layer = nn.AdaptiveAvgPool1d(1)
    gap_feat = gap_layer(F_bandk.reshape(1, bands * cbn, 90))
    gap_feat = gap_feat.repeat(1, 1, 90)
    gap_feat = torch.relu(gap_feat)

    # 计算统一颜色范围
    all_features = torch.cat([F_out, gap_feat])
    all_values = all_features.detach().numpy().flatten()
    vmin, vmax = 0, np.percentile(all_values, 95)  # 从0开始的颜色范围

    # 绘制对比热力图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    im1 = plot_heatmap(F_out, ax1, 'Enhanced Frequency Features', vmin, vmax)
    im2 = plot_heatmap(gap_feat, ax2, 'Global Average Pooling Features', vmin, vmax)

    # 添加颜色条
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.04, pad=0.04)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.04, pad=0.04)
    cbar1.set_label('Amplitude', rotation=270, labelpad=15)
    cbar2.set_label('Amplitude', rotation=270, labelpad=15)

    # 保存并显示
    plt.tight_layout()
    plt.savefig('final_feature_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()