# import os
# import cv2
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from torchvision import transforms
# import torch.nn as nn
# import torch.nn.functional as F
# import random  # 新增：导入random模块
#
# # ------------------------------
# # 新增：固定所有随机种子，确保结果可复现
# # ------------------------------
# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)  # CPU随机种子
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)  # GPU随机种子
#         torch.cuda.manual_seed_all(seed)  # 多GPU随机种子
#     torch.backends.cudnn.deterministic = True  # 确保CUDA卷积算法 deterministic
#     torch.backends.cudnn.benchmark = False  # 禁用benchmark模式（可能引入随机性）
#
# # 调用种子设置函数
# set_seed(seed=100)
#
# # 设置设备
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"使用设备: {device}")
#
#
# # ------------------------------
# # 1. TopK通道注意力模块（不变）
# # ------------------------------
# class TopKChannelAttention(nn.Module):
#     """TopK通道注意力模块，选择最重要的K个通道"""
#     def __init__(self, in_channels, top_k=32):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
#         self.top_k = top_k  # 选择的TopK通道数
#
#     def forward(self, x):
#         b, c, _, _ = x.shape
#         # 提取通道统计信息
#         y = self.avg_pool(x).view(b, c)  # 形状: (b, c)
#
#         # 确定要选择的K值（不超过总通道数）
#         k = min(self.top_k, c)
#
#         # 获取TopK通道的索引（固定种子后，相同输入会得到相同索引）
#         _, top_indices = torch.topk(y, k, dim=1)
#
#         # 初始化权重（全部为0）
#         weights = torch.zeros_like(y)
#
#         # 将TopK通道的权重设为1（硬选择）
#         weights.scatter_(1, top_indices, 1.0)
#
#         return weights.view(b, c, 1, 1)  # 输出形状: (b, c, 1, 1)
#
#
# # ------------------------------
# # 2. 图像预处理（不变）
# # ------------------------------
# def preprocess_image(img_path):
#     """预处理图像为模型输入格式"""
#     # 读取图像
#     Img_data = cv2.imread(img_path)
#     if Img_data is None:
#         raise ValueError(f"无法读取图像: {img_path}，请检查路径是否正确")
#
#     # 转换为RGB格式
#     Img_data = cv2.cvtColor(Img_data, cv2.COLOR_BGR2RGB)
#
#     # 调整大小（保持比例）
#     if Img_data.shape[1] >= Img_data.shape[0]:  # 宽 >= 高
#         rate_1 = 1152.0 / Img_data.shape[1]
#         rate_2 = 768 / Img_data.shape[0]
#         Img_data = cv2.resize(Img_data, (0, 0), fx=rate_1, fy=rate_2)
#     else:  # 高 > 宽
#         rate_1 = 1152.0 / Img_data.shape[0]
#         rate_2 = 768.0 / Img_data.shape[1]
#         Img_data = cv2.resize(Img_data, (0, 0), fx=rate_2, fy=rate_1)
#
#     # 调整到模型输入尺寸(384x384)
#     Img_data = cv2.resize(Img_data, (384, 384))
#
#     # 标准化处理
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     img_tensor = transform(Img_data).unsqueeze(0)  # 添加批次维度
#
#     return img_tensor, Img_data
#
#
# # ------------------------------
# # 3. 模型加载与特征钩子（不变）
# # ------------------------------
# def load_model_with_hooks(model_path):
#     """加载模型并注册钩子获取中间层特征"""
#     # 导入模型（请替换为你的模型实际路径）
#     try:
#         from Networks.models import base_patch16_384_swin_cdpnet3
#     except ImportError:
#         raise ImportError("请修改模型导入路径为你的实际模型位置")
#
#     # 创建模型实例
#     model = base_patch16_384_swin_cdpnet3(pretrained=False, mode=3)
#
#     # 加载权重
#     try:
#         checkpoint = torch.load(model_path, map_location=device)
#         if 'model' in checkpoint:
#             model.load_state_dict(checkpoint['model'], strict=False)
#         else:
#             model.load_state_dict(checkpoint, strict=False)
#         print("模型权重加载成功")
#     except Exception as e:
#         raise RuntimeError(f"模型权重加载失败: {str(e)}")
#
#     model.to(device)
#     model.eval()  # 推理模式
#
#     # 存储中间特征的字典
#     intermediate_features = {}
#
#     # 仅注册x4特征的钩子
#     if len(model.layers) >= 4:
#         model.layers[3].register_forward_hook(
#             lambda m, i, o: intermediate_features.update({'x4': o[1].detach()})
#         )
#     else:
#         raise ValueError("模型layers数量不足4层，无法获取x4特征")
#
#     # 注册钩子获取regression输出
#     if hasattr(model, 'regression'):
#         model.regression.register_forward_hook(
#             lambda m, i, o: intermediate_features.update({'regression_output': o.detach()})
#         )
#     else:
#         raise AttributeError("模型中未找到'regression'模块，请检查模型结构")
#
#     return model, intermediate_features
#
#
# # ------------------------------
# # 4. TopK注意力加权融合（不变）
# # ------------------------------
# def aggregate_attention(features, attention_module):
#     """使用TopK通道注意力加权融合特征通道"""
#     # 扩展为4D张量 (1, C, H, W)
#     features_4d = features.unsqueeze(0)
#
#     # 计算通道权重（TopK选择）
#     with torch.no_grad():
#         channel_weights = attention_module(features_4d)
#
#     # 加权融合（只保留TopK通道的特征）
#     weighted_features = features_4d * channel_weights
#     fused_heatmap = torch.sum(weighted_features, dim=1).squeeze(0)  # 压缩为(H, W)
#
#     return fused_heatmap
#
#
# # ------------------------------
# # 5. 热力图生成与显示（不变）
# # ------------------------------
# def generate_ordered_heatmaps(features_dict, original_img, save_dir=None, top_k=32):
#     """按顺序生成并显示所有热力图"""
#     display_order = [
#         ('original', '原图', None),
#         ('x4', '第四阶段特征', 'x4'),
#         ('regression_output', 'Regression输出', 'regression_output')
#     ]
#
#     # 创建画布（1行3列）
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#     heatmaps = {}
#
#     for i, (key, title, feature_key) in enumerate(display_order):
#         ax = axes[i]
#
#         if key == 'original':
#             ax.imshow(original_img)
#             ax.set_title(title, fontsize=12)
#             ax.axis('off')
#             continue
#
#         # 检查特征是否存在
#         if feature_key not in features_dict:
#             ax.text(0.5, 0.5, f'未找到{title}', ha='center', va='center')
#             ax.axis('off')
#             continue
#
#         # 处理特征格式
#         features = features_dict[feature_key]
#         if len(features.shape) == 3:  # 处理序列格式特征 (B, L, C)
#             B, L, C = features.shape
#             H = W = int(L** 0.5)
#             features = features.view(B, C, H, W)
#
#         # 提取单样本特征
#         features_single = features[0]  # (C, H, W)
#         C = features_single.shape[0]
#
#         # 初始化TopK注意力模块
#         attention = TopKChannelAttention(in_channels=C, top_k=top_k).to(device)
#         attention.eval()
#
#         # 生成热力图
#         heatmap = aggregate_attention(features_single, attention)
#
#         # 后处理
#         heatmap = heatmap.cpu().numpy()
#         heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
#         heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
#
#         # 存储与显示
#         heatmaps[feature_key] = heatmap
#         ax.imshow(original_img)
#         ax.imshow(heatmap, cmap='jet', alpha=0.5)
#         ax.set_title(f'{title}\n(Top{min(top_k, C)}通道注意力)', fontsize=12)
#         ax.axis('off')
#
#     # 调整布局并保存
#     plt.tight_layout()
#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)
#         save_path = os.path.join(save_dir, 'topk_attention_heatmaps.png')
#         plt.savefig(save_path, bbox_inches='tight', dpi=300)
#         print(f"热力图已保存至: {save_path}")
#
#     plt.show()
#     return heatmaps
#
#
# # ------------------------------
# # 6. 主函数（不变）
# # ------------------------------
# def main(img_path, model_path, save_dir=None, top_k=32):
#     """主函数：生成所有层的热力图"""
#     # 预处理图像
#     print("预处理图像...")
#     img_tensor, original_img = preprocess_image(img_path)
#     img_tensor = img_tensor.to(device)
#
#     # 加载模型和特征钩子
#     print("加载模型并注册钩子...")
#     model, intermediate_features = load_model_with_hooks(model_path)
#
#     # 前向传播
#     print("进行前向传播获取特征...")
#     with torch.no_grad():
#         _ = model(img_tensor)
#
#     # 生成热力图
#     print("生成热力图...")
#     heatmaps = generate_ordered_heatmaps(
#         intermediate_features,
#         original_img,
#         save_dir,
#         top_k=top_k
#     )
#
#     print(f"完成！共生成 {len(heatmaps)} 个热力图")
#     return heatmaps, original_img
#
#
# # ------------------------------
# # 运行入口
# # ------------------------------
# if __name__ == "__main__":
#     # 路径设置
#     img_path = "/amax/zs/code/WSCC_TAF-main/datasets/ShanghaiTech/part_A_final/test_data/images/IMG_11.jpg"
#     model_path = "/amax/zs/code/WSCC_TAF-main/save_file/ShanghaiA_swincdpnet3/model_best.pth"
#     save_dir = "./results_topk_attention"
#     top_k = 20  # 选择的TopK通道数
#
#     # 运行主函数
#     main(
#         img_path=img_path,
#         model_path=model_path,
#         save_dir=save_dir,
#         top_k=top_k
#     )


import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import random


# ------------------------------
# 固定所有随机种子，确保结果可复现
# ------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# set_seed(seed=150)
set_seed(seed=50)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ------------------------------
# TopK通道注意力模块
# ------------------------------
class TopKChannelAttention(nn.Module):
    def __init__(self, in_channels, top_k=32):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.top_k = top_k

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        k = min(self.top_k, c)
        _, top_indices = torch.topk(y, k, dim=1)
        weights = torch.zeros_like(y)
        weights.scatter_(1, top_indices, 1.0)
        return weights.view(b, c, 1, 1)


# ------------------------------
# 图像预处理
# ------------------------------
def preprocess_image(img_path):
    Img_data = cv2.imread(img_path)
    if Img_data is None:
        raise ValueError(f"无法读取图像: {img_path}，请检查路径是否正确")

    Img_data = cv2.cvtColor(Img_data, cv2.COLOR_BGR2RGB)

    if Img_data.shape[1] >= Img_data.shape[0]:
        rate_1 = 1152.0 / Img_data.shape[1]
        rate_2 = 768 / Img_data.shape[0]
        Img_data = cv2.resize(Img_data, (0, 0), fx=rate_1, fy=rate_2)
    else:
        rate_1 = 1152.0 / Img_data.shape[0]
        rate_2 = 768.0 / Img_data.shape[1]
        Img_data = cv2.resize(Img_data, (0, 0), fx=rate_2, fy=rate_1)

    Img_data = cv2.resize(Img_data, (384, 384))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(Img_data).unsqueeze(0)

    return img_tensor, Img_data


# ------------------------------
# 模型加载与特征钩子
# ------------------------------
def load_model_with_hooks(model_path):
    try:
        from Networks.models import base_patch16_384_swin_cdpnet3
    except ImportError:
        raise ImportError("请修改模型导入路径为你的实际模型位置")

    model = base_patch16_384_swin_cdpnet3(pretrained=False, mode=3)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("模型权重加载成功")
    except Exception as e:
        raise RuntimeError(f"模型权重加载失败: {str(e)}")

    model.to(device)
    model.eval()

    intermediate_features = {}

    if len(model.layers) >= 4:
        model.layers[3].register_forward_hook(
            lambda m, i, o: intermediate_features.update({'x4': o[1].detach()})
        )
    else:
        raise ValueError("模型layers数量不足4层，无法获取x4特征")

    if hasattr(model, 'regression'):
        model.regression.register_forward_hook(
            lambda m, i, o: intermediate_features.update({'regression_output': o.detach()})
        )
    else:
        raise AttributeError("模型中未找到'regression'模块，请检查模型结构")

    return model, intermediate_features


# ------------------------------
# TopK注意力加权融合
# ------------------------------
def aggregate_attention(features, attention_module):
    features_4d = features.unsqueeze(0)
    with torch.no_grad():
        channel_weights = attention_module(features_4d)
    weighted_features = features_4d * channel_weights
    fused_heatmap = torch.sum(weighted_features, dim=1).squeeze(0)
    return fused_heatmap


# ------------------------------
# 热力图生成与显示（带白色间隙）
# ------------------------------
def generate_ordered_heatmaps(features_dict, original_img, save_dir=None, top_k=32):
    display_order = [
        ('original', None),
        ('x4', 'x4'),
        ('regression_output', 'regression_output')
    ]

    # 创建画布并设置背景为白色
    fig = plt.figure(figsize=(18, 6), facecolor='white')
    # 移除画布边框
    fig.patch.set_edgecolor('none')

    # 创建子图
    axes = [fig.add_subplot(1, 3, i + 1) for i in range(3)]

    # 调整子图间距，设置白色间隙
    plt.subplots_adjust(
        wspace=0.05,  # 图像间水平间隙大小，可根据需要调整
        hspace=0,
        left=0,
        right=1,
        bottom=0,
        top=1
    )

    heatmaps = {}

    for i, (key, feature_key) in enumerate(display_order):
        ax = axes[i]
        # 移除坐标轴
        ax.axis('off')
        # 移除子图边框
        ax.set_frame_on(False)
        # 设置子图背景透明，显示画布的白色
        ax.set_facecolor('none')

        if key == 'original':
            ax.imshow(original_img)
            continue

        if feature_key not in features_dict:
            continue

        features = features_dict[feature_key]
        if len(features.shape) == 3:
            B, L, C = features.shape
            H = W = int(L ** 0.5)
            features = features.view(B, C, H, W)

        features_single = features[0]
        C = features_single.shape[0]

        attention = TopKChannelAttention(in_channels=C, top_k=top_k).to(device)
        attention.eval()

        heatmap = aggregate_attention(features_single, attention)
        heatmap = heatmap.cpu().numpy()
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)

        heatmaps[feature_key] = heatmap
        ax.imshow(original_img)
        ax.imshow(heatmap, cmap='jet', alpha=0.5)

    # 保存图片
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'topk_attention_heatmaps.png')
        plt.savefig(
            save_path,
            bbox_inches='tight',
            pad_inches=0,
            dpi=300,
            facecolor=fig.get_facecolor()  # 保存白色背景
        )
        print(f"热力图已保存至: {save_path}")

    plt.show()
    return heatmaps


# ------------------------------
# 主函数
# ------------------------------
def main(img_path, model_path, save_dir=None, top_k=32):
    print("预处理图像...")
    img_tensor, original_img = preprocess_image(img_path)
    img_tensor = img_tensor.to(device)

    print("加载模型并注册钩子...")
    model, intermediate_features = load_model_with_hooks(model_path)

    print("进行前向传播获取特征...")
    with torch.no_grad():
        _ = model(img_tensor)

    print("生成热力图...")
    heatmaps = generate_ordered_heatmaps(
        intermediate_features,
        original_img,
        save_dir,
        top_k=top_k
    )

    print(f"完成！共生成 {len(heatmaps)} 个热力图")
    return heatmaps, original_img


# ------------------------------
# 运行入口
# ------------------------------
if __name__ == "__main__":
    # 路径设置（请根据实际情况修改）
    img_path = "/amax/zs/code/WSCC_TAF-main/datasets/ShanghaiTech/part_A_final/test_data/images/IMG_38.jpg"
    model_path = "/amax/zs/code/WSCC_TAF-main/save_file/ShanghaiA_swincdpnet3/model_best.pth"
    save_dir = "./results_topk_attention"
    # top_k = 30  # 选择的TopK通道数
    top_k = 60 # 选择的TopK通道数

    # 运行主函数
    main(
        img_path=img_path,
        model_path=model_path,
        save_dir=save_dir,
        top_k=top_k
    )

