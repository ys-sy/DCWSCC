import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from swin_transformer_CDPNet import SwinTransformer_CDPNET

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_image(image_path, patch_size=4, window_size=12, num_downsampling=4):
    """
    预处理图像，确保：
    1. 尺寸能被patch_size整除
    2. 经过多次下采样后仍能被window_size整除
    """
    # 计算总下采样倍数 (4次下采样，每次1/2)
    total_downsample = (2 ** num_downsampling) * patch_size

    # 使用cv2读取图像
    Img_data = cv2.imread(image_path)
    if Img_data is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 转换为RGB格式
    Img_data = cv2.cvtColor(Img_data, cv2.COLOR_BGR2RGB)

    # 按照数据预处理中的缩放逻辑进行处理
    if Img_data.shape[1] >= Img_data.shape[0]:  # 宽度 >= 高度
        rate_1 = 1152.0 / Img_data.shape[1]
        rate_2 = 768.0 / Img_data.shape[0]
        Img_data = cv2.resize(Img_data, (0, 0), fx=rate_1, fy=rate_2)
    else:  # 高度 > 宽度
        rate_1 = 1152.0 / Img_data.shape[0]
        rate_2 = 768.0 / Img_data.shape[1]
        Img_data = cv2.resize(Img_data, (0, 0), fx=rate_2, fy=rate_1)

    # 确保图像尺寸经过总下采样后能被window_size整除
    h, w = Img_data.shape[:2]

    # 计算调整后的高度和宽度
    h = h - (h % total_downsample)
    w = w - (w % total_downsample)

    # 确保至少有一个有效窗口
    if h < window_size * (2 ** num_downsampling):
        h = window_size * (2 ** num_downsampling)
    if w < window_size * (2 ** num_downsampling):
        w = window_size * (2 ** num_downsampling)

    # 裁剪到正确尺寸
    Img_data = Img_data[:h, :w]

    # 转换为Tensor并标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(Img_data).unsqueeze(0).to(device)
    return input_tensor, Img_data


def load_model(model_path, img_size):
    """加载预训练模型"""
    model = SwinTransformer_CDPNET(img_size=img_size,
                                   patch_size=4,
                                   in_chans=3,
                                   num_classes=1,
                                   embed_dim=128,
                                   depths=[2, 2, 18, 2],
                                   num_heads=[4, 8, 16, 32],
                                   window_size=12,
                                   mlp_ratio=4.0,
                                   qkv_bias=True,
                                   qk_scale=None,
                                   drop_rate=0.0,
                                   drop_path_rate=0.2,
                                   ape=False,
                                   patch_norm=True,
                                   use_checkpoint=False,
                                   mode=3)

    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if 'attn_mask' in k:
            continue
        if k.startswith('module.'):
            new_k = k[len('module.'):]
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def get_regression_layers(model):
    """获取Regression2中的所有可可视化层"""
    regression = model.regression
    layers = {
        'v1': regression.v1,
        'v2': regression.v2,
        'v3': regression.v3,
        'stage1': regression.stage1,
        'stage2': regression.stage2,
        'stage3': regression.stage3,
        'stage4': regression.stage4,
        'concat_add': 'special'
    }
    return layers


def visualize_regression_layers(model, image_path, target_layers, save_dir=None):
    """可视化Regression2中的指定层输出"""
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 加载并预处理图像
    input_tensor, original_image = preprocess_image(image_path)
    img_h, img_w = original_image.shape[:2]
    print(f"处理后的图像尺寸: {img_w}x{img_h}")
    print(f"总下采样倍数: 64 (4次下采样，每次1/2，加上4x4 patch)")
    print(f"最终特征图尺寸: {img_w // 64}x{img_h // 64} (必须能被12整除)")

    # 存储中间层输出
    activation = {}

    def get_activation(name):
        """创建钩子函数"""

        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    # 注册钩子
    hooks = []
    layers = get_regression_layers(model)

    for layer_name in target_layers:
        if layer_name not in layers:
            print(f"警告: 层 {layer_name} 不存在于Regression2中")
            continue

        if layer_name == 'concat_add':
            def hook_concat_add(model, input, output):
                activation['concat_add'] = output.detach()

            hooks.append(regression.register_forward_hook(hook_concat_add))
        else:
            layer = layers[layer_name]
            hooks.append(layer.register_forward_hook(get_activation(layer_name)))

    # 前向传播
    with torch.no_grad():
        try:
            _ = model(input_tensor)
        except RuntimeError as e:
            print(f"前向传播错误: {e}")
            print(f"输入张量形状: {input_tensor.shape}")
            b, c, h, w = input_tensor.shape
            print(f"Patch数量: {(h // 4) * (w // 4)}")
            print(f"最终特征图尺寸: {(h // 64)}x{(w // 64)}")
            print(f"是否能被window_size(12)整除: {(h // 64) % 12 == 0}, {(w // 64) % 12 == 0}")
            return

    # 移除钩子
    for hook in hooks:
        hook.remove()

    # 可视化原图
    plt.figure(figsize=(15, 5 * (len(target_layers) + 1)))
    plt.subplot(len(target_layers) + 1, 1, 1)
    plt.imshow(original_image)
    plt.title("原始图像")
    plt.axis('off')

    # 可视化各层特征
    for i, layer_name in enumerate(target_layers):
        if layer_name not in activation:
            continue

        feature_map = activation[layer_name][0]
        print(f"层 {layer_name} 特征形状: {feature_map.shape}")

        if len(feature_map.shape) == 3:
            vis_feature = torch.mean(feature_map, dim=0).cpu().numpy()
            vis_feature = (vis_feature - vis_feature.min()) / (vis_feature.max() - vis_feature.min() + 1e-8)
            vis_feature_resized = cv2.resize(vis_feature, (img_w, img_h))

            plt.subplot(len(target_layers) + 1, 1, i + 2)
            plt.imshow(vis_feature_resized, cmap='viridis')
            plt.title(f"Regression2 - {layer_name} (mean of {feature_map.shape[0]} channels)")
            plt.axis('off')

            if save_dir:
                save_path = os.path.join(save_dir, f"{layer_name}_feature.png")
                plt.imsave(save_path, vis_feature, cmap='viridis')
                save_path_resized = os.path.join(save_dir, f"{layer_name}_resized.png")
                plt.imsave(save_path_resized, vis_feature_resized, cmap='viridis')

    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == "__main__":
    MODEL_PATH = '/amax/zs/code/WSCC_TAF-main/save_file/ShanghaiA_swin_cdpnet_multi_GAP2/model_best.pth'
    IMAGE_PATH = '/amax/zs/code/WSCC_TAF-main/datasets/ShanghaiTech/part_A_final/test_data/images/IMG_1.jpg'
    TARGET_LAYERS = ['v1', 'v2', 'v3', 'stage1', 'stage2', 'stage3', 'stage4', 'concat_add']
    SAVE_DIR = "full_image_regression_visualizations"

    # 预处理图像以获取尺寸
    _, original_image = preprocess_image(IMAGE_PATH)
    img_size = (original_image.shape[0], original_image.shape[1])
    print(f"图像尺寸: {img_size[1]}x{img_size[0]}")

    # 加载模型
    print("加载模型...")
    model = load_model(MODEL_PATH, img_size)

    # 可视化
    print("开始可视化...")
    visualize_regression_layers(
        model=model,
        image_path=IMAGE_PATH,
        target_layers=TARGET_LAYERS,
        save_dir=SAVE_DIR
    )
    print("可视化完成!")
