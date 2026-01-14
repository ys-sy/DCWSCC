import cv2
import numpy as np
import torch
import torch.nn as nn
# 请根据实际路径调整模型导入，确保 base_patch16_384_swin_cdpnet3 能正确导入
from Networks.models import base_patch16_384_swin_cdpnet3


# -------------------------- 1. 加载模型 --------------------------
def load_model(model_path, device):
    """加载模型结构并加载训练好的权重"""
    model = base_patch16_384_swin_cdpnet3(pretrained=False, mode=3)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    return model.to(device)


# -------------------------- 2. 图像预处理 --------------------------
def preprocess_image(img_path, device):
    """预处理图像，与训练时保持一致，并返回原图尺寸"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    original_height, original_width = img.shape[:2]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (384, 384))
    img = img / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float().unsqueeze(0)
    return img.to(device), (original_width, original_height)


# -------------------------- 3. 提取regression层特征 --------------------------
def extract_regression_feature(model, input_tensor):
    """提取regression层的输出特征"""
    regression_feature = None

    def hook_fn(module, input, output):
        nonlocal regression_feature
        regression_feature = output.detach()

    handle = model.regression.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(input_tensor)
    handle.remove()
    return regression_feature


# -------------------------- 4. 可视化前N个通道特征（类似示例风格） --------------------------
def visualize_top_channels(feature, original_size, top_n=50, save_path="merged_feature.jpg"):
    """
    可视化前N个通道特征，将每个通道归一化到0-255后，按通道维度拼接或混合展示（这里简单取前3通道示例，可按需改）
    实际若要更贴近你给的示例，可调整通道映射逻辑，比如直接缩放通道值到0-255显示
    """
    B, C, H, W = feature.shape
    print(f"特征图信息：通道数 {C}，空间尺寸 {H}x{W}")

    # 取前top_n个通道
    if C > top_n:
        feature = feature[:, :top_n, :, :]
    else:
        top_n = C

    # 这里演示：对每个通道单独归一化后，简单取前3个通道拼为RGB（可根据需求调整，比如加权混合等）
    # 先初始化一个用于展示的空白特征图（单通道示例，多通道可扩展）
    display_feature = torch.zeros((H, W), dtype=torch.float32).to(feature.device)
    for i in range(min(top_n, 3)):  # 先简单用前3通道示例，想全用可改循环逻辑
        channel_data = feature[0, i, :, :]
        # 通道内归一化到0-1
        channel_min = torch.min(channel_data)
        channel_max = torch.max(channel_data)
        normalized_channel = (channel_data - channel_min) / (channel_max - channel_min + 1e-8)
        # 叠加到显示图（这里简单相加，也可按权重、通道顺序等调整）
        display_feature += normalized_channel

    # 整体归一化到0-255
    display_min = torch.min(display_feature)
    display_max = torch.max(display_feature)
    display_feature = (display_feature - display_min) / (display_max - display_min + 1e-8)
    display_feature = (display_feature * 255).byte().cpu().numpy()

    # 调整到原图尺寸
    display_feature = cv2.resize(display_feature, original_size, interpolation=cv2.INTER_NEAREST)
    # 转换为彩色（这里是单通道伪彩色，若要真彩色可扩展多通道映射）
    colored = cv2.applyColorMap(display_feature, cv2.COLORMAP_JET)  # 用JET伪彩色，接近示例风格
    cv2.imwrite(save_path, colored)
    print(f"特征可视化图已保存至：{save_path}")
    return colored


# -------------------------- 主函数 --------------------------
def main():
    model_path = "/amax/zs/code/WSCC_TAF-main/save_file/ShanghaiA_swincdpnet3/model_best.pth"
    img_path = "/amax/zs/code/WSCC_TAF-main/datasets/ShanghaiTech/part_A_final/test_data/images/IMG_167.jpg"
    merged_save_path = "merged_feature_custom167.jpg"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")

    # 加载模型
    print("加载模型...")
    model = load_model(model_path, device)

    # 预处理图像 + 获取原图尺寸
    print("预处理图像...")
    input_tensor, original_size = preprocess_image(img_path, device)

    # 提取特征
    print("提取regression层特征...")
    regression_feature = extract_regression_feature(model, input_tensor)
    if regression_feature is None:
        raise ValueError("未能提取到特征，请检查模型结构或钩子注册是否正确")
    print(f"提取的特征形状：{regression_feature.shape}")

    # 可视化特征（生成类似示例的风格）
    print("可视化特征图...")
    visualize_top_channels(regression_feature, original_size, top_n=50, save_path=merged_save_path)


if __name__ == "__main__":
    main()