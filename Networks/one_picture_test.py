import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import os
from Networks.models import base_patch16_384_swin, base_patch16_384_swin_dk  # 导入对应模型类
from PIL import Image

# -------------------------- 配置参数 --------------------------
# 模型路径
model_path = "/amax/zs/code/WSCC_TAF-main/save_file/ShanghaiB_swin_mode3_bands2_dk/model_best.pth"
# 输入图像路径（替换为你的图片路径）
img_path = "/amax/zs/code/WSCC_TAF-main/datasets/ShanghaiTech/part_B_final/test_data/images/IMG_1.jpg"  # 例如："crowd_image.jpg"
# 模型参数（需与训练时一致）
input_size = (384, 384)  # 模型输入尺寸（384x384，与patch16_384对应）
mode = 3  # 路径中包含mode3，需与训练时一致
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------- 图像预处理 --------------------------
# 与训练时相同的预处理
transform = transforms.Compose([
    transforms.Resize(input_size),  # 调整尺寸为384x384
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 训练时使用的归一化参数
                         std=[0.229, 0.224, 0.225])
])


# -------------------------- 加载模型 --------------------------
def load_model(model_path, mode):
    # 初始化模型（根据路径判断为swin类型，mode=3）
    model = base_patch16_384_swin_dk(pretrained=False, mode=mode)  # 不加载预训练权重，用自己的模型
    # 加载训练好的权重
    checkpoint = torch.load(model_path, map_location=device)
    # 处理DataParallel保存的权重（若有module.前缀）
    state_dict = checkpoint['state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  # 移除module.前缀
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()  # 切换到评估模式
    return model


# -------------------------- 生成热力图 --------------------------
def generate_heatmap(model, img_path, input_size):
    # 读取原图
    img = Image.open(img_path).convert('RGB')
    orig_img = np.array(img)  # 保存原图用于可视化
    orig_h, orig_w = orig_img.shape[:2]

    # 预处理图像
    img_tensor = transform(img).unsqueeze(0)  # 添加批次维度 (1, 3, 384, 384)
    img_tensor = img_tensor.to(device)

    # 模型推理（获取密度图）
    with torch.no_grad():
        output = model(img_tensor)
        print("模型输出结构：", output.shape if not isinstance(output, tuple) else [o.shape for o in output])
        print("模型输出结构：", output[0].shape)  # 应为[1, 1, H, W]
        print("密度图统计：最大值={}, 最小值={}, 平均值={}".format(
            output[0].max().item(),
            output[0].min().item(),
            output[0].mean().item()
        ))
        # 若模型输出为元组，取第一个元素作为密度图（根据训练代码推测）
        density_map = output[0] if isinstance(output, tuple) else output
        density_map = density_map.squeeze(0).squeeze(0)  # 去除批次和通道维度 (384, 384)
        density_map = density_map.cpu().numpy()  # 转换为numpy数组

    # 将密度图 resize 到原图尺寸（保持比例一致）
    density_map_resized = cv2.resize(density_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # 计算总人数（密度图求和）
    total_count = density_map_resized.sum()

    return orig_img, density_map_resized, total_count


# -------------------------- 可视化结果 --------------------------
def visualize(orig_img, density_map, total_count):
    plt.figure(figsize=(12, 6))

    # 显示原图
    plt.subplot(121)
    plt.imshow(orig_img)
    plt.title("Original Image")
    plt.axis('off')

    # 显示热力图（叠加在原图上）
    plt.subplot(122)
    plt.imshow(orig_img, alpha=0.5)  # 原图半透明
    plt.imshow(density_map, cmap='jet', alpha=0.5)  # 热力图叠加
    plt.colorbar(label='People Density')
    plt.title(f"Heatmap (Total Count: {total_count:.1f})")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    # 加载模型
    model = load_model(model_path, mode=3)
    # 生成热力图
    orig_img, heatmap, count = generate_heatmap(model, img_path, input_size)
    # 可视化
    visualize(orig_img, heatmap, count)