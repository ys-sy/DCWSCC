import cv2
import torch
import numpy as np
import scipy.io as io
from torchvision import transforms
from Networks.models import base_patch16_384_swin_cdpnet3  # 导入模型定义
import torch.nn as nn


def load_model(checkpoint_path, device='cuda'):
    """加载训练好的模型"""
    # 初始化模型（与训练时保持一致的参数）
    model = base_patch16_384_swin_cdpnet3(pretrained=False, mode=3)  # mode需与训练时一致

    # 加载模型权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 处理DataParallel保存的权重（若训练时使用了多卡）
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 去除可能的module.前缀（单卡推理时）
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)

    # 模型设置为评估模式
    model = model.to(device)
    model.eval()
    return model


def split_image_into_regions(image_path):
    """将图像分割为四个区域：左上、右上、左下、右下"""
    # 读取图像（OpenCV默认BGR格式）
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 获取图像尺寸
    height, width = img.shape[:2]

    # 计算分割点
    mid_width = width // 2
    mid_height = height // 2

    # 分割为四个区域
    regions = {
        "左上": img[0:mid_height, 0:mid_width],
        "右上": img[0:mid_height, mid_width:width],
        "左下": img[mid_height:height, 0:mid_width],
        "右下": img[mid_height:height, mid_width:width]
    }

    # 返回分割后的区域和原始图像尺寸（用于计算真实人数）
    return regions, (width, height)


def preprocess_region(region, input_size=(384, 384)):
    """预处理分割后的区域图像"""
    # 转换为RGB格式（与训练数据一致）
    img = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)

    # 调整尺寸到模型输入大小
    img = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)

    # 转换为Tensor并归一化（与训练时的预处理一致）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet均值
            std=[0.229, 0.224, 0.225]  # ImageNet标准差
        )
    ])
    img_tensor = transform(img).unsqueeze(0)  # 添加batch维度
    return img_tensor


def predict_count(model, image_tensor, device='cuda'):
    """预测图像中的人数"""
    with torch.no_grad():  # 关闭梯度计算，加速推理
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)  # 模型输出
        count = output.item()  # 转换为标量
    return round(count)  # 人数通常取整


def get_region_gt_counts(mat_path, image_size):
    """从.mat标注文件中获取每个区域的真实人数"""
    try:
        # 加载.mat文件
        mat_data = io.loadmat(mat_path)
        width, height = image_size

        # 提取标注的坐标数据（上海Tech数据集的标注结构）
        gt_points = mat_data["image_info"][0][0][0][0][0]  # 形状为 (N, 2)，N为人数

        # 计算分割点
        mid_width = width // 2
        mid_height = height // 2

        # 初始化各区域人数计数器
        region_counts = {
            "左上": 0,
            "右上": 0,
            "左下": 0,
            "右下": 0
        }

        # 统计每个区域的人数
        for (x, y) in gt_points:
            # 判断点所在的区域（注意：坐标可能是(x,y)对应(width,height)）
            if x <= mid_width and y <= mid_height:
                region_counts["左上"] += 1
            elif x > mid_width and y <= mid_height:
                region_counts["右上"] += 1
            elif x <= mid_width and y > mid_height:
                region_counts["左下"] += 1
            else:
                region_counts["右下"] += 1

        return region_counts

    except FileNotFoundError:
        raise ValueError(f"标注文件不存在: {mat_path}")
    except KeyError as e:
        raise ValueError(f"标注文件结构异常，缺少关键字段: {e}")
    except Exception as e:
        raise ValueError(f"读取标注文件失败: {str(e)}")


def main():
    # 配置参数
    image_path = "/amax/zs/code/WSCC_TAF-main/datasets/ShanghaiTech/part_A_final/train_data/images/IMG_32.jpg"
    checkpoint_path = "/amax/zs/code/WSCC_TAF-main/save_file/ShanghaiA_swincdpnet3/model_best.pth"
    mat_path = "/amax/zs/code/WSCC_TAF-main/datasets/ShanghaiTech/part_A_final/train_data/ground_truth/GT_IMG_32.mat"  # 对应的标注文件路径
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        # 加载模型
        print("加载模型...")
        model = load_model(checkpoint_path, device)

        # 分割图像为四个区域
        print("分割图像为四个区域...")
        regions, image_size = split_image_into_regions(image_path)

        # 获取每个区域的真实人数
        print("读取各区域真实人数...")
        region_gt_counts = get_region_gt_counts(mat_path, image_size)

        # 输出结果标题
        print("\n===== 各区域人数统计 =====")
        print(f"图像路径: {image_path}")
        print(f"标注文件路径: {mat_path}")
        print("----------------------------------------")

        # 总人数统计
        total_pred = 0
        total_gt = 0

        # 逐个处理每个区域
        for region_name, region_img in regions.items():
            # 预处理区域图像
            region_tensor = preprocess_region(region_img)

            # 预测区域人数
            pred_count = predict_count(model, region_tensor, device)

            # 获取该区域的真实人数
            gt_count = region_gt_counts[region_name]

            # 累加总人数
            total_pred += pred_count
            total_gt += gt_count

            # 输出区域结果
            print(f"{region_name}区域:")
            print(f"  预测人数: {pred_count}")
            print(f"  真实人数: {gt_count}")
            print(f"  误差: {abs(pred_count - gt_count)}")
            print("----------------------------------------")

        # 输出总结果
        print("总结果:")
        print(f"  总预测人数: {total_pred}")
        print(f"  总真实人数: {total_gt}")
        print(f"  总误差: {abs(total_pred - total_gt)}")

    except Exception as e:
        print(f"执行失败: {e}")


if __name__ == "__main__":
    main()
