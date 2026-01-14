import json
import numpy as np
from collections import defaultdict


def load_detection_data(pred_path: str, gt_path: str) -> tuple:
    """加载预测结果和真实标注数据"""
    with open(pred_path, 'r', encoding='utf-8') as f:
        preds = json.load(f)  # 预测结果：list of dict，每个dict包含timestamp和detections
    with open(gt_path, 'r', encoding='utf-8') as f:
        gts = json.load(f)  # 真实标注：list of dict，格式同preds
    return preds, gts


def calculate_iou(box1: dict, box2: dict) -> float:
    """计算两个边界框的交并比（IoU）"""
    # 提取边界框坐标（x1, y1为左上角；x2, y2为右下角）
    x1_1, y1_1 = box1['x1'], box1['y1']
    x2_1, y2_1 = box1['x2'], box1['y2']
    x1_2, y1_2 = box2['x1'], box2['y1']
    x2_2, y2_2 = box2['x2'], box2['y2']

    # 计算交集区域坐标
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    # 计算交集面积
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    # 计算并集面积
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def evaluate(preds: list, gts: list) -> dict:
    """
    计算评价指标：mAP@0.5、漏检率、平均IoU、帧率
    preds: 预测结果列表，每个元素为{"timestamp": t, "detections": [预测框]}
    gts: 真实标注列表，格式同preds（检测框无confidence字段）
    """
    # 校验帧数是否一致
    if len(preds) != len(gts):
        raise ValueError(f"预测帧数量（{len(preds)}）与真实帧数量（{len(gts)}）不匹配")

    # --------------------------
    # 1. 初始化统计变量
    # --------------------------
    total_frames = len(preds)
    total_correct = 0  # 所有帧中检测正确的数量（IoU≥0.5且类别匹配）
    total_gt_objects = 0  # 所有帧中真实目标总数
    total_missed = 0  # 所有帧中漏检数量
    total_iou_sum = 0.0  # 所有预测框与最佳匹配真实框的IoU总和
    total_pred_objects = 0  # 所有帧中预测目标总数
    frame_precisions = []  # 每帧的检测精度（用于计算mAP@0.5）

    # --------------------------
    # 2. 帧率（FPS）计算
    # --------------------------
    if total_frames == 0:
        fps = 0.0
    else:
        start_time = preds[0]['timestamp']
        end_time = preds[-1]['timestamp']
        total_duration = end_time - start_time
        fps = total_frames / total_duration if total_duration > 0 else 0.0

    # --------------------------
    # 3. 逐帧处理
    # --------------------------
    for pred_frame, gt_frame in zip(preds, gts):
        # 提取当前帧的预测框和真实框
        pred_boxes = pred_frame['detections']  # 预测框：包含class_id、confidence、bbox
        gt_boxes = gt_frame['detections']  # 真实框：包含class_id、bbox

        # 统计当前帧的目标数量
        current_gt_count = len(gt_boxes)
        current_pred_count = len(pred_boxes)
        total_gt_objects += current_gt_count
        total_pred_objects += current_pred_count

        # 若当前帧无真实框，直接计算精度
        if current_gt_count == 0:
            frame_precision = 0.0 if current_pred_count > 0 else 1.0
            frame_precisions.append(frame_precision)
            continue

        # 按类别分组（只匹配同类别目标）
        pred_by_class = defaultdict(list)  # {class_id: [bbox1, bbox2, ...]}
        for box in pred_boxes:
            pred_by_class[box['class_id']].append(box['bbox'])

        gt_by_class = defaultdict(list)  # {class_id: [bbox1, bbox2, ...]}
        for box in gt_boxes:
            gt_by_class[box['class_id']].append(box['bbox'])

        # 记录已匹配的真实框索引（避免重复匹配）
        matched_gt_indices = defaultdict(set)  # {class_id: {已匹配的gt索引}}
        current_correct = 0
        current_iou_sum = 0.0

        # 遍历所有预测框，寻找最佳匹配的真实框
        for pred in pred_boxes:
            pred_class = pred['class_id']
            pred_bbox = pred['bbox']
            best_iou = 0.0
            best_gt_idx = -1

            # 只在同类别中匹配真实框
            if pred_class not in gt_by_class:
                # 无同类别真实框，匹配失败
                total_iou_sum += 0.0
                continue

            # 计算当前预测框与所有同类别未匹配真实框的IoU
            for gt_idx, gt_bbox in enumerate(gt_by_class[pred_class]):
                if gt_idx in matched_gt_indices[pred_class]:
                    continue  # 跳过已匹配的真实框
                iou = calculate_iou(pred_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # 累计IoU（无论是否匹配成功，均计入平均IoU计算）
            total_iou_sum += best_iou

            # 判断是否检测正确（IoU≥0.5且找到匹配的真实框）
            if best_iou >= 0.5 and best_gt_idx != -1:
                matched_gt_indices[pred_class].add(best_gt_idx)
                current_correct += 1

        # 计算当前帧精度并累计
        frame_precision = current_correct / current_gt_count
        frame_precisions.append(frame_precision)
        total_correct += current_correct

        # 计算当前帧漏检数（真实框总数 - 正确检测数）
        current_missed = current_gt_count - current_correct
        total_missed += current_missed

    # --------------------------
    # 4. 计算最终指标
    # --------------------------
    # mAP@0.5：所有帧精度的平均值
    mAP = np.mean(frame_precisions) if frame_precisions else 0.0

    # 漏检率：总漏检数 / 总真实目标数
    miss_rate = total_missed / total_gt_objects if total_gt_objects > 0 else 0.0

    # 平均IoU：所有预测框的IoU平均值
    avg_iou = total_iou_sum / total_pred_objects if total_pred_objects > 0 else 0.0

    return {
        "mAP@0.5": round(mAP, 4),
        "漏检率": round(miss_rate, 4),
        "平均IoU": round(avg_iou, 4),
        "FPS": round(fps, 2)
    }


# --------------------------
# 示例用法
# --------------------------
if __name__ == "__main__":
    # 替换为实际文件路径
    PREDICTION_PATH = "/amax/zs/code/WSCC_TAF-main/txt/output_pre.txt"  # 预测结果文件
    GROUND_TRUTH_PATH = "/amax/zs/code/WSCC_TAF-main/txt/output_real.txt"  # 真实标注文件

    # 加载数据
    preds, gts = load_detection_data(PREDICTION_PATH, GROUND_TRUTH_PATH)

    # 计算指标
    metrics = evaluate(preds, gts)

    # 输出结果
    print("评价指标计算结果：")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value}")