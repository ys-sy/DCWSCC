

import matplotlib.pyplot as plt
import numpy as np

# ----------------------
# 1. 全局配置（匹配论文级可视化风格）
# ----------------------
plt.rcParams["font.family"] = "Arial"  # 贴近学术图表的无衬线字体
plt.rcParams["figure.figsize"] = (8, 5)  # 图像比例（宽×高）
plt.rcParams["axes.linewidth"] = 1.2  # 坐标轴边线加粗
plt.rcParams["xtick.direction"] = "in"  # 刻度朝内（更专业）
plt.rcParams["ytick.direction"] = "in"


# ----------------------
# 2. 数据加载（模拟数据/真实数据替换区）
# ----------------------
def load_data():
    """
    说明：
    1. 模拟数据仅作演示，实际需替换为真实实验数据！
    2. 真实数据建议从 CSV/Excel 读取，格式为：
       - epochs: 0~160的连续整数（共161个点）
       - 三类模型的MAE值：csrnet_mae, trans_token_mae, trans_gap_mae
    """
    # 模拟数据（替换为真实数据时删除以下代码）
    np.random.seed(0)
    epochs = np.arange(0, 161, 1)  # 强制0~160连续刻度

    # 模拟三类模型的MAE波动（大致匹配示例趋势）
    csrnet_mae = np.random.normal(100, 30, size=161)
    csrnet_mae[0] = 250  # 初始值
    csrnet_mae[140] = 68.2  # 标注点（140轮）

    trans_token_mae = np.random.normal(80, 20, size=161)
    trans_token_mae[0] = 300  # 初始值
    trans_token_mae[120] = 69.0  # 标注点（120轮）

    trans_gap_mae = np.random.normal(70, 15, size=161)
    trans_gap_mae[0] = 280  # 初始值
    trans_gap_mae[80] = 66.1  # 标注点（80轮）

    return {
        "CSRNet": (epochs, csrnet_mae, "blue"),
        "TransCrowd-Token": (epochs, trans_token_mae, "green"),
        "TransCrowd-GAP": (epochs, trans_gap_mae, "red")
    }


# 加载数据（真实数据替换后需确保格式一致）
data = load_data()

# ----------------------
# 3. 绘制主图（严格匹配坐标需求）
# ----------------------
fig, ax = plt.subplots()

# 循环绘制三条曲线
for model_name, (epochs, mae, color) in data.items():
    ax.plot(epochs, mae, color=color, label=model_name, linewidth=1.2)

# ----------------------
# 4. 坐标轴精准控制（核心需求）
# ----------------------
# X轴：0起步无空白，刻度0/20/40…160
ax.set_xlim(0, 160)  # 左边界严格对齐0
ax.set_xticks(np.arange(0, 161, 20))  # 强制20间隔刻度

# Y轴：刻度60/120/240/480
ax.set_ylim(60, 480)  # 上下限匹配需求
ax.set_yticks([60, 120, 240, 480])  # 严格显示目标刻度

# 坐标轴标签（与需求一致）
ax.set_xlabel("Epochs", fontsize=12, fontweight="bold")
ax.set_ylabel("MAE", fontsize=12, fontweight="bold")

# ----------------------
# 5. 关键标注（还原66.1/69.0/68.2）
# ----------------------
annotations = [
    # (x坐标, y坐标, 标注文本, 颜色)
    (80, 66.1, "66.1", "red"),  # TransCrowd-GAP @ 80轮
    (120, 69.0, "69.0", "green"),  # TransCrowd-Token @ 120轮
    (140, 68.2, "68.2", "blue")  # CSRNet @ 140轮
]

for x, y, text, color in annotations:
    ax.text(
        x, y - 10,  # 文本在点下方10个单位（避免遮挡曲线）
        text,
        color=color,
        ha="center",  # 水平居中
        fontsize=9,
        fontweight="bold"
    )

# ----------------------
# 6. 图例与网格（学术图表风格）
# ----------------------
ax.legend(
    loc="upper right",  # 图例位置
    fontsize=10,
    frameon=True,  # 显示图例边框
    edgecolor="gray",  # 边框颜色
    facecolor="white",  # 背景色（透明感）
    framealpha=0.8  # 透明度
)

ax.grid(
    True,
    linestyle="--",  # 虚线网格
    alpha=0.5,  # 透明度
    color="lightgray"  # 浅灰色（不抢曲线焦点）
)

# ----------------------
# 7. 保存/显示图像（可选扩展）
# ----------------------
# 如需保存高清图（300dpi）：
# plt.savefig("mae_curve.png", dpi=300, bbox_inches="tight")

# 显示图像
plt.tight_layout()  # 自动优化布局（避免标签截断）
plt.show()



import pandas as pd
def load_data():
    df = pd.read_csv("/amax/zs/code/WSCC_TAF-main/txt/Default Dataset.xlsx")  # 或 pd.read_excel(...)
    return {
        "CSRNet": (df["epochs"], df["csrnet_mae"], "blue"),
        "TransCrowd-Token": (df["epochs"], df["trans_token_mae"], "green"),
        "TransCrowd-GAP": (df["epochs"], df["trans_gap_mae"], "red")
    }