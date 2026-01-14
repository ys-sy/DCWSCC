# save as visualize_swin_features.py (直接复制运行)
import os
import math
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from Networks.models import base_patch16_384_swin_cdpnet3
from Networks.swin_transformer_CDPNet3 import SwinTransformer_CDPNET3

# ----------------- 配置（请修改这两个路径） -----------------
WEIGHTS_PATH = "/amax/zs/code/WSCC_TAF-main/save_file/ShanghaiA_swincdpnet3/model_best.pth"
IMG_PATH = "/amax/zs/code/WSCC_TAF-main/datasets/ShanghaiTech/part_A_final/test_data/images/IMG_161.jpg"
OUTPUT_DIR = "./feat_vis_out"
MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- 载入模型 ----
try:
    model = base_patch16_384_swin_cdpnet3(pretrained=False, mode=3)
except Exception:
    model = SwinTransformer_CDPNET3(img_size=384, patch_size=4, in_chans=3, num_classes=1,
                                    embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32],
                                    window_size=12, use_checkpoint=True, mode=3)

model.to(MODEL_DEVICE)
model.eval()

# ---- 加载权重 ----
if os.path.exists(WEIGHTS_PATH):
    ck = torch.load(WEIGHTS_PATH, map_location="cpu")
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state, strict=False)
    print("Loaded weights from", WEIGHTS_PATH)
else:
    print("Warning: WEIGHTS_PATH not found, using random init.")

# ---- 注册 hook ----
feat_dict = {}
def make_layer_hook(name):
    def hook(module, input, output):
        try:
            before = output[1]  # BasicLayer 返回 (x_after, x_before)
        except Exception:
            before = output
        feat_dict[name] = before.detach().cpu()
    return hook

# 捕获 x1, x2, x3, x4
for i in range(len(model.layers)):
    model.layers[i].register_forward_hook(make_layer_hook(f"stage{i+1}"))

# 捕获 regression 输出
def regression_hook(module, input, output):
    feat_dict["regression"] = output.detach().cpu()
model.regression.register_forward_hook(regression_hook)

# 捕获 post_conv 输出
def post_conv_hook(module, input, output):
    feat_dict["post_conv"] = output.detach().cpu()
model.post_conv.register_forward_hook(post_conv_hook)

# ---- 读图预处理 ----
orig_img = Image.open(IMG_PATH).convert("RGB")
orig_w, orig_h = orig_img.size
MODEL_INPUT_SIZE = (384, 384)

preprocess = transforms.Compose([
    transforms.Resize(MODEL_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
input_tensor = preprocess(orig_img).unsqueeze(0).to(MODEL_DEVICE)

# ---- 前向推理 ----
with torch.no_grad():
    _ = model(input_tensor)

print("Captured stages:", list(feat_dict.keys()))

# ---- 特征转热图 ----
def feat_to_heatmap_tensor(feat_tensor, target_hw):
    t = feat_tensor
    if t.ndim == 3:  # [B, L, C]
        B, L, C = t.shape
        Hf = Wf = int(math.sqrt(L))
        t = t.permute(0, 2, 1).reshape(B, C, Hf, Wf)
    elif t.ndim == 4:  # [B, C, H, W]
        pass
    else:
        raise ValueError("Unsupported shape: " + str(t.shape))

    heat = t.mean(dim=1, keepdim=True)
    heat_up = F.interpolate(heat, size=MODEL_INPUT_SIZE, mode='bilinear', align_corners=False)
    heat_up = F.interpolate(heat_up, size=(target_hw[1], target_hw[0]), mode='bilinear', align_corners=False)
    heat_up = heat_up.squeeze(0).squeeze(0).cpu().numpy()

    heat_up = heat_up - heat_up.min()
    denom = heat_up.max() if heat_up.max() != 0 else 1.0
    heat_norm = heat_up / denom

    hm_u8 = (heat_norm * 255.0).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)

    return hm_color, heat_norm

# ---- 保存可视化 ----
# ---- 保存可视化 ----
orig_cv_bgr = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
base_name = os.path.splitext(os.path.basename(IMG_PATH))[0]

for name, feat in feat_dict.items():
    if name == "post_conv":
        # post_conv 是频域 [B, C, L]，转换回空间域 [B, C, H, W]
        B, C, L = feat.shape
        H = W = int(L ** 0.5)
        feat_spatial = feat.view(B, C, H, W)
        hm_color, _ = feat_to_heatmap_tensor(feat_spatial, target_hw=(orig_w, orig_h))
    else:
        hm_color, _ = feat_to_heatmap_tensor(feat, target_hw=(orig_w, orig_h))

    overlay = cv2.addWeighted(orig_cv_bgr, 0.6, hm_color, 0.4, 0)
    out_hm_path = os.path.join(OUTPUT_DIR, f"{base_name}_{name}_heatmap.png")
    out_ov_path = os.path.join(OUTPUT_DIR, f"{base_name}_{name}_overlay.png")
    cv2.imwrite(out_hm_path, hm_color)
    cv2.imwrite(out_ov_path, overlay)
    print(f"Saved {out_hm_path} and {out_ov_path}")

print("All done. Results in:", OUTPUT_DIR)

