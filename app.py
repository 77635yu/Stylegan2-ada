import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import gradio as gr

# ========== 路径配置 ==========
STYLEGAN_REPO_PATH = r"D:\GraduationProject\stylegan2-ada-pytorch"
STYLEGAN_PKL_PATH = r"D:\GraduationProject\training_runs\00000-anime-auto1-kimg2000-batch8-noaug\network-snapshot-001600.pkl"
UPSCALE_MODEL_PATH = r"D:\GraduationProject\weights\RealESRGAN_x4plus_anime_6B.pth"
# 建议改为绝对路径，避免工作目录依赖
SEFA_PATH = r"D:\GraduationProject\sefa_results\sefa_directions.npy"

# 检查关键路径
for path in [STYLEGAN_REPO_PATH, STYLEGAN_PKL_PATH, UPSCALE_MODEL_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到路径: {path}")
sys.path.append(STYLEGAN_REPO_PATH)

import dnnlib
import legacy

# ========== RRDBNet 超分网络定义 ==========
def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=6, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))

# ========== 全局模型加载 ==========
print("正在初始化 Web 应用，请稍候...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Loading StyleGAN: {STYLEGAN_PKL_PATH}...")
with dnnlib.util.open_url(STYLEGAN_PKL_PATH) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device).eval()

print("Loading SeFa Control Vectors...")
if not os.path.exists(SEFA_PATH):
    print("警告: 找不到 SeFa 文件，属性控制可能失效。")
    dir_hair = 0
else:
    directions = np.load(SEFA_PATH)
    dir_hair = torch.from_numpy(directions[6]).float().to(device)  # 发色

print(f"Loading Real-ESRGAN: {UPSCALE_MODEL_PATH}...")
upscaler_net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
loadnet = torch.load(UPSCALE_MODEL_PATH, map_location=device)
keyname = 'params_ema' if 'params_ema' in loadnet else 'params'
upscaler_net.load_state_dict(loadnet[keyname], strict=True)
upscaler_net.eval().to(device)

print("模型加载完成！")

# ========== 预设颜色映射 ==========
PRESET_COLORS = {
    "默认 (0.0)": 0.0,  
    "🟩 绿色 (-5.0)": -5.0,
    "🩵 青色 (-3.0)": -3.0,
    "⬛ 黑色 (-1.0)": -1.0,      
    "🟫 棕色 (4.0)": 4.0,                
}

# ========== 核心生成函数（批处理优化） ==========
def generate_four(seed, truncation_psi, hair_preset, hair_fine):
    """
    一次性批量生成4张头像，大幅提升速度。
    同时通过替换色彩层保持发色统一。
    """
    try:
        base_seed = int(seed)
        hair_alpha = PRESET_COLORS.get(hair_preset, 0.0) + hair_fine

        # 1. 生成基准样式的 w，用于统一色彩
        z_base = torch.from_numpy(np.random.RandomState(base_seed).randn(1, G.z_dim)).to(device)
        label_base = torch.zeros([1, G.c_dim], device=device)
        if G.c_dim != 0:
            label_base[:, 0] = 1
        w_base = G.mapping(z_base, label_base, truncation_psi=truncation_psi)

        if isinstance(dir_hair, torch.Tensor):
            w_color_base = w_base + (hair_alpha * dir_hair)
        else:
            w_color_base = w_base

        # 2. 批量准备四个种子的潜在向量
        z_list = []
        for i in range(4):
            current_seed = base_seed + i
            z_i = np.random.RandomState(current_seed).randn(1, G.z_dim)
            z_list.append(z_i)
        z_curr = torch.from_numpy(np.concatenate(z_list, axis=0)).to(device)   # (4, z_dim)

        label_curr = torch.zeros([4, G.c_dim], device=device)
        if G.c_dim != 0:
            label_curr[:, 0] = 1

        # 3. 一次性映射得到所有 w，然后混合色彩层
        w_curr = G.mapping(z_curr, label_curr, truncation_psi=truncation_psi)   # (4, num_ws, 512)
        w_mixed = w_curr.clone()
        color_layers = slice(6, 9)           # 控制发色的层，需要根据实际模型调整
        w_mixed[:, color_layers, :] = w_color_base[:, color_layers, :]

        # 4. 批量合成图像
        with torch.no_grad():
            img_batch = G.synthesis(w_mixed, noise_mode='const')   # (4, 3, H, W)

        # 5. 后处理并转为 PIL 列表
        img_batch = (img_batch.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        generated_images = []
        for i in range(4):
            img_np = img_batch[i].cpu().numpy()
            pil_img = Image.fromarray(img_np, 'RGB')
            generated_images.append((pil_img, f"Seed: {base_seed + i}"))

        return generated_images, f"生成成功! 基准Seed: {base_seed} (色彩已统一 | 发色: {hair_alpha:.1f})"

    except Exception as e:
        return None, f"生成错误: {str(e)}"

# ========== 随机生成入口 ==========
def random_gen(truncation_psi, hair_preset, hair_fine):
    rnd_seed = np.random.randint(0, 1000000)
    return generate_four(rnd_seed, truncation_psi, hair_preset, hair_fine)

# ========== 超分修复函数（修复了 Gallery 取图 Bug，并简化色彩处理） ==========
def upscale_selected(evt: gr.SelectData, gallery_data, truncation_psi, history_state):
    try:
        # 修复点1：安全地从 Gallery 组件取值
        selected_item = gallery_data[evt.index]
        img_path = selected_item[0]['name']          # Gallery 缓存字典的 key
        caption = selected_item[1] if len(selected_item) > 1 else "Selected"

        # 修复点5：直接使用 RGB 流程，避免 BGR 反复转换
        pil_orig = Image.open(img_path).convert('RGB')
        img_np = np.array(pil_orig).astype(np.float32) / 255.0          # H, W, C, RGB, [0,1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)   # (1,3,H,W)

        with torch.no_grad():
            output = upscaler_net(img_tensor)          # (1,3,H',W')

        # 转回 PIL 图像
        output = output.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).byte().cpu().numpy()
        pil_upscaled = Image.fromarray(output, 'RGB')

        status_msg = f"成功修复! {caption}"

        # 更新历史记录
        if history_state is None:
            history_state = []
        hist_caption = f"{caption} | Psi: {truncation_psi}"
        history_state.insert(0, (pil_upscaled, hist_caption))
        if len(history_state) > 20:
            history_state = history_state[:20]

        return pil_upscaled, status_msg, history_state, history_state

    except Exception as e:
        return None, f"修复错误: {str(e)}", history_state, history_state

# ========== Gradio 界面 ==========
with gr.Blocks(title="动漫头像生成与修复系统") as demo:
    session_history = gr.State([])

    gr.Markdown("# 🎨 动漫头像生成与定制系统")
    gr.Markdown("毕业设计演示：左侧设置参数并生成四宫格，**点击你最喜欢的一张**进行高清修复(256x256)并存入历史记录。")

    with gr.Row():
        with gr.Column(scale=1):
            seed_input = gr.Number(label="基准种子 (Base Seed)", value=60, precision=0)
            psi_slider = gr.Slider(minimum=0, maximum=1.5, value=0.7, label="多样性截断 (Psi)")

            gr.Markdown("### ✨ 特征控制")
            gr.Markdown("选择基础色调后，可使用滑块进行微调")

            color_choices = list(PRESET_COLORS.keys())
            hair_preset = gr.Radio(choices=color_choices, value="默认 (0.0)", label="💇‍♀️ 发色基础预设")
            hair_fine = gr.Slider(minimum=-5.0, maximum=5.0, step=0.1, value=0.0, label="发色高级微调")

            btn_generate = gr.Button("指定种子生成4张", variant="primary")
            btn_random = gr.Button("🎲 随机抽卡")
            status_output = gr.Textbox(label="状态信息", interactive=False)

        with gr.Column(scale=2):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 1. 原始生成 (请点击下方图片进行修复)")
                    img_output_orig_gallery = gr.Gallery(
                        label="原图 (64x64)", columns=2, rows=2, object_fit="contain", height=400
                    )
                with gr.Column():
                    gr.Markdown("### 2. 高清修复结果")
                    img_output_up = gr.Image(label="Upscaled (256x256)", type="pil")

        with gr.Column(scale=1):
            gr.Markdown("### 🕒 历史修复记录 (最多20张)")
            history_gallery = gr.Gallery(label="收藏夹", show_label=False, columns=2, object_fit="contain", height=500)

    # 事件绑定
    gen_inputs = [seed_input, psi_slider, hair_preset, hair_fine]
    btn_generate.click(fn=generate_four, inputs=gen_inputs, outputs=[img_output_orig_gallery, status_output])

    rnd_inputs = [psi_slider, hair_preset, hair_fine]
    btn_random.click(fn=random_gen, inputs=rnd_inputs, outputs=[img_output_orig_gallery, status_output])

    img_output_orig_gallery.select(
        fn=upscale_selected,
        inputs=[img_output_orig_gallery, psi_slider, session_history],
        outputs=[img_output_up, status_output, session_history, history_gallery]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, share=False)