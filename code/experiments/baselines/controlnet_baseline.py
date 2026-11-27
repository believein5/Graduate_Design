"""ControlNet 基线（CRG 论文 "ControlNet (Text Cond.)"）：文本 + 结构图条件的全图生成

python controlnet_baseline.py train   在划分 A 的 300 张真实训练裁剪上微调 ControlNet（SD 1.5 冻结）
python controlnet_baseline.py gen     对 jobs_A 的 300 个任务生成图像，写到 data/synthetic/controlnet_A/
python controlnet_baseline.py train board / gen board   同上，用按板划分的训练裁剪与 jobs 的前 300 个任务（data/synthetic/controlnet/）

条件（论文：文本描述 + 辅助结构图）：
- 文本："a {类别} defect on a printed circuit board"
- 结构图：Canny 边缘图，缺陷框区域置空（训练时取真实裁剪；生成时取 jobs_A 的无缺陷背景 + 放置框），
  让模型在空出的区域内按文本生成缺陷，其余部分跟随背景结构
标签：与其他方法相同（放置框尺寸，以 64 窗口中心为中心）。环境：conda env "baselines"（diffusers）。
"""
import csv
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "A"                # A：按照片划分（默认）；board：按板划分
SYN = ROOT / "data" / "synthetic"
DET = ROOT / "data" / ("detection_A" if SPLIT == "A" else "detection")
JOBS = SYN / ("jobs_A" if SPLIT == "A" else "jobs")
OUT = SYN / ("controlnet_A" if SPLIT == "A" else "controlnet")
CKPT = ROOT / "runs" / "baselines" / ("controlnet" if SPLIT == "A" else "controlnet_board")
BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
INIT_CN = "lllyasviel/control_v11p_sd15_canny"
NAMES = {"missinghole": "missing hole", "mousebite": "mouse bite", "opencircuit": "open circuit",
         "short": "short circuit", "spur": "spur", "spuriouscopper": "spurious copper"}
CLASSES = list(NAMES)
SIZE, GEN_SIZE, PATCH = 256, 512, 64          # SD 1.5 在 512 上工作：放大到 512 训练/生成，再缩回 256
STEPS, MICRO, ACCUM, LR = 3000, 2, 4, 1e-5   # 有效 batch 8，与原设置相同
MIN_FREE = int(os.environ.get("CN_MIN_FREE", 1500))   # GPU 空闲显存低于此值时保存并停止；队列中由 gpu_queue 控制显存，设 0 关闭
DEV = "cuda"


def prompt(c):
    return f"a {NAMES[c]} defect on a printed circuit board, close-up photo"


def control_map(img, boxes):
    """Canny 边缘图，缺陷框（外扩 4 px）内置空"""
    g = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    e = cv2.Canny(g, 60, 140)
    for x0, y0, x1, y1 in boxes:
        e[max(0, int(y0) - 4):int(y1) + 4, max(0, int(x0) - 4):int(x1) + 4] = 0
    e = cv2.resize(e, (GEN_SIZE, GEN_SIZE), interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(np.stack([e] * 3, -1))


def train_items():
    items = []
    for lp in sorted((DET / "train" / "labels").glob("*.txt")):
        lines = [l.split() for l in lp.read_text().splitlines() if l.strip()]
        if not lines:
            continue
        bx = [((float(cx) - float(w) / 2) * SIZE, (float(cy) - float(h) / 2) * SIZE,
               (float(cx) + float(w) / 2) * SIZE, (float(cy) + float(h) / 2) * SIZE) for _, cx, cy, w, h in lines]
        img = Image.open(DET / "train" / "images" / f"{lp.stem}.png").convert("RGB")
        items.append((img, control_map(img, bx), prompt(CLASSES[int(lines[0][0])])))
    return items


def free_mib():
    import subprocess
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(out.split()[0])


def train():
    """省显存设置（实测峰值约 6.7 GB，可与 256 模型训练同时运行）：
    文本编码与 VAE 潜变量预先算好并放回 CPU；UNet 冻结且为 bf16；梯度检查点；8-bit AdamW；batch 2 × 梯度累积 4"""
    import bitsandbytes as bnb
    from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    torch.manual_seed(0); random.seed(0)
    items = train_items()
    print(f"{len(items)} training crops", flush=True)

    # 1) 预计算：每张图（含水平翻转）的潜变量、控制图与文本编码，之后释放 VAE 和文本编码器
    tok = CLIPTokenizer.from_pretrained(BASE, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(BASE, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(DEV).eval()
    vae = AutoencoderKL.from_pretrained(BASE, subfolder="vae", torch_dtype=torch.bfloat16).to(DEV).eval()

    def tensor(im, norm):
        a = torch.from_numpy(np.asarray(im.resize((GEN_SIZE, GEN_SIZE), Image.BICUBIC), np.float32) / 255).permute(2, 0, 1)
        return a * 2 - 1 if norm else a

    cache = []
    with torch.no_grad():
        for img, cond, p in items:
            for flip in (False, True):
                i2, c2 = (img.transpose(Image.FLIP_LEFT_RIGHT), cond.transpose(Image.FLIP_LEFT_RIGHT)) if flip else (img, cond)
                lat = vae.encode(tensor(i2, True)[None].to(DEV, torch.bfloat16)).latent_dist.sample() * vae.config.scaling_factor
                ids = tok([p], padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids
                cache.append((lat[0].float().cpu(), tensor(c2, False), te(ids.to(DEV))[0][0].float().cpu()))
    del te, vae; torch.cuda.empty_cache()
    print(f"cached {len(cache)} samples", flush=True)

    # 2) 只把冻结 UNet（bf16）和待训练的 ControlNet 放在 GPU 上
    unet = UNet2DConditionModel.from_pretrained(BASE, subfolder="unet", torch_dtype=torch.bfloat16).to(DEV).eval().requires_grad_(False)
    cn = ControlNetModel.from_pretrained(INIT_CN).to(DEV).train()
    unet.enable_gradient_checkpointing(); cn.enable_gradient_checkpointing()
    sched = DDPMScheduler.from_pretrained(BASE, subfolder="scheduler")
    opt = bnb.optim.AdamW8bit(cn.parameters(), lr=LR)
    CKPT.mkdir(parents=True, exist_ok=True)
    for step in range(1, STEPS + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(ACCUM):
            batch = random.sample(cache, MICRO)
            lat = torch.stack([b[0] for b in batch]).to(DEV)
            cond = torch.stack([b[1] for b in batch]).to(DEV)
            emb = torch.stack([b[2] for b in batch]).to(DEV)
            noise = torch.randn_like(lat)
            t = torch.randint(0, sched.config.num_train_timesteps, (lat.shape[0],), device=DEV)
            noisy = sched.add_noise(lat, noise, t)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                down, mid = cn(noisy, t, encoder_hidden_states=emb, controlnet_cond=cond, return_dict=False)
                pred = unet(noisy, t, encoder_hidden_states=emb, down_block_additional_residuals=down,
                            mid_block_additional_residual=mid).sample
                loss = torch.nn.functional.mse_loss(pred.float(), noise.float()) / ACCUM
            loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step} loss {loss.item() * ACCUM:.4f} · peak {torch.cuda.max_memory_reserved() / 2**20:.0f} MiB · "
                  f"GPU free {free_mib()} MiB", flush=True)
            if free_mib() < MIN_FREE:                                  # 给 256 训练让路：保存后退出，之后可续训
                cn.save_pretrained(CKPT); (CKPT / "last_step.txt").write_text(str(step))
                print(f"GPU free below {MIN_FREE} MiB: saved at step {step} and stopped to protect the 256 training", flush=True)
                return
        if step % 500 == 0:
            cn.save_pretrained(CKPT); (CKPT / "last_step.txt").write_text(str(step))
    cn.save_pretrained(CKPT); (CKPT / "last_step.txt").write_text(str(STEPS))
    print("saved", CKPT, flush=True)


def gen():
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
    cn = ControlNetModel.from_pretrained(CKPT, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(BASE, controlnet=cn, torch_dtype=torch.float16,
                                                             safety_checker=None).to(DEV)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()                                   # 生成时降低峰值显存
    (OUT / "images").mkdir(parents=True, exist_ok=True); (OUT / "labels").mkdir(parents=True, exist_ok=True)
    jobs = list(csv.DictReader(open(JOBS / "jobs.csv")))[:int(os.environ.get("CN_N", 300))]   # 默认 1×：前 300 个任务（每类 50）；CN_N=900 生成 2×、3× 所需的 j0300–j0899
    for i, j in enumerate(jobs):
        jid = j["jid"]
        if (OUT / "labels" / f"{jid}.txt").exists():
            continue
        bg = Image.open(JOBS / f"{jid}.png").convert("RGB")
        cx, cy = [float(v) * SIZE for v in (JOBS / f"{jid}.txt").read_text().split()[1:3]]
        lx, ly = int(max(0, min(cx - PATCH // 2, SIZE - PATCH))), int(max(0, min(cy - PATCH // 2, SIZE - PATCH)))
        w, h = float(j["bx1"]) - float(j["bx0"]), float(j["by1"]) - float(j["by0"])
        box = (lx + PATCH / 2 - w / 2, ly + PATCH / 2 - h / 2, lx + PATCH / 2 + w / 2, ly + PATCH / 2 + h / 2)
        g = torch.Generator(device=DEV).manual_seed(int(jid[1:]))
        out = pipe(prompt(j["class"]), image=control_map(bg, [box]), num_inference_steps=30, guidance_scale=7.5,
                   generator=g, height=GEN_SIZE, width=GEN_SIZE).images[0].resize((SIZE, SIZE), Image.LANCZOS)
        out.save(OUT / "images" / f"{jid}.png")
        if free_mib() < MIN_FREE:                                      # 给 256 训练让路；已生成的图保留，重跑会接着做
            print(f"GPU free below {MIN_FREE} MiB: stopped after {jid} to protect the 256 training", flush=True)
            c = CLASSES.index(j["class"])
            (OUT / "labels" / f"{jid}.txt").write_text(
                f"{c} {(lx + PATCH / 2) / SIZE:.6f} {(ly + PATCH / 2) / SIZE:.6f} {w / SIZE:.6f} {h / SIZE:.6f}\n")
            return
        c = CLASSES.index(j["class"])
        (OUT / "labels" / f"{jid}.txt").write_text(
            f"{c} {(lx + PATCH / 2) / SIZE:.6f} {(ly + PATCH / 2) / SIZE:.6f} {w / SIZE:.6f} {h / SIZE:.6f}\n")
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    {"train": train, "gen": gen}[sys.argv[1]]()
