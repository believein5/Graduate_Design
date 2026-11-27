"""DefFiller 基线（CRG 论文 "DefFiller (Mask Cond.)"）：GLIGEN-sem 上以缺陷掩码 + 类别文本为条件、从零生成整幅图

python deffiller_baseline.py convert 官方权重转存为只含张量的文件（一次）
python deffiller_baseline.py train   在划分 A 的 300 张真实训练缺陷裁剪上训练（官方默认：30k iter，batch 2，lr 5e-5）
python deffiller_baseline.py gen     对 jobs_A 的 300 个任务生成图像，写到 data/synthetic/deffiller_A/

与官方仓库（third_party/DefFiller，不修改）的差别，均为适配本数据集所必需：
- 数据：PKU 只有检测框，没有像素级掩码 → 掩码 = 框内填充（值 = 类别号 1..6）；caption = 类别名
- 路径：SD 1.4 权重路径在 trainer.py 中写死，这里替换为本地下载的 sd-v1-4.ckpt；CLIP 文本编码器用 HF 名称
- 显存：关闭冻结参数的梯度（优化不变）
- 采样：全程 alpha = 1（官方 [0.7, 0, 0.3] 在 256×256 上产生非电路板图像，见 gen() 注释），guidance 3
其余（网络、可训练参数、优化器、学习率预热、PLMS 50 步、guidance 3、alpha [0.7, 0, 0.3]）与官方一致。
标签：与其他方法相同（放置框尺寸，以 64 窗口中心为中心）。环境：conda env "baselines"。
"""
import csv
import os
import random
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "third_party" / "DefFiller"
DET = ROOT / "data" / "detection_A"
SYN = ROOT / "data" / "synthetic"
JOBS = SYN / "jobs_A"
OUT = SYN / "deffiller_A"
RUN = ROOT / "runs" / "baselines" / "deffiller"
# 官方权重转存为只含张量的文件（convert），训练代码用安全模式 torch.load 读取
SD14_SRC = Path.home() / ".cache/huggingface/hub/models--CompVis--stable-diffusion-v-1-4-original/snapshots/f0bb45b49990512c454cf2c5670b0952ef2f9c71/sd-v1-4.ckpt"
GLIGEN_SRC = ROOT / "checkpoint" / "baselines" / "gligen-generation-sem.bin"
SD14 = ROOT / "checkpoint" / "baselines" / "sd-v1-4_tensors.pt"
GLIGEN_SEM = ROOT / "checkpoint" / "baselines" / "gligen-sem_tensors.pt"
# CLIP ViT-L/14 文本编码器 + tokenizer（链接到本地 SD 1.5 的同一组文件；权重随后被 SD 1.4 的 cond_stage_model 覆盖）
CLIP_TEXT = ROOT / "checkpoint" / "baselines" / "clip-vit-large-patch14-text"
NAMES = {"missinghole": "missing hole", "mousebite": "mouse bite", "opencircuit": "open circuit",
         "short": "short circuit", "spur": "spur", "spuriouscopper": "spurious copper"}
CLASSES = list(NAMES)
SIZE, PATCH, N_SEM = 256, 64, 152
DEV = "cuda"

sys.path.insert(0, str(REPO))


def sem_onehot(boxes):
    """boxes: [(cls, x0, y0, x1, y1)] 像素坐标 → 152 通道 one-hot（0 = 背景，1..6 = 类别）"""
    sem = torch.zeros(SIZE, SIZE, dtype=torch.long)
    for c, x0, y0, x1, y1 in boxes:
        sem[max(0, int(round(y0))):int(round(y1)), max(0, int(round(x0))):int(round(x1))] = c + 1
    return torch.zeros(N_SEM, SIZE, SIZE).scatter_(0, sem[None], 1.0)


def caption(classes):
    return " and ".join(NAMES[CLASSES[c]] for c in dict.fromkeys(classes))


class PCBSemanticDataset:
    """划分 A 的训练缺陷裁剪（跳过 50 张无缺陷图），输出格式与官方 SemanticDataset 相同"""

    def __init__(self, image_size=256, random_flip=True, **kwargs):
        self.items = []
        for lp in sorted((DET / "train" / "labels").glob("*.txt")):
            lines = [l.split() for l in lp.read_text().splitlines() if l.strip()]
            if not lines:
                continue
            bx = [(int(c), (float(cx) - float(w) / 2) * SIZE, (float(cy) - float(h) / 2) * SIZE,
                   (float(cx) + float(w) / 2) * SIZE, (float(cy) + float(h) / 2) * SIZE) for c, cx, cy, w, h in lines]
            self.items.append((DET / "train" / "images" / f"{lp.stem}.png", bx))
        self.random_flip = random_flip

    def total_images(self):
        return len(self)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, bx = self.items[i]
        img = torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), np.float32) / 255).permute(2, 0, 1)
        sem = sem_onehot(bx)
        if self.random_flip and random.random() < 0.5:
            img, sem = img.flip(-1), sem.flip(-1)
        return {"id": i, "image": (img - 0.5) / 0.5, "sem": sem, "mask": torch.tensor(1.0),
                "caption": caption([b[0] for b in bx])}


def convert():
    """把官方权重（CompVis SD 1.4、gligen 官方 HF 仓库）转存为只含张量的文件；pytorch_lightning 对象用占位类跳过"""
    import pickle
    import types

    class Skip:
        def __init__(self, *a, **k): pass
        def __setstate__(self, s): pass

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            return Skip if mod.startswith(("pytorch_lightning", "lightning")) else super().find_class(mod, name)
    pm = types.ModuleType("pm"); pm.Unpickler, pm.load, pm.__name__ = U, pickle.load, "pickle"
    for src, dst, key in ((SD14_SRC, SD14, "state_dict"), (GLIGEN_SRC, GLIGEN_SEM, "model")):
        if dst.exists():
            continue
        sd = torch.load(src, map_location="cpu", weights_only=False, pickle_module=pm)
        torch.save({key: {k: v for k, v in sd[key].items() if torch.is_tensor(v)}}, dst)
        print("saved", dst, flush=True)


def train():
    from omegaconf import OmegaConf
    import trainer as T
    import dataset.concat_dataset as CD
    os.chdir(REPO)                                                       # 官方代码按仓库根目录的相对路径工作

    orig_read = T.read_official_ckpt
    def read_sd14(_path):                                                # 替换写死的 SD 1.4 路径
        sd = orig_read(str(SD14))
        sd["text_encoder"].pop("transformer.text_model.embeddings.position_ids", None)   # 新版 transformers 中为非持久 buffer
        return sd
    T.read_official_ckpt = read_sd14

    class Catalog(CD.DatasetCatalog):
        def __init__(self, root):
            super().__init__(root)
            self.PCBSemantic = {"target": "__main__.PCBSemanticDataset", "train_params": {}}
    CD.DatasetCatalog = Catalog

    config = OmegaConf.load(REPO / "configs" / "sd_saliency_900.yaml")
    config.text_encoder.params = {"version": str(CLIP_TEXT)}
    config.train_dataset_names = {"PCBSemantic": {"image_size": SIZE, "random_flip": True}}
    config.update(dict(                                                  # main.py 的默认参数
        DATA_ROOT=str(DET), OUTPUT_ROOT=str(RUN), name=os.environ.get("RUN_NAME", "pcb_A"), seed=123, local_rank=0,
        yaml_file=str(REPO / "configs" / "sd_saliency_900.yaml"), base_learning_rate=5e-5, weight_decay=0.0,
        warmup_steps=10000, scheduler_type="constant", batch_size=2, workers=2, official_ckpt_name="sd-v1-4.ckpt",
        ckpt=str(GLIGEN_SEM), inpaint_mode=False, randomize_fg_mask=False, random_add_bg_mask=False,
        enable_ema=False, ema_rate=0.9999, total_iters=int(os.environ.get("TOTAL_ITERS", 30000)),
        save_every_iters=10000, disable_inference_in_training=False, distributed=False))
    config.total_batch_size = config.batch_size
    torch.manual_seed(config.seed); random.seed(config.seed); np.random.seed(config.seed)
    if os.environ.get("MAX_MIB"):                                       # 共享 GPU：超出上限时本进程 OOM，不挤占其他任务
        torch.cuda.set_per_process_memory_fraction(int(os.environ["MAX_MIB"]) * 2**20 / torch.cuda.get_device_properties(0).total_memory)
    tr = T.Trainer(config)
    # 官方代码只优化新增的 GLIGEN 层（tr.opt 中的参数），但没有关闭其余冻结 UNet 参数的梯度；
    # 关闭后优化完全相同（这些梯度从不被使用），可省约 3.4 GB 显存
    train_ids = {id(p) for g in tr.opt.param_groups for p in g["params"]}
    for p in tr.model.parameters():
        if id(p) not in train_ids:
            p.requires_grad_(False)
    print(f"trainable {sum(p.numel() for p in tr.model.parameters() if p.requires_grad) / 1e6:.0f} M params · "
          f"peak after setup {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB", flush=True)
    tr.start_training()


def gen():
    from omegaconf import OmegaConf
    from ldm.models.diffusion.plms import PLMSSampler
    from ldm.util import instantiate_from_config
    # 采样最后 30% 步（alpha = 0）时官方代码把第一层卷积换回原始 SD 的卷积，从相对路径读取这份权重；仓库未附带，从 SD 1.4 取出
    sd_conv = RUN / "SD_input_conv_weight_bias.pth"
    if not sd_conv.exists():
        sd = torch.load(SD14, map_location="cpu", weights_only=True)["state_dict"]
        torch.save({k: sd[f"model.diffusion_model.input_blocks.0.0.{k}"] for k in ("weight", "bias")}, sd_conv)
    os.chdir(RUN)
    ck = sorted(RUN.glob("pcb_A/tag*/checkpoint_latest.pth"))[-1]
    saved = torch.load(ck, map_location="cpu", weights_only=False)     # 本项目自己训练保存的文件
    config = saved["config_dict"]["_content"]
    model = instantiate_from_config(config["model"]).to(DEV).eval()
    ae = instantiate_from_config(config["autoencoder"]).to(DEV).eval()
    te = instantiate_from_config(config["text_encoder"]).to(DEV).eval()
    diffusion = instantiate_from_config(config["diffusion"]).to(DEV)
    for net, k in ((model, "model"), (ae, "autoencoder"), (te, "text_encoder"), (diffusion, "diffusion")):
        net.load_state_dict(saved[k])
    tok_in = instantiate_from_config(config["grounding_tokenizer_input"])
    ds_in = instantiate_from_config(config["grounding_downsampler_input"])
    model.grounding_tokenizer_input = tok_in
    gligen_conv = model.input_blocks[0][0]                            # 官方 infer.py 每张图重新加载模型；这里每张图前恢复 GLIGEN 第一层
    sys.path.insert(0, str(REPO))
    from infer import alpha_generator, set_alpha_scale
    # 采样设置：官方 infer.py 为 alpha [0.7, 0, 0.3]（最后 30% 步关闭 GLIGEN 层、交给原始 SD）、guidance 3；
    # 训练中的预览图为全程 alpha = 1、guidance 5。DF_ALPHA=none / DF_GUIDANCE / DF_OUT / DF_N 用于比较
    # 默认全程 alpha = 1：在 256×256 上官方 [0.7, 0, 0.3] 的最后 30% 步由原始 SD 1.4 完成，生成的不是电路板图像
    # （results/experiments/deffiller_check/settings.png）；guidance 3 在 README 建议的 2–4 范围内
    alpha = os.environ.get("DF_ALPHA", "none")
    guidance = float(os.environ.get("DF_GUIDANCE", 3))
    out = SYN / os.environ["DF_OUT"] if os.environ.get("DF_OUT") else OUT
    sampler = PLMSSampler(diffusion, model) if alpha == "none" else PLMSSampler(
        diffusion, model, alpha_generator_func=partial(alpha_generator, type=[float(a) for a in alpha.split(",")]),
        set_alpha_scale=set_alpha_scale)
    (out / "images").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(parents=True, exist_ok=True)
    print("checkpoint", ck, flush=True)
    for i, j in enumerate(list(csv.DictReader(open(JOBS / "jobs.csv")))[:int(os.environ.get("DF_N", 300))]):
        jid = j["jid"]
        if (out / "labels" / f"{jid}.txt").exists():
            continue
        c = CLASSES.index(j["class"])
        cx, cy = [float(v) * SIZE for v in (JOBS / f"{jid}.txt").read_text().split()[1:3]]
        lx, ly = int(max(0, min(cx - PATCH // 2, SIZE - PATCH))), int(max(0, min(cy - PATCH // 2, SIZE - PATCH)))
        w, h = float(j["bx1"]) - float(j["bx0"]), float(j["by1"]) - float(j["by0"])
        box = (lx + PATCH / 2 - w / 2, ly + PATCH / 2 - h / 2, lx + PATCH / 2 + w / 2, ly + PATCH / 2 + h / 2)
        batch = {"sem": sem_onehot([(c, *box)])[None].to(DEV), "mask": torch.ones(1, 1, device=DEV)}
        model.input_blocks[0][0], model.first_conv_type = gligen_conv, "GLIGEN"
        torch.manual_seed(int(jid[1:]))
        with torch.no_grad():
            inp = dict(x=None, timesteps=None, context=te.encode([NAMES[j["class"]]]),
                       grounding_input=tok_in.prepare(batch), inpainting_extra_input=None,
                       grounding_extra_input=ds_in.prepare(batch))
            z = sampler.sample(S=50, shape=(1, model.in_channels, model.image_size, model.image_size), input=inp,
                               uc=te.encode([""]), guidance_scale=guidance)
            x = ae.decode(z).clamp(-1, 1)[0] * 0.5 + 0.5
        Image.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(out / "images" / f"{jid}.png")
        (out / "labels" / f"{jid}.txt").write_text(
            f"{c} {(lx + PATCH / 2) / SIZE:.6f} {(ly + PATCH / 2) / SIZE:.6f} {w / SIZE:.6f} {h / SIZE:.6f}\n")
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/300", flush=True)


if __name__ == "__main__":
    {"convert": convert, "train": train, "gen": gen}[sys.argv[1]]()
