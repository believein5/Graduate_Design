"""AnomalyDiffusion 基线（Hu et al., AAAI 2024）：LDM text2img-large 上学习异常嵌入 + 空间嵌入，按给定掩码在无缺陷背景中修复出缺陷

python anomalydiffusion_baseline.py prepare          由划分 A 构建 MVTec 格式数据集（runs/baselines/anomalydiffusion/mvtec_pcb/pcb）
python anomalydiffusion_baseline.py train [类别...]  每个缺陷类别训练一个异常生成模型（5000 步，batch 4）；已完成的类别跳过，中断后从最近的 1000 步续训
python anomalydiffusion_baseline.py gen [N]          对 jobs_A 的 j0000..j(N-1)（默认 300）生成图像；已有标签的任务跳过

环境变量：AD_CKPT（checkpoint 目录，默认 runs/baselines/anomalydiffusion/ckpt）、AD_OUT（生成输出目录，默认 data/synthetic/anomalydiffusion_A）、
AD_MAX_STEPS（默认 5000）、AD_MICRO_BATCH（默认 2）、MAX_MIB（本进程显存上限，超出时本进程 OOM）、AD_GPU_ADAM=1（Adam 状态放 GPU）、
AD_TEXT_FP32=1（BERT 用 fp32 代替 bf16）。训练日志：前 10 步每步、之后每 50 步打印 loss、GradScaler scale、是否跳过、
空间编码器输出 |max|；前向损失非有限时报出第一个输出 NaN/inf 的模块后退出，GradScaler 连续跳过 20 步时报出第一个非有限梯度后退出。
未设置时脚本把 PYTORCH_CUDA_ALLOC_CONF 设为 expandable_segments:True。环境：conda env "baselines"（未新增任何包）。

与官方实现（third_party/anomalydiffusion，不修改）一致的部分：模型与配置 txt2img-1p4B-finetune-encoder+embedding.yaml
（EmbeddingManager2：每个 sample+anomaly 4 个可学习词向量 + 空间编码器 GradualStyleEncoder 由掩码产生 4 个词向量）、
LDM text2img-large 权重冻结、初始化词 "anomaly"、--spatial_encoder_embedding --data_enhance 的数据增强、Adam、
学习率 = base_lr 5e-3 × batch 4（main.py 的 scale_lr）、掩码区域内的噪声损失、生成时 DDIM 200 步 eta 1、潜空间掩码修复
（generate_with_mask.py → log_images(inpaint=True)，不用 adaptive attention reweighting：README 只对纹理类异常开启，PCB 缺陷是结构类）。
输出直接保存解码结果（官方同样不把背景贴回）。

与官方的差别（均为适配本数据集或本机环境所必需）：
- 不训练、不使用官方的掩码生成模型（train_mask.py / generate_mask.py）：生成掩码由本实验给定，
  = 放置框（尺寸 bx1-bx0 × by1-by0，以 64 窗口中心为中心）的填充矩形，与其他方法的标签框完全相同
- 每个类别单独训练一个模型（6 个类别各 5000 步、batch 4）；官方在全部 MVTec 类别上联合训练一个模型（配置中 max_steps 1e6）
- 训练用该类别全部 50 张裁剪（官方只用每类异常图像的前 1/3，其余做测试）；掩码 = 该裁剪中本类别检测框的并集（PKU 只有框）
- 不使用 pytorch_lightning（baselines 环境未安装；官方要求 1.5.0 + torch 1.12）：用等价的手写训练循环
  （Adam、恒定学习率、每 1000 步保存 embeddings.pt / spatial_encoder.pt，格式与官方相同）；DataLoader num_workers=0
- 显存（与其他实验共用一块 4090）：混合精度。冻结的 UNet / VAE 以 fp16 存放、在 fp16 autocast 下运行，GradScaler 缩放损失；
  可训练的嵌入与空间编码器的参数为 fp32；空间编码器全程 fp32 计算（关闭 autocast，与官方相同）；
  冻结的 BERT 文本编码器以 bf16 存放并在 bf16 autocast 下计算（AD_TEXT_FP32=1 改为 fp32，显存多约 1.2 GB）；
  UNet 输出转 fp32 后再算损失与 DDIM 更新。原因：按官方配方（Adam，lr 2e-2）训练几十步后空间编码器输出的量级
  远超 fp16 上限（冒烟测试 60 步后 |max| ≈ 7.5e8），全 fp16 时 BERT 前向得到 inf/NaN，GradScaler 跳过此后每一步；
  bf16 与 fp32 数值范围相同，BERT 末层 LayerNorm 把条件向量拉回正常量级，之后的 UNet 在 fp16 下安全；
  冻结网络用 torch.utils.checkpoint 重计算激活（官方关闭了 ldm 的 checkpoint，只影响显存不影响数值）；
  空间编码器有 356 M 可训练参数，Adam 状态默认放在 CPU（CPUAdam，算法不变；AD_GPU_ADAM=1 放回 GPU，更快但多约 2.8 GB）；
  batch 4 拆成 2 个微批（AD_MICRO_BATCH=2）做梯度累积：每微批损失除以累积数，GradScaler 只在累积边界 step / update，
  有效 batch、学习率（5e-3 × 4）与优化步数（5000）不变；batch 4 一次前向在 8 GB 上限内 OOM（UNet cross-attention）。
  累积带来的唯一差别：data_enhance 的随机增强（概率 0.5，与随机裁剪一样整批共用一次）改为每个微批各取一次
- 新版 torchvision 的张量 Resize 默认抗锯齿，这里改回官方环境（torchvision 0.13）的不抗锯齿行为，保证掩码下采样到 32×32 的结果一致
"""
import csv
import gc
import os
import random
import shutil
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # 须在 torch 初始化 CUDA 之前设置，减少显存碎片

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "third_party" / "anomalydiffusion"
DET = ROOT / "data" / "detection_A"
JOBS = ROOT / "data" / "synthetic" / "jobs_A"
RUN = ROOT / "runs" / "baselines" / "anomalydiffusion"
MVTEC = RUN / "mvtec_pcb"                                             # MVTec 根目录，唯一的 sample 名为 "pcb"
CKPT = Path(os.environ.get("AD_CKPT", RUN / "ckpt"))
OUT = Path(os.environ.get("AD_OUT", ROOT / "data" / "synthetic" / "anomalydiffusion_A"))
LDM_CKPT = ROOT / "checkpoint" / "baselines" / "ldm-txt2img-f8-large.ckpt"
CONFIG = REPO / "configs" / "latent-diffusion" / "txt2img-1p4B-finetune-encoder+embedding.yaml"
CLASSES = ["missinghole", "mousebite", "opencircuit", "short", "spur", "spuriouscopper"]   # 与 common.py 相同的 id 顺序
SIZE, PATCH = 256, 64
MAX_STEPS = int(os.environ.get("AD_MAX_STEPS", 5000))
BATCH, BASE_LR, SAVE_EVERY, SEED = 4, 5e-3, 1000, 23                 # 配置 batch_size / base_learning_rate；main.py 默认 seed 23
MICRO_BATCH = int(os.environ.get("AD_MICRO_BATCH", 2))               # 每次前向的样本数；BATCH / MICRO_BATCH 个微批累积成一个优化步
MAX_SKIPS = 20                                                        # GradScaler 连续跳过这么多优化步时报错退出（正常只在开头跳过几步）
DDIM_STEPS, DDIM_ETA = 200, 1.0                                       # log_images 的默认值（generate_with_mask.py 未改）
DEV = "cuda"


def name_of(cls):
    return f"pcb+{cls}"


# ---------------------------------------------------------------- 加载官方代码所需的兼容层
def setup_repo():
    """让官方 ldm 代码在 baselines 环境（torch 2.14、无 pytorch_lightning）中导入；只替换与数值无关的部分"""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")                     # bert-base-uncased tokenizer 已在本地缓存
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(REPO))

    # pytorch_lightning 占位：官方模型只把 LightningModule 当 nn.Module 用（外加 self.device）
    class LightningModule(torch.nn.Module):
        @property
        def device(self):
            for t in self.parameters():
                return t.device
            return torch.device("cpu")
    pl = types.ModuleType("pytorch_lightning")
    pl.LightningModule = LightningModule
    pl_util = types.ModuleType("pytorch_lightning.utilities")
    pl_dist = types.ModuleType("pytorch_lightning.utilities.distributed")
    pl_dist.rank_zero_only = lambda f: f
    pl.utilities, pl_util.distributed = pl_util, pl_dist
    sys.modules.update({"pytorch_lightning": pl, "pytorch_lightning.utilities": pl_util,
                        "pytorch_lightning.utilities.distributed": pl_dist})

    # stylegan2 的 CUDA 扩展在导入时 JIT 编译；空间编码器只用到不带激活的 EqualLinear（F.linear），不调用这些算子
    op = types.ModuleType("ldm.models.psp_encoder.stylegan2.op")
    def _unused(*a, **k):
        raise NotImplementedError("stylegan2 CUDA op 未编译（AnomalyDiffusion 的空间编码器不使用）")
    op.FusedLeakyReLU = op.fused_leaky_relu = op.upfirdn2d = _unused
    sys.modules["ldm.models.psp_encoder.stylegan2.op"] = op
    # taming-transformers 未安装：VectorQuantizer2 只被 VQModel 使用，本配置的第一阶段是 AutoencoderKL
    for m in ("taming", "taming.modules", "taming.modules.vqvae", "taming.modules.vqvae.quantize"):
        sys.modules.setdefault(m, types.ModuleType(m))
    sys.modules["taming.modules.vqvae.quantize"].VectorQuantizer2 = _unused
    # imgaug 未安装：ldm.data.mask_generate.generate_mask 只在 --random_mask 时使用
    mg = types.ModuleType("ldm.data.mask_generate")
    mg.generate_mask = _unused
    sys.modules["ldm.data.mask_generate"] = mg

    # 张量 Resize 恢复 torchvision 0.13 的默认（不抗锯齿）
    import torchvision.transforms as T
    orig_init = T.Resize.__init__
    def resize_init(self, size, interpolation=T.InterpolationMode.BILINEAR, max_size=None, antialias=False):
        orig_init(self, size, interpolation, max_size, antialias)
    T.Resize.__init__ = resize_init

    # 冻结网络的激活重计算（torch.utils.checkpoint，非重入版本允许参数 requires_grad=False）
    import ldm.modules.attention as A
    import ldm.modules.diffusionmodules.openaimodel as O
    from torch.utils.checkpoint import checkpoint as tc
    def checkpoint(func, inputs, params, flag):
        if flag and torch.is_grad_enabled():
            return tc(func, *inputs, use_reentrant=False)
        return func(*inputs)
    A.checkpoint = O.checkpoint = checkpoint


def build_model(names, workdir, train):
    """与 main.py / generate_with_mask.py 相同的构建过程：配置 → 载入 LDM 权重 → prepare_spatial_encoder"""
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "name-anomaly.txt").write_text("\n".join(names))      # EmbeddingManager 从当前目录读取 sample+anomaly 列表
    os.chdir(workdir)
    cfg = OmegaConf.load(CONFIG)
    pc = cfg.model.params.personalization_config.params
    pc.initializer_words[0] = "anomaly"                              # README：--init_word anomaly
    pc.embedding_manager_ckpt = ""
    model = instantiate_from_config(cfg.model)
    sd = torch.load(LDM_CKPT, map_location="cpu", mmap=True, weights_only=True)["state_dict"]
    model.load_state_dict(sd, strict=False)
    del sd
    model.prepare_spatial_encoder(optimze_together=True, data_enhance=train)
    # 精度：UNet / VAE 以 fp16 存放、在外层 fp16 autocast 下运行；
    # 空间编码器的输出在训练几十步后即远超 fp16 上限 65504（冒烟测试 60 步后 |max| ≈ 7.5e8，fp32 下有限），
    # 因此空间编码器全程 fp32（关闭 autocast，与官方相同），BERT 文本编码器以 bf16 存放并在 bf16 autocast 下运行
    # （bf16 的数值范围与 fp32 相同；AD_TEXT_FP32=1 时 BERT 用 fp32，显存多约 1.2 GB）。
    # BERT 最后的 LayerNorm 把条件向量拉回正常量级，之后的 UNet cross-attention 在 fp16 下安全
    text_fp32 = os.environ.get("AD_TEXT_FP32") == "1"
    model.model.half(); model.first_stage_model.half()
    model.cond_stage_model.to(torch.float32 if text_fp32 else torch.bfloat16)
    model.to(DEV)
    se = model.embedding_manager.spatial_encoder_model
    se_forward = se.forward
    def se_fp32(x):
        with torch.autocast("cuda", enabled=False):
            y = se_forward(x.float())
        se.last_absmax = y.detach().abs().amax()                     # 诊断：最近一次输出的最大绝对值（不同步，记录时才读取）
        return y
    se.forward = se_fp32
    get_cond = model.get_learned_conditioning
    def cond_text(c, x=None, name=None):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not text_fp32):
            return get_cond(c, x=x, name=name)
    model.get_learned_conditioning = cond_text
    apply_model = model.apply_model
    model.apply_model = lambda *a, **k: apply_model(*a, **k).float()   # UNet 输出转 fp32 后再算损失 / DDIM 更新
    model.learning_rate = BASE_LR * BATCH
    return model


def set_mem_cap():
    if os.environ.get("MAX_MIB"):
        torch.cuda.set_per_process_memory_fraction(
            int(os.environ["MAX_MIB"]) * 2**20 / torch.cuda.get_device_properties(0).total_memory)


def set_seed(seed):
    """generate_with_mask.py 的 setup_seed"""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def box_mask(boxes):
    """[(x0, y0, x1, y1)] 像素坐标 → 256×256 的 0/1 掩码（矩形填充，取整方式与 deffiller_baseline 相同）"""
    m = np.zeros((SIZE, SIZE), np.uint8)
    for x0, y0, x1, y1 in boxes:
        m[max(0, int(round(y0))):int(round(y1)), max(0, int(round(x0))):int(round(x1))] = 1
    return m


# ---------------------------------------------------------------- prepare
def prepare():
    pcb = MVTEC / "pcb"
    for cls in CLASSES:
        k = CLASSES.index(cls)
        (pcb / "test" / cls).mkdir(parents=True, exist_ok=True)
        (pcb / "ground_truth" / cls).mkdir(parents=True, exist_ok=True)
        imgs = sorted((DET / "train" / "images").glob(f"train_{cls}_*.png"))
        assert len(imgs) == 50, (cls, len(imgs))
        for i, ip in enumerate(imgs):
            bx = []
            for line in (DET / "train" / "labels" / f"{ip.stem}.txt").read_text().split("\n"):
                if line.strip() and int(line.split()[0]) == k:      # 只取本类别的框（部分裁剪含其他类别的框）
                    cx, cy, w, h = (float(v) * SIZE for v in line.split()[1:])
                    bx.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
            assert bx, ip
            shutil.copyfile(ip, pcb / "test" / cls / f"{i:03d}.png")
            Image.fromarray(box_mask(bx) * 255).save(pcb / "ground_truth" / cls / f"{i:03d}_mask.png")
    (pcb / "train" / "good").mkdir(parents=True, exist_ok=True)       # 正常图像 = 任务背景（划分 A 训练照片的无缺陷裁剪）
    for bg in sorted(JOBS.glob("j[0-9]*.png")):
        shutil.copyfile(bg, pcb / "train" / "good" / f"{int(bg.stem[1:]):03d}.png")
    print("prepared", pcb, flush=True)


# ---------------------------------------------------------------- train
class ClassDataset(torch.utils.data.Dataset):
    """与 Personalized_mvtec_encoder（data_enhance=True → 读成 512×512）相同的样本格式，但只含一个类别、用全部图像"""

    def __init__(self, cls, repeats=20):
        from ldm.data.personalized import imagenet_templates_small
        self.templates, self.name, self.repeats = imagenet_templates_small, name_of(cls), repeats
        d = MVTEC / "pcb"
        self.data = []
        for ip in sorted((d / "test" / cls).glob("*.png")):
            img = Image.open(ip).convert("RGB").resize((512, 512), resample=Image.BICUBIC)
            mask = Image.fromarray(np.asarray(Image.open(d / "ground_truth" / cls / f"{ip.stem}_mask.png").convert("L"),
                                              np.float32)).resize((512, 512), resample=Image.BICUBIC)
            mask = np.asarray(mask, np.float32) / 255.0
            self.data.append(((np.asarray(img, np.float32) / 127.5 - 1.0).astype(np.float32),
                              (mask >= 0.5).astype(np.float32)))

    def __len__(self):
        return len(self.data) * self.repeats

    def __getitem__(self, i):
        img, mask = self.data[i % len(self.data)]
        return {"caption": random.choice(self.templates).format("*"), "image": img, "mask": mask, "name": self.name}


class CPUAdam(torch.optim.Optimizer):
    """参数留在 GPU，fp32 主副本与 Adam 状态放在 CPU（fused CPU Adam）：算法与 GPU 上的 Adam 相同，省约 2.8 GB 显存，
    每步多约 0.4 s。GradScaler 照常在 GPU 上反缩放 / 检查 inf，再调用这里的 step"""

    def __init__(self, params, lr):
        params = list(params)
        super().__init__(params, dict(lr=lr))
        self.cpu = [p.detach().to("cpu", copy=True).pin_memory() for p in params]
        for c in self.cpu:
            c.grad = torch.zeros_like(c).pin_memory()
        self.inner = torch.optim.Adam(self.cpu, lr=lr, fused=True)

    @torch.no_grad()
    def step(self, closure=None):
        gpu = self.param_groups[0]["params"]
        for p, c in zip(gpu, self.cpu):
            c.grad.copy_(p.grad, non_blocking=True)
        torch.cuda.synchronize()
        self.inner.step()
        for p, c in zip(gpu, self.cpu):
            p.copy_(c, non_blocking=True)

    def state_dict(self):
        return self.inner.state_dict()

    def load_state_dict(self, sd):
        self.inner.load_state_dict(sd)


def make_opt(params, lr):
    """configure_optimizers（spatial_encoder_and_embedding）：一个 Adam 同时优化嵌入与空间编码器；AD_GPU_ADAM=1 时状态放 GPU"""
    if os.environ.get("AD_GPU_ADAM") == "1":
        return torch.optim.Adam(params, lr=lr)
    return CPUAdam(params, lr)


def save_ckpt(model, d, step, opt=None, scaler=None):
    em = model.embedding_manager
    torch.save(em.spatial_encoder_model.state_dict(), d / "spatial_encoder.pt")   # 与官方 on_save_checkpoint 相同的两个文件
    em.save(d / "embeddings.pt")
    if opt is not None:
        torch.save({"step": step, "opt": opt.state_dict(), "scaler": scaler.state_dict()}, d / "resume.pt.tmp")
        os.replace(d / "resume.pt.tmp", d / "resume.pt")


def load_ckpt(model, d):
    em = model.embedding_manager
    em.spatial_encoder_model.load_state_dict(torch.load(d / "spatial_encoder.pt", map_location=DEV, weights_only=True))
    ck = torch.load(d / "embeddings.pt", map_location="cpu", weights_only=False)   # 本项目自己保存的文件（含 ParameterDict）
    em.string_to_token_dict = ck["string_to_token"]                                # 与 EmbeddingManager.load 相同
    em.string_to_param_dict = ck["string_to_param"].to(DEV)


def first_nonfinite_module(model, batch):
    """诊断：在同一批数据上重跑一次前向（no_grad），按完成顺序返回第一个输出含 NaN/inf 的模块"""
    found = []
    def hook(name):
        def f(m, i, o):
            for t in (o if isinstance(o, (tuple, list)) else (o,)):
                if not found and torch.is_tensor(t) and t.is_floating_point() and not torch.isfinite(t).all():
                    found.append(f"{name} ({type(m).__name__}, {t.dtype})")
        return f
    hs = [m.register_forward_hook(hook(n)) for n, m in model.named_modules() if n]
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            l, _ = model.shared_step(batch)
    finally:
        for h in hs:
            h.remove()
    return found[0] if found else f"none in rerun (rerun loss {float(l)})"


def first_nonfinite_grad(named):
    for n, p in named:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return n
    return "none"


def train_class(cls):
    d = CKPT / cls
    set_seed(SEED)
    model = build_model([name_of(cls)], d, train=True)
    model.train()                                                   # 冻结网络的 train 已被官方替换为空操作，仍为 eval
    assert BATCH % MICRO_BATCH == 0, (BATCH, MICRO_BATCH)
    accum = BATCH // MICRO_BATCH                                    # 每个优化步累积的微批数
    loader = torch.utils.data.DataLoader(ClassDataset(cls), batch_size=MICRO_BATCH, shuffle=True, num_workers=0)
    em = model.embedding_manager
    scaler = torch.amp.GradScaler("cuda")
    step = 0
    if (d / "resume.pt").exists():
        load_ckpt(model, d)
    params = list(em.embedding_parameters()) + list(em.spatial_encoder_model.parameters())
    opt = make_opt(params, model.learning_rate)
    if (d / "resume.pt").exists():
        r = torch.load(d / "resume.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(r["opt"]); scaler.load_state_dict(r["scaler"]); step = r["step"]
        set_seed(SEED + step)
        print(f"{cls}: resume from step {step}", flush=True)
    print(f"{cls}: {len(loader.dataset.data)} images, batch {BATCH} = {MICRO_BATCH} × {accum} accum, lr {model.learning_rate:.2e}, "
          f"trainable {sum(p.numel() for p in params) / 1e6:.1f} M, setup peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB",
          flush=True)
    se = em.spatial_encoder_model
    named = [(f"embedding[{k}]", p) for k, p in em.string_to_param_dict.items()] + \
            [(f"spatial_encoder.{n}", p) for n, p in se.named_parameters()]
    it, t0, t_log, log_step, skips = iter(loader), time.time(), time.time(), step, 0
    while step < MAX_STEPS:                                         # step = 优化步数（不是微批数），resume.pt 中保存的也是它
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for _ in range(accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader); batch = next(it)
            with torch.autocast("cuda", dtype=torch.float16):
                l, _ = model.shared_step(batch)
            if not torch.isfinite(l):                               # 前向非有限：重跑一次前向，报出第一个输出 NaN/inf 的模块
                raise RuntimeError(f"{cls} step {step + 1}: non-finite loss {float(l)}; first non-finite forward output: "
                                   f"{first_nonfinite_module(model, batch)}; spatial-encoder |max| {float(se.last_absmax):.3g}")
            scaler.scale(l / accum).backward()                      # 各微批损失为批内均值，除以累积数后梯度 = batch 4 的均值梯度
            loss += l.detach() / accum
        scale = scaler.get_scale()
        scaler.step(opt)                                            # 只在累积边界上反缩放 / 检查 inf / 更新
        scaler.update()
        step += 1
        skipped = scaler.get_scale() < scale                        # GradScaler 发现 inf/NaN 梯度时跳过本步并减小 scale
        bad = first_nonfinite_grad(named) if skipped else ""
        skips = skips + 1 if skipped else 0
        if skips >= MAX_SKIPS:
            raise RuntimeError(f"{cls} step {step}: {skips} consecutive optimizer steps skipped by GradScaler "
                               f"(scale {scaler.get_scale():.3g}); first non-finite grad: {bad}")
        if step <= 10 or step % 50 == 0 or step == MAX_STEPS:
            torch.cuda.synchronize()
            print(f"{cls} step {step}/{MAX_STEPS} loss {float(loss):.4f} scale {scale:.3g}"
                  f"{f' SKIPPED (non-finite grad: {bad})' if skipped else ''} se|max| {float(se.last_absmax):.3g} "
                  f"{(time.time() - t_log) / (step - log_step):.3f} s/step peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB "
                  f"(reserved {torch.cuda.max_memory_reserved() / 2**20:.0f})", flush=True)
            t_log, log_step = time.time(), step
        if step % SAVE_EVERY == 0 and step < MAX_STEPS:
            save_ckpt(model, d, step, opt, scaler)
    save_ckpt(model, d, step)
    (d / "resume.pt").unlink(missing_ok=True)
    (d / "done").write_text(f"{step} steps, {time.time() - t0:.0f} s\n")
    print(f"{cls}: done in {time.time() - t0:.0f} s", flush=True)
    del model, opt, loader, params
    gc.collect(); torch.cuda.empty_cache()


def train(classes):
    setup_repo()
    set_mem_cap()
    torch.backends.cudnn.benchmark = True                           # 配置 trainer.benchmark: True
    for cls in classes or CLASSES:
        if (CKPT / cls / "done").exists():
            print(f"{cls}: already trained, skip", flush=True)
            continue
        train_class(cls)


# ---------------------------------------------------------------- gen
def gen(n):
    jobs = [j for j in list(csv.DictReader(open(JOBS / "jobs.csv")))[:n]
            if not (OUT / "labels" / f"{j['jid']}.txt").exists()]
    if not jobs:
        print("nothing to do", flush=True)
        return
    missing = sorted({j["class"] for j in jobs if not (CKPT / j["class"] / "done").exists()})
    assert not missing, f"untrained classes: {missing}"
    os.environ.setdefault("TQDM_DISABLE", "1")
    setup_repo()
    set_mem_cap()
    from torchvision import transforms
    from ldm.data.personalized import imagenet_templates_small
    model = build_model([name_of(c) for c in CLASSES], CKPT / "gen_work", train=False)
    model.eval()
    resize32 = transforms.Resize((32, 32))
    (OUT / "images").mkdir(parents=True, exist_ok=True); (OUT / "labels").mkdir(parents=True, exist_ok=True)
    jobs.sort(key=lambda j: (CLASSES.index(j["class"]), j["jid"]))   # 按类别分组，每个类别只载入一次权重；每个任务单独设种子，顺序不影响结果
    loaded, t0 = None, time.time()
    for i, j in enumerate(jobs):
        jid, cls = j["jid"], j["class"]
        if cls != loaded:
            load_ckpt(model, CKPT / cls); loaded = cls
        c = CLASSES.index(cls)
        cx, cy = [float(v) * SIZE for v in (JOBS / f"{jid}.txt").read_text().split()[1:3]]
        lx, ly = int(max(0, min(cx - PATCH // 2, SIZE - PATCH))), int(max(0, min(cy - PATCH // 2, SIZE - PATCH)))  # common.window_of
        w, h = float(j["bx1"]) - float(j["bx0"]), float(j["by1"]) - float(j["by0"])
        ox, oy = lx + PATCH / 2, ly + PATCH / 2
        mask = torch.from_numpy(box_mask([(ox - w / 2, oy - h / 2, ox + w / 2, oy + h / 2)]).astype(np.float32))[None, None].to(DEV)
        img = torch.from_numpy(np.asarray(Image.open(JOBS / f"{jid}.png").convert("RGB"), np.float32) / 127.5 - 1.0)
        img = img.permute(2, 0, 1)[None].to(DEV)
        set_seed(int(jid[1:]))
        # 与 log_images(inpaint=True) 相同的步骤：编码背景 → 文本 + 掩码空间嵌入 → 32×32 掩码上的 DDIM 修复 → 解码
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            z = model.get_first_stage_encoding(model.encode_first_stage(img)).detach()
            caption = random.choice(imagenet_templates_small).format("*")
            cond, position = model.get_learned_conditioning([caption], x=mask.clone(), name=[name_of(cls)])
            m32 = resize32(mask)
            m32 = (m32 >= 0.5).float()
            samples, _ = model.sample_log(cond=cond, batch_size=1, ddim=True, ddim_steps=DDIM_STEPS, eta=DDIM_ETA,
                                          x0=z, mask=m32, embedding_position=position, adaptive_mask=False,
                                          embedding_manager=model.cond_stage_model, decoder=model.decode_first_stage)
            x = model.decode_first_stage(samples).float()[0]
        if not torch.isfinite(x).all():                             # 不写入 NaN 图像（会被存成全黑）
            raise RuntimeError(f"{jid}: non-finite output (latent finite: {bool(torch.isfinite(samples).all())}, "
                               f"spatial-encoder |max| {float(model.embedding_manager.spatial_encoder_model.last_absmax):.3g})")
        x = x.clamp(-1, 1)
        Image.fromarray(((x.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8)).save(OUT / "images" / f"{jid}.png")
        (OUT / "labels" / f"{jid}.txt").write_text(f"{c} {ox / SIZE:.6f} {oy / SIZE:.6f} {w / SIZE:.6f} {h / SIZE:.6f}\n")
        if (i + 1) % 10 == 0 or i + 1 == len(jobs):
            print(f"{i + 1}/{len(jobs)} {(time.time() - t0) / (i + 1):.2f} s/image "
                  f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prepare":
        prepare()
    elif cmd == "train":
        train(sys.argv[2:])
    elif cmd == "gen":
        gen(int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    else:
        raise SystemExit(__doc__)
