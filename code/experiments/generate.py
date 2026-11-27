"""生成合成缺陷数据：data/synthetic/<method>/{images,labels}/jXXXX.{png,txt}

用法: python generate.py --method crg --start 0 --end 300 [--shard 0/2] [--ckpt-64 ...] [--ckpt-256 ...]

方法（全部用同一份任务列表 jobs/，同一背景、同一位置、同一标签框）：
  copypaste  同类同板另一张照片中的真实缺陷，贴到放置位置（参考基线）
  dpmpaste   64 模型（DPM）从噪声生成的缺陷块直接贴进 64 窗口，不做融合
  repaint    把 dpmpaste 的结果加噪到 T_start，用 256 模型（CSM）做 RePaint 融合（窗口外为已知区域，2 次重采样）
  copaint    同样从加噪的 dpmpaste 出发，用仓库的 CoPaint 式优化，只保留背景一致项
  ecics      仓库采样器，关闭注意力引导：DEFectDiffuser 论文 Eq.4 的损失（10·缺陷项 + 0.06·背景项）；也是消融的 base
  abl_sig    + Sigmoid 激活（无高斯平滑、无连通域过滤、无 EMA）
  abl_gs     + 高斯平滑
  abl_ccf    + 连通域过滤
  crg        + EMA 时间稳定 = 作者的完整 CRG（confs/pcb_gen.yml 原样）
  crgpaper   检查：CRG 按论文 Eq. 17 的设置（见 paper_ddim()）；其余与 crg 相同
  crgwin     诊断：只取 crg 结果中的 64×64 窗口，羽化后贴回原背景（窗口外与真实背景完全相同；需要 crg 先生成）
所有扩散方法都从同一起点开始：背景（或贴好缺陷的背景）加噪到 250 步中的第 30 步（T_start = 30）。
标签框：放置位置的真实缺陷框大小，以 64 窗口中心为中心；所有方法相同，只有图像内容不同。
"""
import argparse
import copy
import csv
import sys
import time

import numpy as np
import torch
from PIL import Image

from common import CKPT_256, CKPT_64, CLASSES, FREE, PATCH, PROC, REPO, SYN, window_of

sys.path.insert(0, str(REPO))
import conf_mgt                                                            # noqa: E402
from guided_diffusion import ddim as D                                     # noqa: E402
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule    # noqa: E402
from guided_diffusion.script_util import (create_gaussian_diffusion, create_model, diffusion_defaults,  # noqa: E402
                                          model_defaults, select_args)
from utils import yamlread                                                 # noqa: E402
from utils.config import Config                                            # noqa: E402

DEV = "cuda"
JOBS = SYN / "jobs"
PAPER_K = 1e5 / (3 * PATCH * PATCH)                                     # crgpaper 的权重换算系数（见 PAPER_EDITS 上方说明）
CRG_LIKE = {
    "ecics":   {"attention_features": {"enabled": False}},
    "abl_sig": {"attention_features": {"enabled": True, "blur_sigma": 0.0, "connected_components": False, "accumulation_enabled": False}},
    "abl_gs":  {"attention_features": {"enabled": True, "blur_sigma": 1.0, "connected_components": False, "accumulation_enabled": False}},
    "abl_ccf": {"attention_features": {"enabled": True, "blur_sigma": 1.0, "connected_components": True, "accumulation_enabled": False}},
    "crg":     {},
    "copaint": {"attention_features": {"enabled": False}},
    "crgpaper": {"optimize_xt": {"coef_xt_reg": 0.01 * PAPER_K}},
}
METHODS = ["copypaste", "dpmpaste", "repaint", "crgwin", *CRG_LIKE]

D.visualize_and_save_attention_process = lambda *a, **k: None             # 不写每张图的注意力可视化
_original_loss = D.original_loss
BG_ONLY = {"on": False}


def _loss(_x0, _pred_x0, _mask=None):
    """CoPaint：去掉缺陷一致项（无 mask 的那次调用），只保留背景一致项"""
    if BG_ONLY["on"] and _mask is None:
        return (_x0 * 0).sum()
    return _original_loss(_x0, _pred_x0, _mask)


D.original_loss = _loss

# crgpaper：论文 Eq. 17 与发布代码的差别（代码不改动 third_party，在内存中生成一份修改后的 ddim 模块）
#  1) ddim.py:848 `if t[0] / 10 == 0` 只在 t = 0 时成立；按意图改为每 10 步（t[0] % 10 == 0）重新同步 DPM 预测
#  2) 论文明确不对 ROI 内的背景像素施加损失；代码有一项 1e4·mean(attn_bg·(x̂CSM − xfree)²) → 去掉
#  3) 论文的权重 λ = 1 / 0.1 / 0.01 作用于平方和 ‖·‖²；代码各项先取平均再乘 1e5 / 5e4 / 1e-4。
#     这里按论文的平方和形式，并整体乘 K = 1e5 / (3·64·64)，使缺陷项与代码完全相同（学习率 lr_xt 仍然适用）：
#     ROI 外背景项 = 0.1·K·sum(...)，轨迹正则 = 0.01·K·reg_fn（reg_fn 本身就是平方和），即 coef_xt_reg = 0.01·K
#  每步的梯度步数由 g_scha 决定（2，与论文 N_opt = 2 相同；num_iteration_optimize_xt = 4 在代码中未使用），不改
PAPER_EDITS = [
    ("if t[0] / 10 == 0:", "if t[0] % 10 == 0:"),
    ("weighted_crop_loss = defect_loss_term + bg_loss_term", "weighted_crop_loss = defect_loss_term"),
    ("outside_loss = torch.mean(bg_mask * (pred_x0 - img_free)**2) * 5e4",
     f"outside_loss = torch.sum(bg_mask * (pred_x0 - img_free)**2) * {0.1 * PAPER_K!r}"),
]


# 扫描（--scan）：三项损失权重的倍数与 % 10 修正，同样在内存中改 ddim.py 源码
LOSS_LINES = {"w_def": ("defect_loss_term = torch.mean(attn_defect * (pred_x0_crop - pred_crop0)**2) * ", 1e5, "1e5"),
              "w_bg": ("bg_loss_term = torch.mean(attn_bg * (pred_x0_crop - img_free_crop)**2) * ", 1e4, "1e4"),
              "w_out": ("outside_loss = torch.mean(bg_mask * (pred_x0 - img_free)**2) * ", 5e4, "5e4")}
OVERRIDE = {}                                                              # --scan 的配置覆盖项


def code_edits(code):
    edits = [("if t[0] / 10 == 0:", "if t[0] % 10 == 0:")] if code.get("fix10") else []
    for k, (head, w, lit) in LOSS_LINES.items():
        if code.get(k, 1) != 1:
            edits.append((head + lit, head + repr(w * code[k])))
    return edits


def patched_ddim(edits):
    import types
    src = (REPO / "guided_diffusion" / "ddim.py").read_text()
    for a, b in edits:
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    m = types.ModuleType("guided_diffusion.ddim_paper"); m.__package__ = "guided_diffusion"
    m.__file__ = str(REPO / "guided_diffusion" / "ddim.py")
    exec(compile(src, m.__file__, "exec"), m.__dict__)
    m.visualize_and_save_attention_process = lambda *a, **k: None
    return m


def deep_update(d, u):
    for k, v in u.items():
        d[k] = deep_update(d.get(k, {}), v) if isinstance(v, dict) else v
    return d


class Generator:
    def __init__(self, ckpt_64, ckpt_256):
        base = yamlread(str(REPO / "confs" / "pcb_gen.yml"))
        self.base = base
        conf = conf_mgt.conf_base.Default_Conf(); conf.update(copy.deepcopy(base))
        self.unet = create_model(**select_args(Config(default_config_dict=conf, use_argparse=False),
                                               model_defaults().keys())).to(DEV).eval()
        self.unet.load_state_dict(torch.load(ckpt_256, map_location=DEV, weights_only=True))
        self.dpm = create_model(**base["crop"], num_classes=len(CLASSES)).to(DEV).eval()
        self.dpm.load_state_dict(torch.load(ckpt_64, map_location=DEV, weights_only=True))
        self.samplers = {}
        self.ab1000 = torch.tensor(np.cumprod(1 - get_named_beta_schedule("linear", 1000, use_scale=True)),
                                   dtype=torch.float32, device=DEV)

    def sampler(self, method):
        """按方法覆盖配置并建采样器（A_DDIMSampler 在构造时就读取时间表，所以每种配置一个实例）"""
        if method not in self.samplers:
            d = deep_update(copy.deepcopy(self.base), CRG_LIKE.get(method, {}))
            over = copy.deepcopy(OVERRIDE); code = over.pop("_code", {})
            d = deep_update(d, over)
            conf = conf_mgt.conf_base.Default_Conf(); conf.update(d)
            cfg = Config(default_config_dict=conf, use_argparse=False)
            edits = PAPER_EDITS if method == "crgpaper" else code_edits(code)
            base_cls = patched_ddim(edits).A_DDIMSampler if edits else D.A_DDIMSampler
            s = create_gaussian_diffusion(**select_args(cfg, diffusion_defaults().keys()), conf=cfg, base_cls=base_cls)
            self.samplers[method] = (s, cfg)
        return self.samplers[method]

    @torch.no_grad()
    def dpm_patch(self, cid, seed, steps=50):
        """DPM 从纯噪声生成一个 64×64 缺陷块（确定性 DDIM），返回 [-1,1] 张量 (1,3,64,64)"""
        g = torch.Generator(device=DEV).manual_seed(seed)
        x = torch.randn((1, 3, PATCH, PATCH), device=DEV, generator=g)
        y = torch.tensor([cid], device=DEV)
        ts = np.linspace(999, 0, steps).round().astype(int)
        ab = self.ab1000
        for i, t in enumerate(ts):
            eps = self.dpm(x, torch.tensor([int(t)], device=DEV), y)[:, :3]
            x0 = ((x - (1 - ab[t]).sqrt() * eps) / ab[t].sqrt()).clamp(-1, 1)
            x = x0 if i == len(ts) - 1 else ab[ts[i + 1]].sqrt() * x0 + (1 - ab[ts[i + 1]]).sqrt() * eps
        return x

    def crg_like(self, method, img, init, cid, lx, ly):
        """用仓库的 A_DDIMSampler：img 为背景，init 为加噪起点的来源（CRG 用背景本身，CoPaint 用贴好的图）"""
        s, cfg = self.sampler(method)
        start = cfg["ddim"]["schedule_params"]["start_step"]
        mk = {"img_free": img, "img": img.clone(), "crop_y": torch.tensor([cid], device=DEV), "location": [lx, ly]}
        noised = s.q_sample(init, torch.tensor([start], device=DEV))
        mk.update(crop_t=noised[:, :, ly:ly + PATCH, lx:lx + PATCH].clone(), noise_step=start, image_base_name="x")
        model_fn = lambda x, t, y=None, gt=None, **kw: self.unet(x, t, None, gt=gt)
        BG_ONLY["on"] = method == "copaint"
        try:
            out = s.p_sample_loop(model_fn, shape=tuple(img.shape), guide_models=self.dpm, noise=noised,
                                  model_kwargs=mk, device=DEV, conf=cfg, start_step=start)["sample"]
        finally:
            BG_ONLY["on"] = False
        return out

    @torch.no_grad()
    def repaint(self, img, init, lx, ly, start=30, resample=2):
        """RePaint（DDIM σ=0）：窗口外每步替换为背景的前向加噪结果，窗口内由 CSM 去噪；每步重采样 resample 次"""
        s, _ = self.sampler("crg")
        ab = torch.tensor(s.alphas_cumprod, dtype=torch.float32, device=DEV)     # 250 步的累计 α
        net = s._wrap_model(lambda x, t, **kw: self.unet(x, t, None))
        keep = torch.ones_like(img); keep[:, :, ly:ly + PATCH, lx:lx + PATCH] = 0
        x = ab[start].sqrt() * init + (1 - ab[start]).sqrt() * torch.randn_like(init)
        for t in range(start, 0, -1):
            for r in range(resample):
                e = net(x, torch.tensor([t], device=DEV))[:, :3]
                x0 = ((x - (1 - ab[t]).sqrt() * e) / ab[t].sqrt()).clamp(-1, 1)
                x_un = ab[t - 1].sqrt() * x0 + (1 - ab[t - 1]).sqrt() * e
                x_kn = ab[t - 1].sqrt() * img + (1 - ab[t - 1]).sqrt() * torch.randn_like(img)
                x_prev = keep * x_kn + (1 - keep) * x_un
                if r < resample - 1:                                              # 回到 t 再走一次
                    a = ab[t] / ab[t - 1]
                    x = a.sqrt() * x_prev + (1 - a).sqrt() * torch.randn_like(x_prev)
            x = x_prev
        return x


def to_t(im):
    return torch.from_numpy(np.asarray(im, np.float32) / 127.5 - 1).permute(2, 0, 1)[None].to(DEV)


def to_im(x):
    return Image.fromarray(((x[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy())


def label_line(job, lx, ly):
    """放置框大小，以 64 窗口中心为中心"""
    w, h = float(job["bx1"]) - float(job["bx0"]), float(job["by1"]) - float(job["by0"])
    cx, cy = lx + PATCH / 2, ly + PATCH / 2
    return f"{CLASSES.index(job['class'])} {cx / FREE:.6f} {cy / FREE:.6f} {w / FREE:.6f} {h / FREE:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=900)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--ckpt-64", default=str(CKPT_64))
    ap.add_argument("--ckpt-256", default=str(CKPT_256))
    ap.add_argument("--tag", default="", help="输出目录后缀，例如 _ours，用于换自训 checkpoint")
    ap.add_argument("--jobs", default="jobs", help="data/synthetic/ 下的任务目录（划分 A 用 jobs_A）")
    ap.add_argument("--scan", default=None, help="scan_configs.SCAN 中的设置名（配合 --method crg；默认输出到 crgscan_<名字>_A）")
    args = ap.parse_args()
    if args.scan:
        import scan_configs
        OVERRIDE.update(scan_configs.get(args.scan))
        args.tag = args.tag or f"scan_{args.scan}_A"
    global JOBS
    JOBS = SYN / args.jobs
    k, n = map(int, args.shard.split("/"))
    jobs = list(csv.DictReader(open(JOBS / "jobs.csv")))[args.start:args.end]
    jobs = [j for i, j in enumerate(jobs) if i % n == k]
    out = SYN / f"{args.method}{args.tag}"
    (out / "images").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(parents=True, exist_ok=True)
    dpm_dir = SYN / f"dpm_patch{args.tag}"; dpm_dir.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (out / "labels" / f"{j['jid']}.txt").exists()]
    if args.method == "crgwin":                                                # 只处理 crg 已生成的任务
        todo = [j for j in todo if (SYN / f"crg{args.tag}" / "labels" / f"{j['jid']}.txt").exists()]
    print(f"{args.method}{args.tag}: {len(jobs)} jobs in range, {len(todo)} to do", flush=True)
    gen = Generator(args.ckpt_64, args.ckpt_256) if args.method not in ("copypaste", "crgwin") else None
    t0 = time.time()
    for i, job in enumerate(todo):
        jid, cid = job["jid"], CLASSES.index(job["class"])
        seed = int(jid[1:])
        torch.manual_seed(seed); np.random.seed(seed)
        bg = Image.open(JOBS / f"{jid}.png").convert("RGB")
        cx = int(float(open(JOBS / f"{jid}.txt").read().split()[1]) * FREE)
        cy = int(float(open(JOBS / f"{jid}.txt").read().split()[2]) * FREE)
        lx, ly = window_of(cx, cy)
        if args.method == "crgwin":
            crg_img = np.asarray(Image.open(SYN / f"crg{args.tag}" / "images" / f"{jid}.png").convert("RGB"), np.float32)
            a = np.zeros((FREE, FREE), np.float32); a[ly:ly + PATCH, lx:lx + PATCH] = 1.0
            from scipy import ndimage
            a = ndimage.uniform_filter(a, 9)[..., None] * (ndimage.uniform_filter(a, 9)[..., None] > 0)  # 窗口边缘 ±4 px 线性过渡
            res = Image.fromarray((a * crg_img + (1 - a) * np.asarray(bg, np.float32)).round().clip(0, 255).astype(np.uint8))
            line = (SYN / f"crg{args.tag}" / "labels" / f"{jid}.txt").read_text().strip()
        elif args.method == "copypaste":
            from common import photo
            sw, sh = float(job["sx1"]) - float(job["sx0"]), float(job["sy1"]) - float(job["sy0"])
            m = 4                                                                 # 带 4 px 边缘的真实缺陷
            sx, sy = (float(job["sx0"]) + float(job["sx1"])) / 2, (float(job["sy0"]) + float(job["sy1"])) / 2
            half_w, half_h = sw / 2 + m, sh / 2 + m
            patch = photo(job["src_photo"]).crop((int(sx - half_w), int(sy - half_h), int(sx + half_w), int(sy + half_h)))
            res = bg.copy(); res.paste(patch, (int(lx + PATCH / 2 - patch.width / 2), int(ly + PATCH / 2 - patch.height / 2)))
            line = f"{cid} {(lx + PATCH / 2) / FREE:.6f} {(ly + PATCH / 2) / FREE:.6f} {sw / FREE:.6f} {sh / FREE:.6f}"
        else:
            img = to_t(bg)
            pp = dpm_dir / f"{jid}.png"
            if pp.exists():
                patch = to_t(Image.open(pp).convert("RGB"))
            else:
                patch = gen.dpm_patch(cid, seed)
                to_im(patch).save(pp)
            pasted = img.clone(); pasted[:, :, ly:ly + PATCH, lx:lx + PATCH] = patch
            if args.method == "dpmpaste":
                x = pasted
            elif args.method == "repaint":
                x = gen.repaint(img, pasted, lx, ly)
            elif args.method == "copaint":
                x = gen.crg_like("copaint", img, pasted, cid, lx, ly)
            else:
                x = gen.crg_like(args.method, img, img, cid, lx, ly)
            res = to_im(x)
            line = label_line(job, lx, ly)
        res.save(out / "images" / f"{jid}.png")
        (out / "labels" / f"{jid}.txt").write_text(line + "\n")
        if (i + 1) % 10 == 0 or i == len(todo) - 1:
            el = time.time() - t0
            print(f"  {i + 1}/{len(todo)} · {el / (i + 1):.1f} s/img · {el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min left", flush=True)


if __name__ == "__main__":
    main()
