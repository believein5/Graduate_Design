"""协同引导原理图的素材（2026-11-27）：对一个生成任务运行一次协同引导，记录每个去噪步的中间结果
python capture_guidance.py [--jid j0021]   → results/experiments/principle/<jid>.npz
记录：无缺陷图像、窗口位置、起始加噪图像，以及每个去噪步（时间步、x̂0、ĉ0、A_d、A_b、即时掩码 A^(t)、x_{t-1}）。
采样设置与 generate.py --method crg 相同（同一随机种子），不改动 third_party。
"""
import argparse

import numpy as np
import torch
from PIL import Image

import generate as G
from common import CLASSES, FREE, RES, SYN, window_of

OUT = RES / "principle"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jid", default="j0021")
    ap.add_argument("--jobs", default="jobs_A")
    args = ap.parse_args()
    J = SYN / args.jobs
    job = next(j for j in __import__("csv").DictReader(open(J / "jobs.csv")) if j["jid"] == args.jid)
    cid, seed = CLASSES.index(job["class"]), int(args.jid[1:])
    torch.manual_seed(seed); np.random.seed(seed)
    bg = Image.open(J / f"{args.jid}.png").convert("RGB")
    lab = open(J / f"{args.jid}.txt").read().split()
    lx, ly = window_of(int(float(lab[1]) * FREE), int(float(lab[2]) * FREE))

    gen = G.Generator(G.CKPT_64, G.CKPT_256)
    s, cfg = gen.sampler("crg")
    rec = []
    orig = s.p_sample

    def p_sample(*a, **k):
        out = orig(*a, **k)
        g = lambda v: v.detach().float().cpu().numpy()[0]
        rec.append(dict(t=int(k["t"][0]), x0=g(out["pred_x0"]), c0=g(out["pred_crop0"]), ad=g(out["attn_defect"]),
                        ab=g(out["attn_bg"]), ai=g(out["attn_defect_instant"]), xprev=g(out["x_prev"])))
        return out

    s.p_sample = p_sample
    img = G.to_t(bg)
    start = cfg["ddim"]["schedule_params"]["start_step"]
    torch.manual_seed(seed); np.random.seed(seed)
    noised = s.q_sample(img, torch.tensor([start], device=G.DEV))
    mk = {"img_free": img, "img": img.clone(), "crop_y": torch.tensor([cid], device=G.DEV), "location": [lx, ly],
          "crop_t": noised[:, :, ly:ly + G.PATCH, lx:lx + G.PATCH].clone(), "noise_step": start, "image_base_name": "x"}
    model_fn = lambda x, t, y=None, gt=None, **kw: gen.unet(x, t, None, gt=gt)
    res = s.p_sample_loop(model_fn, shape=tuple(img.shape), guide_models=gen.dpm, noise=noised, model_kwargs=mk,
                          device=G.DEV, conf=cfg, start_step=start)["sample"]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{args.jid}.npz", free=G.to_t(bg).cpu().numpy()[0], noised=noised.cpu().numpy()[0],
                        result=res.detach().cpu().numpy()[0], loc=np.array([lx, ly]), cls=cid, start=start,
                        t=np.array([r["t"] for r in rec]), **{k: np.stack([r[k] for r in rec]) for k in ("x0", "c0", "ad", "ab", "ai", "xprev")})
    G.to_im(res).save(OUT / f"{args.jid}_result.png")
    print(args.jid, job["class"], "window", lx, ly, "steps", len(rec), "t:", [r["t"] for r in rec][:12], "...", [r["t"] for r in rec][-6:])


if __name__ == "__main__":
    main()
