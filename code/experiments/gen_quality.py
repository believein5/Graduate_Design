"""各方法生成图像的质量评价（2026-11-25，对应 CRG 论文表 II 的 KID / LPIPS），只用 1× 的 300 张（j0000–j0299）
python gen_quality.py <方法目录名> [<方法目录名> ...]      例如 crg_A dpmpaste_A repaint_A
参考集为划分 A 的选择集（300 张真实 256×256 缺陷图像，与训练、测试照片都不重叠），不用测试集。
  kid_full / fid_full   整幅 256×256 图像与参考集比较
  kid_win  / fid_win    以标签框中心截取的 64×64 缺陷窗口，与参考集中以标注框中心截取的窗口比较（缺陷区域的逼真程度）
  lpips_div             同类生成窗口两两之间 LPIPS（VGG）的平均，越大越多样；参考集同样计算一次作对照
结果：results/experiments/gen_quality/<目录名>.json（已存在则跳过）
"""
import itertools
import json
import random
import sys

import numpy as np
import torch
from PIL import Image

from common import CLASSES, RES, SYN

DET_A = SYN.parent / "detection_A"
OUT = RES / "gen_quality"
SIZE, WIN, N = 256, 64, 300
PAIRS = 400                                  # 每类随机抽取的窗口对数
DEV = "cuda"


def window(img, cx, cy):
    l = int(round(min(max(cx * SIZE - WIN / 2, 0), SIZE - WIN)))
    t = int(round(min(max(cy * SIZE - WIN / 2, 0), SIZE - WIN)))
    return img.crop((l, t, l + WIN, t + WIN))


def load(img_dir, lbl_dir, names):
    full, win, cls = [], [], []
    for n in names:
        img = Image.open(img_dir / f"{n}.png").convert("RGB")
        c, cx, cy = (float(v) for v in (lbl_dir / f"{n}.txt").read_text().split()[:3])
        full.append(np.asarray(img)); win.append(np.asarray(window(img, cx, cy))); cls.append(int(c))
    t = lambda a: torch.from_numpy(np.stack(a)).permute(0, 3, 1, 2).contiguous()
    return t(full), t(win), np.array(cls)


class U8(torch.utils.data.Dataset):
    def __init__(self, x): self.x = x
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]


def fidelity(gen, real):
    import torch_fidelity
    m = torch_fidelity.calculate_metrics(input1=U8(gen), input2=U8(real), cuda=True, fid=True, kid=True, verbose=False,
                                         kid_subset_size=min(len(gen), len(real), 1000))
    return m["kernel_inception_distance_mean"] * 1e3, m["frechet_inception_distance"]


def diversity(win, cls, fn):
    rng = random.Random(0)
    out = {}
    for k, name in enumerate(CLASSES):
        idx = np.flatnonzero(cls == k)
        pairs = list(itertools.combinations(idx, 2)); rng.shuffle(pairs); pairs = pairs[:PAIRS]
        a = win[[p[0] for p in pairs]].float().div(127.5).sub(1).to(DEV)
        b = win[[p[1] for p in pairs]].float().div(127.5).sub(1).to(DEV)
        with torch.no_grad():
            out[name] = float(torch.cat([fn(a[i:i + 100], b[i:i + 100]).flatten() for i in range(0, len(a), 100)]).mean())
    return float(np.mean(list(out.values()))), out


def main():
    import lpips
    fn = lpips.LPIPS(net="vgg", verbose=False).to(DEV)
    OUT.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in (DET_A / "select" / "images").glob("*.png"))
    r_full, r_win, r_cls = load(DET_A / "select" / "images", DET_A / "select" / "labels", names)
    ref = OUT / "_reference.json"
    if not ref.exists():
        d, per = diversity(r_win, r_cls, fn)
        json.dump({"name": "select (real)", "n": len(names), "lpips_div": d, "lpips_div_per_class": per}, open(ref, "w"), indent=1)
    for d in sys.argv[1:]:
        f = OUT / f"{d}.json"
        if f.exists():
            print("done already:", d); continue
        jids = [f"j{i:04d}" for i in range(N)]
        g_full, g_win, g_cls = load(SYN / d / "images", SYN / d / "labels", jids)
        kf, ff = fidelity(g_full, r_full)
        kw, fw = fidelity(g_win, r_win)
        div, per = diversity(g_win, g_cls, fn)
        res = {"name": d, "n": len(jids), "kid_full_x1e3": kf, "fid_full": ff, "kid_win_x1e3": kw, "fid_win": fw,
               "lpips_div": div, "lpips_div_per_class": per}
        json.dump(res, open(f, "w"), indent=1)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k != "lpips_div_per_class"}, flush=True)


if __name__ == "__main__":
    main()
