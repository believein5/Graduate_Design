"""检查 D（2026-11-20）：作者的 256 背景模型（CSM）见过哪些板？
同一块板的所有照片是逐像素对齐的副本，所以无法区分"训练照片"与"测试照片"，只能看板：
- 作者的模型与我们的模型（只用板 01、04–09 训练）各无条件采样 N 张（DDIM 50 步）
- 每块板取一张对齐照片（0.5×），以步长 16 取全部 256 窗口（含水平翻转）作为参考池，缩到 64×64
- 每张样本找像素 L2 最近的窗口，统计最近邻落在哪块板；对照：每块板另一张照片的真实裁剪
我们的模型的最近邻应只落在训练板上（校准）；作者的模型若也落在板 10–12，说明它训练时见过这些板
结果：results/experiments/csm_boards/{summary.json, nn_examples.png}
"""
import json
import random
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import FREE, PKU, RES, ROOT, board, misaligned, photo
import generate as G

OUT = RES / "csm_boards"
N, BATCH, STEPS, STRIDE, SMALL = 300, 20, 50, 16, 64


@torch.no_grad()
def sample(unet, ab, n, seed):
    """无条件 DDIM（σ = 0），与 Generator.dpm_patch 相同的写法"""
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    ts = np.linspace(999, 0, STEPS).round().astype(int)
    for b in range(0, n, BATCH):
        x = torch.randn((min(BATCH, n - b), 3, FREE, FREE), device="cuda", generator=g)
        for i, t in enumerate(ts):
            eps = unet(x, torch.full((x.shape[0],), int(t), device="cuda"), None)[:, :3]
            x0 = ((x - (1 - ab[t]).sqrt() * eps) / ab[t].sqrt()).clamp(-1, 1)
            x = x0 if i == len(ts) - 1 else ab[ts[i + 1]].sqrt() * x0 + (1 - ab[ts[i + 1]]).sqrt() * eps
        out.append(((x + 1) / 2).float())
    return torch.cat(out)


def small(t):
    return F.interpolate(t, size=SMALL, mode="area").flatten(1).half()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    bad = misaligned()
    stems = sorted(p.stem for p in (PKU / "images").glob("*.jpg"))
    boards = sorted({board(s) for s in stems})
    pool, owner, ctrl, ctrl_owner = [], [], [], []
    for b in boards:
        ph = [s for s in stems if board(s) == b and s not in bad]
        ref, other = ph[0], rng.choice(ph[1:])
        im = torch.from_numpy(np.asarray(photo(ref), np.float32) / 255).permute(2, 0, 1)
        _, h, w = im.shape
        pos = [(y, x) for y in range(0, h - FREE + 1, STRIDE) for x in range(0, w - FREE + 1, STRIDE)]
        for i in range(0, len(pos), 256):                                   # 分块在 CPU 上缩小，只把 64×64 结果放到 GPU
            crops = torch.stack([im[:, y:y + FREE, x:x + FREE] for y, x in pos[i:i + 256]])
            pool.append(small(torch.cat([crops, crops.flip(-1)])).cuda())
        owner += [b] * (2 * len(pos))
        im2 = np.asarray(photo(other), np.float32) / 255
        for _ in range(30):
            y, x = rng.randrange(0, h - FREE), rng.randrange(0, w - FREE)
            ctrl.append(torch.from_numpy(im2[y:y + FREE, x:x + FREE]).permute(2, 0, 1)); ctrl_owner.append(b)
        print(f"board {b}: {2 * len(pos)} reference windows from {ref}", flush=True)
    pool, owner = torch.cat(pool), np.array(owner)

    def nn(imgs):
        q = small(imgs.cuda())
        d = torch.cat([torch.cdist(q.float(), pool[i:i + 8192].float()) for i in range(0, len(pool), 8192)], 1)
        v, k = d.min(1)
        return v.cpu().numpy() / SMALL, owner[k.cpu().numpy()], k.cpu().numpy()

    gen = G.Generator(G.CKPT_64, G.CKPT_256)
    res = {}
    models = {"authors": G.CKPT_256, "ours": ROOT / "runs" / "free_256" / "model375000.pt"}
    samples = {}
    for name, ck in models.items():
        gen.unet.load_state_dict(torch.load(ck, map_location="cuda", weights_only=True))
        samples[name] = sample(gen.unet, gen.ab1000, N, seed=1)
    del gen; torch.cuda.empty_cache()
    for name, imgs in [*samples.items(), ("real_control", torch.stack(ctrl))]:
        dist, b, k = nn(imgs)
        res[name] = {"nn_board_share": {x: round(float(np.mean(b == x)), 3) for x in boards},
                     "median_nn_distance": round(float(np.median(dist)), 4)}
        if name == "real_control":
            res[name]["own_board_match_rate"] = round(float(np.mean(b == np.array(ctrl_owner))), 3)
        if name == "authors":
            ex = [(imgs[i], k[i]) for i in range(8)]
        print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open(OUT / "summary.json", "w"), indent=1)
    # 示例：作者模型的 8 张样本（上）与它们的最近参考窗口（下，64×64 放大）
    sheet = Image.new("RGB", (8 * 128, 256))
    flat = pool.float().view(-1, 3, SMALL, SMALL)
    for i, (img, kk) in enumerate(ex):
        a = Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).resize((128, 128))
        r = Image.fromarray((flat[kk].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)).resize((128, 128))
        sheet.paste(a, (i * 128, 0)); sheet.paste(r, (i * 128, 128))
    sheet.save(OUT / "nn_examples.png")


if __name__ == "__main__":
    sys.argv = sys.argv[:1]
    main()
