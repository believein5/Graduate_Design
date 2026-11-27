"""检查合成样本的标签框是否落在实际生成的缺陷上（只用 CPU）

做法：生成图与它的背景（jobs/jXXXX.png）逐像素比较，差异区域就是方法实际改动的地方。
在 64 窗口（外扩 16 px）内取超过阈值的像素，去掉小于 MIN_AREA 的碎片，
其外接框即“实际缺陷框”，与标签框比较 IoU、中心偏移和面积比。
输出 results/experiments/box_check/{summary.csv, per_image.csv, examples_<method>.png}
"""
import csv

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from common import CLASSES, FREE, PATCH, RES, SYN, window_of

METHODS = ["copypaste", "dpmpaste", "repaint", "copaint", "ecics", "crg"]
N = 300
THRESH = 30            # 3×3 平均后、三通道最大差异的阈值（0–255）
MIN_AREA = 6           # 小于此像素数的连通块视为噪声
PAD = 16               # 在 64 窗口外再看 16 px
OUT = RES / "box_check"


def label_box(path):
    _, cx, cy, w, h = map(float, path.read_text().split())
    return np.array([(cx - w / 2) * FREE, (cy - h / 2) * FREE, (cx + w / 2) * FREE, (cy + h / 2) * FREE])


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def change_box(gen, bg, lx, ly):
    d = np.abs(gen.astype(np.int16) - bg.astype(np.int16)).max(2).astype(np.float32)
    d = ndimage.uniform_filter(d, 3)
    x0, y0, x1, y1 = max(0, lx - PAD), max(0, ly - PAD), min(FREE, lx + PATCH + PAD), min(FREE, ly + PATCH + PAD)
    outside = np.ones_like(d, bool); outside[y0:y1, x0:x1] = False
    floor = float(np.percentile(d[outside], 99.5))                  # 窗口外的重建噪声水平
    m = np.zeros_like(d, bool); m[y0:y1, x0:x1] = d[y0:y1, x0:x1] > THRESH
    lab, n = ndimage.label(m)
    if n:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, 1 + np.nonzero(sizes >= MIN_AREA)[0])
    if not m.any():
        return None, floor, 0
    ys, xs = np.nonzero(m)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float), floor, int(m.sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = {r["jid"]: r for r in csv.DictReader(open(SYN / "jobs" / "jobs.csv"))}
    rows, summary = [], []
    for method in METHODS:
        res = []
        for i in range(N):
            jid = f"j{i:04d}"
            gp = SYN / method / "images" / f"{jid}.png"
            if not gp.exists():
                continue
            cx, cy = [float(v) * FREE for v in (SYN / "jobs" / f"{jid}.txt").read_text().split()[1:3]]
            lx, ly = window_of(int(cx), int(cy))
            bg = np.asarray(Image.open(SYN / "jobs" / f"{jid}.png").convert("RGB"))
            gen = np.asarray(Image.open(gp).convert("RGB"))
            lb = label_box(SYN / method / "labels" / f"{jid}.txt")
            cb, floor, area = change_box(gen, bg, lx, ly)
            r = dict(method=method, jid=jid, cls=jobs[jid]["class"], floor=floor, area=area,
                     iou=iou(lb, cb) if cb is not None else 0.0, detected=cb is not None,
                     offset=float(np.hypot(*((cb[:2] + cb[2:]) / 2 - (lb[:2] + lb[2:]) / 2))) if cb is not None else np.nan,
                     area_ratio=float((cb[2] - cb[0]) * (cb[3] - cb[1]) / ((lb[2] - lb[0]) * (lb[3] - lb[1]))) if cb is not None else np.nan,
                     lb=lb, cb=cb)
            res.append(r)
        if not res:
            continue
        iou_ = np.array([r["iou"] for r in res]); det = np.array([r["detected"] for r in res])
        s = dict(method=method, n=len(res), changed=det.mean(), iou_median=float(np.median(iou_)),
                 iou_ge_05=float((iou_ >= 0.5).mean()), iou_lt_03=float((iou_ < 0.3).mean()),
                 offset_median=float(np.nanmedian([r["offset"] for r in res])),
                 area_ratio_median=float(np.nanmedian([r["area_ratio"] for r in res])),
                 noise_floor_median=float(np.median([r["floor"] for r in res])))
        for c in CLASSES:
            ic = np.array([r["iou"] for r in res if r["cls"] == c]); s[f"iou_ge_05_{c}"] = float((ic >= 0.5).mean())
        summary.append(s); rows += res
        print(f"{method:10s} changed {s['changed']:5.1%} · IoU median {s['iou_median']:.2f} · IoU≥0.5 {s['iou_ge_05']:5.1%} · "
              f"IoU<0.3 {s['iou_lt_03']:5.1%} · centre offset {s['offset_median']:4.1f} px · area ratio {s['area_ratio_median']:.2f} · "
              f"noise outside window {s['noise_floor_median']:.1f}")
        print("           IoU≥0.5 per class: " + ", ".join(f"{c} {s['iou_ge_05_' + c]:.0%}" for c in CLASSES))

        # 示例图：每类 2 张，红 = 标签框，青 = 实际改动框；上图生成结果，下图背景
        Z, tiles = 3, []
        for c in CLASSES:
            for r in [r for r in res if r["cls"] == c][:2]:
                cxw, cyw = (r["lb"][0] + r["lb"][2]) / 2, (r["lb"][1] + r["lb"][3]) / 2
                x0, y0 = int(np.clip(cxw - 48, 0, FREE - 96)), int(np.clip(cyw - 48, 0, FREE - 96))
                pair = []
                for src in [SYN / method / "images" / f"{r['jid']}.png", SYN / "jobs" / f"{r['jid']}.png"]:
                    im = Image.open(src).convert("RGB").crop((x0, y0, x0 + 96, y0 + 96)).resize((96 * Z, 96 * Z), Image.NEAREST)
                    dr = ImageDraw.Draw(im)
                    dr.rectangle([(v - o) * Z for v, o in zip(r["lb"], (x0, y0, x0, y0))], outline=(255, 0, 0), width=2)
                    if r["cb"] is not None:
                        dr.rectangle([(v - o) * Z for v, o in zip(r["cb"], (x0, y0, x0, y0))], outline=(0, 255, 255), width=2)
                    dr.text((4, 4), f"{c} IoU {r['iou']:.2f}" if src.parent.name == "images" else "background", fill=(255, 255, 0))
                    pair.append(np.asarray(im))
                tiles.append(np.concatenate(pair, 0))
        Image.fromarray(np.concatenate(tiles, 1)).save(OUT / f"examples_{method}.png")

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    with open(OUT / "per_image.csv", "w", newline="") as f:
        keys = ["method", "jid", "cls", "detected", "iou", "offset", "area_ratio", "area", "floor"]
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows({k: r[k] for k in keys} for r in rows)


if __name__ == "__main__":
    main()
