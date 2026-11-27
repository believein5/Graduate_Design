"""CRG 诊断：为什么 CRG 合成数据让检测器在 spur / short 上的 precision 下降

python diagnose_crg.py inspect   CPU：真实缺陷与合成缺陷的对比度、改动面积，及并排图（spur、short，另附全部类别的统计）
python diagnose_crg.py fp        GPU：用保留权重的检测器（original 与 crg_1x，seed 0）在 test 上找误检，按类统计并画图
输出：results/experiments/diag_crg/
"""
import csv
import json
import random
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from common import CLASSES, DET, FREE, PROC, RES, SYN, TRAIN, board, boxes, misaligned, photo

OUT = RES / "diag_crg"
THRESH = 30                        # 与 check_boxes.py 相同的改动阈值
SHOW = ["spur", "short"]
N_SHOW = 10


def diff_map(a, b):
    return ndimage.uniform_filter(np.abs(a.astype(np.int16) - b.astype(np.int16)).max(2).astype(np.float32), 3)


def stats_in_box(d, box):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    r = d[max(0, y0):y1, max(0, x0):x1]
    return float(r.max()) if r.size else 0.0, int((r > THRESH).sum())


def crop(img, box, size=48, z=4):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x0 = int(np.clip(cx - size / 2, 0, img.width - size)); y0 = int(np.clip(cy - size / 2, 0, img.height - size))
    im = img.crop((x0, y0, x0 + size, y0 + size)).resize((size * z, size * z), Image.NEAREST)
    ImageDraw.Draw(im).rectangle([(box[0] - x0) * z, (box[1] - y0) * z, (box[2] - x0) * z, (box[3] - y0) * z],
                                 outline=(255, 0, 0), width=2)
    return np.asarray(im)


def inspect():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    bad = misaligned()
    real = {c: [] for c in CLASSES}
    for s in rng.sample([s for s in TRAIN if s not in bad], 200):
        ph = photo(s); tp = Image.open(PROC / f"templates/board_{board(s)}.png").convert("RGB")
        d = diff_map(np.asarray(ph), np.asarray(tp))
        for c, *b in boxes(s):
            mx, area = stats_in_box(d, b)
            real[CLASSES[c]].append(dict(contrast=mx, area=area, img=ph, box=b))
    syn = {}
    for m in ["crg", "ecics", "crgwin"]:
        syn[m] = {c: [] for c in CLASSES}
        for lp in sorted((SYN / m / "labels").glob("j*.txt"))[:300]:
            jid = lp.stem
            _, cx, cy, w, h = map(float, lp.read_text().split())
            b = [(cx - w / 2) * FREE, (cy - h / 2) * FREE, (cx + w / 2) * FREE, (cy + h / 2) * FREE]
            g = Image.open(SYN / m / "images" / f"{jid}.png").convert("RGB")
            bg = Image.open(SYN / "jobs" / f"{jid}.png").convert("RGB")
            mx, area = stats_in_box(diff_map(np.asarray(g), np.asarray(bg)), b)
            cls = CLASSES[int(lp.read_text().split()[0])]
            syn[m][cls].append(dict(contrast=mx, area=area, img=g, box=b, bg=bg))

    rows = []
    for c in CLASSES:
        for name, items in [("real", real[c])] + [(m, syn[m][c]) for m in syn]:
            con = np.array([i["contrast"] for i in items]); ar = np.array([i["area"] for i in items])
            rows.append(dict(cls=c, source=name, n=len(items), contrast_median=float(np.median(con)),
                             faint_share=float((con < THRESH).mean()), changed_px_median=float(np.median(ar))))
    with open(OUT / "defect_strength.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{'class':15s} {'source':7s} {'n':>4s} {'contrast':>9s} {'faint':>6s} {'changed px':>10s}")
    for r in rows:
        print(f"{r['cls']:15s} {r['source']:7s} {r['n']:4d} {r['contrast_median']:9.1f} {r['faint_share']:6.0%} {r['changed_px_median']:10.0f}")

    # 并排图：每类一张，行 = 真实 / crg / crg 背景 / ecics
    for c in SHOW:
        lines = [("real (PKU)", [crop(i["img"], i["box"]) for i in real[c][:N_SHOW]]),
                 ("crg", [crop(i["img"], i["box"]) for i in syn["crg"][c][:N_SHOW]]),
                 ("crg background", [crop(i["bg"], i["box"]) for i in syn["crg"][c][:N_SHOW]]),
                 ("ecics", [crop(i["img"], i["box"]) for i in syn["ecics"][c][:N_SHOW]])]
        H = lines[0][1][0].shape[0]; pad = 6; lab = 120
        canvas = Image.new("RGB", (lab + N_SHOW * (H + pad), len(lines) * (H + pad)), "white")
        dr = ImageDraw.Draw(canvas)
        for r, (name, tiles) in enumerate(lines):
            dr.text((4, r * (H + pad) + H // 2), name, fill="black")
            for k, t in enumerate(tiles):
                canvas.paste(Image.fromarray(t), (lab + k * (H + pad), r * (H + pad)))
        canvas.save(OUT / f"compare_{c}.png")


def fp():
    from ultralytics import YOLO
    OUT.mkdir(parents=True, exist_ok=True)
    imgs = sorted((DET / "test" / "images").glob("*.png"))
    gt = {}
    for p in imgs:
        gt[p.stem] = []
        for line in (DET / "test" / "labels" / f"{p.stem}.txt").read_text().splitlines():
            c, cx, cy, w, h = map(float, line.split())
            gt[p.stem].append((int(c), np.array([(cx - w / 2), (cy - h / 2), (cx + w / 2), (cy + h / 2)]) * FREE))

    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        return ix * iy / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy)

    summary = {}
    for cfg in ["original", "crg_1x"]:
        w = RES / "detection_diag" / cfg / "seed0" / "train" / "weights" / "best.pt"
        if not w.exists():
            print("missing weights:", w); continue
        model = YOLO(str(w))
        fps = {c: [] for c in CLASSES}; tps = {c: 0 for c in CLASSES}
        for p in imgs:
            r = model.predict(str(p), imgsz=FREE, conf=0.25, verbose=False)[0]
            used = set()
            for bx, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
                match = [k for k, (gc, gb) in enumerate(gt[p.stem]) if gc == c and k not in used and iou(bx, gb) >= 0.5]
                if match:
                    used.add(match[0]); tps[CLASSES[c]] += 1
                else:
                    near = max([iou(bx, gb) for _, gb in gt[p.stem]] or [0])
                    fps[CLASSES[c]].append(dict(img=p, box=bx, conf=float(s), near_gt_iou=float(near)))
        summary[cfg] = {c: dict(tp=tps[c], fp=len(fps[c]),
                                fp_on_background=sum(f["near_gt_iou"] < 0.1 for f in fps[c])) for c in CLASSES}
        for c in SHOW:
            tiles = [crop(Image.open(f["img"]).convert("RGB"), f["box"], size=64, z=3)
                     for f in sorted(fps[c], key=lambda f: -f["conf"])[:N_SHOW]]
            if tiles:
                Image.fromarray(np.concatenate(tiles, 1)).save(OUT / f"fp_{cfg}_{c}.png")
    json.dump(summary, open(OUT / "false_positives.json", "w"), indent=1)
    for cfg, s in summary.items():
        print(cfg, {c: f"TP {v['tp']} FP {v['fp']} (on clean board {v['fp_on_background']})" for c, v in s.items()})


if __name__ == "__main__":
    {"inspect": inspect, "fp": fp}[sys.argv[1]]()
