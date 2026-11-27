"""扫描的代理评分：用只在划分 A 真实训练数据上训练的 Original 检测器（640，种子 0）给合成图打分；不接触 select / test

python score_proxy.py --dir crgscan_t20_A --name t20 [--n 60]
结果写到 results/experiments/scan/proxy/<名字>_n<n>.json：
  det_rate    检测器在标签框处找到同类缺陷（IoU ≥ 0.5，置信度 ≥ 0.25）的比例，另有按类别的比例
  mean_iou    同类预测框与标签框最大 IoU 的平均
  bg_change   64×64 窗口外与原背景的平均绝对差（0–255）
  win_change  窗口内的平均绝对差
检测器还在训练时先等它训练完（metrics.json 出现）；合成图不足 n 张时也等。
"""
import argparse
import csv
import json
import time
from collections import defaultdict

import numpy as np
from PIL import Image

from common import CLASSES, RES, SYN

DET_RUN = RES / "detection_diag_img640" / "original" / "seed0"
OUT = RES / "scan" / "proxy"
SIZE, PATCH, IOU_MIN = 256, 64, 0.5


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    src = SYN / args.dir
    jids = [j["jid"] for j in csv.DictReader(open(SYN / "jobs_A" / "jobs.csv"))][:args.n]
    t0 = time.time()
    while not (DET_RUN / "metrics.json").exists() or not all((src / "labels" / f"{j}.txt").exists() for j in jids):
        if time.time() - t0 > 12 * 3600:
            raise SystemExit("gave up waiting for the detector or the images")
        time.sleep(60)
    from ultralytics import YOLO
    model = YOLO(str(DET_RUN / "train" / "weights" / "best.pt"))
    hit, ious, bg, win, per = [], [], [], [], defaultdict(list)
    for jid in jids:
        c, cx, cy, w, h = (src / "labels" / f"{jid}.txt").read_text().split()
        c, cx, cy, w, h = int(c), *(float(v) * SIZE for v in (cx, cy, w, h))
        box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        r = model.predict(str(src / "images" / f"{jid}.png"), imgsz=640, conf=0.25, verbose=False)[0]
        best = max((iou(box, b) for b, k in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()) if int(k) == c), default=0.0)
        hit.append(best >= IOU_MIN); ious.append(best); per[CLASSES[c]].append(best >= IOU_MIN)
        g = np.asarray(Image.open(src / "images" / f"{jid}.png").convert("RGB"), np.float32)
        b0 = np.asarray(Image.open(SYN / "jobs_A" / f"{jid}.png").convert("RGB"), np.float32)
        lx, ly = int(round(cx - PATCH / 2)), int(round(cy - PATCH / 2))
        m = np.zeros((SIZE, SIZE), bool); m[ly:ly + PATCH, lx:lx + PATCH] = True
        d = np.abs(g - b0).mean(-1)
        bg.append(float(d[~m].mean())); win.append(float(d[m].mean()))
    res = {"name": args.name, "dir": args.dir, "n": len(jids), "det_rate": float(np.mean(hit)), "mean_iou": float(np.mean(ious)),
           "bg_change": float(np.mean(bg)), "win_change": float(np.mean(win)),
           "det_rate_per_class": {k: float(np.mean(v)) for k, v in per.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f"{args.name}_n{args.n}.json", "w"), indent=1)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
