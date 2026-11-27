"""合成图标签框修正（2026-11-25，P6 思路 2）：data/synthetic/<src>/ → data/synthetic/<dst>/
原来合成图的标签框 = 放置处真实缺陷标注框的大小、以 64 窗口中心为中心，与生成出的缺陷不一定吻合。
这里用只在划分 A 真实训练数据上训练的检测器（detection_A_vbase/original/seed0，选择依据为 select，未接触测试集）
检测每张合成图：取与原标签框同类、IoU ≥ IOU_MIN、置信度 ≥ CONF_MIN 的预测框中 IoU 最大的一个代替原标签框；
找不到时保留原标签框（张数不变，与其他 1× 设置可比）。
python relabel_synthetic.py [--src crg_A] [--dst crgrelab_A] [--n 300]
"""
import argparse
import json
import shutil

import numpy as np

from common import CLASSES, RES, SYN

DET_RUN = RES / "detection_A_vbase" / "original" / "seed0" / "train" / "weights" / "best.pt"
SIZE, IOU_MIN, CONF_MIN = 256, 0.3, 0.25


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="crg_A")
    ap.add_argument("--dst", default="crgrelab_A")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(str(DET_RUN))
    src, dst = SYN / args.src, SYN / args.dst
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "images").mkdir(parents=True); (dst / "labels").mkdir()
    imgs = sorted((src / "images").glob("j*.png"))[:args.n]
    stats, ious, area_ratio = {"replaced": 0, "kept": 0}, [], []
    for p in imgs:
        line = (src / "labels" / f"{p.stem}.txt").read_text().split()
        c, cx, cy, w, h = int(line[0]), *(float(v) * SIZE for v in line[1:])
        lab = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        r = model.predict(str(p), imgsz=640, conf=CONF_MIN, verbose=False)[0].boxes
        cand = [(iou(lab, b), b) for b, k in zip(r.xyxy.tolist(), r.cls.tolist()) if int(k) == c]
        cand = [x for x in cand if x[0] >= IOU_MIN]
        shutil.copy(p, dst / "images" / p.name)
        if cand:
            v, b = max(cand)
            ious.append(v); area_ratio.append((b[2] - b[0]) * (b[3] - b[1]) / (w * h))
            out = f"{c} {(b[0] + b[2]) / 2 / SIZE:.6f} {(b[1] + b[3]) / 2 / SIZE:.6f} {(b[2] - b[0]) / SIZE:.6f} {(b[3] - b[1]) / SIZE:.6f}"
            stats["replaced"] += 1
        else:
            out = " ".join(line)
            stats["kept"] += 1
        (dst / "labels" / f"{p.stem}.txt").write_text(out + "\n")
    stats.update(n=len(imgs), median_iou_old_new=float(np.median(ious)) if ious else None,
                 median_area_new_over_old=float(np.median(area_ratio)) if area_ratio else None, detector=str(DET_RUN))
    json.dump(stats, open(dst / "relabel_stats.json", "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
