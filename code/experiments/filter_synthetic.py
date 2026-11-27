"""用只在真实训练数据上训练的检测器筛选合成样本（DefFiller README 的建议：预测框与标签框同类且 IoU ≥ 0.6 才保留）

python filter_synthetic.py --method crg --weights <best.pt>
  读 data/synthetic/<method>_A/，保留的样本以符号链接写到 data/synthetic/<method>filt_A/{images,labels}/，
  每张图的结果写到 <method>filt_A/filter.csv。之后用 detect.py --config <method>filt_all --split A 训练检测器。
筛选只用训练数据训练的检测器，不接触 select / test。
"""
import argparse
import csv
from collections import Counter

from common import CLASSES, SYN

IOU_MIN, IMGSZ, SIZE = 0.6, 640, 256


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--weights", required=True)
    args = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(args.weights)
    src, dst = SYN / f"{args.method}_A", SYN / f"{args.method}filt_A"
    (dst / "images").mkdir(parents=True, exist_ok=True); (dst / "labels").mkdir(parents=True, exist_ok=True)
    rows, kept = [], Counter()
    for img in sorted((src / "images").glob("j*.png")):
        c, cx, cy, w, h = (src / "labels" / f"{img.stem}.txt").read_text().split()
        c, cx, cy, w, h = int(c), *(float(v) * SIZE for v in (cx, cy, w, h))
        box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        r = model.predict(str(img), imgsz=IMGSZ, conf=0.25, verbose=False)[0]
        best = max((iou(box, b) for b, k in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()) if int(k) == c), default=0.0)
        keep = best >= IOU_MIN
        if keep:
            kept[CLASSES[c]] += 1
            for sub, name in (("images", img.name), ("labels", f"{img.stem}.txt")):
                link = dst / sub / name
                if not link.exists():
                    link.symlink_to((src / sub / name).resolve())
        rows.append({"jid": img.stem, "class": CLASSES[c], "best_iou": f"{best:.3f}", "kept": int(keep)})
    with open(dst / "filter.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(f"kept {sum(kept.values())} of {len(rows)}:", dict(kept), flush=True)


if __name__ == "__main__":
    main()
