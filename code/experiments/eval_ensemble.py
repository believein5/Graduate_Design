"""不重新训练的推理增强（2026-11-24）：测试时增强（TTA）与多种子集成，先在 select 上评分，选定后再看 test

python eval_ensemble.py --runs <结果目录 1> [<结果目录 2> ...] --name <名字> [--tta]
  每个结果目录形如 results/experiments/detection_A_vbase/crg_1x/seed0（里面有 train/weights/best.pt）
  --tta    每个模型用 Ultralytics 的 augment=True（左右翻转 + 多尺度）推理
  多个目录：每张图把各模型的预测用加权框融合（WBF）合成一组框
结果：results/experiments/ensemble/<名字>.json，含 select 与 test 两组指标；匹配与 AP 计算与 Ultralytics 验证一致
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, box_iou

from common import CLASSES, DET, RES

SPLIT = DET.parent / "detection_A"
IOUV = np.linspace(0.5, 0.95, 10)
SIZE, IMGSZ = 256, 640


def match(pred_cls, true_cls, iou):
    """与 BaseValidator.match_predictions（use_scipy=False）相同；预测按置信度降序"""
    correct = np.zeros((len(pred_cls), len(IOUV)), bool)
    if len(true_cls) == 0 or len(pred_cls) == 0:
        return correct
    iou = iou * (true_cls[:, None] == pred_cls[None, :])
    matched = np.zeros((iou.shape[0], len(IOUV)), bool)
    for j in np.flatnonzero((iou >= IOUV.min()).any(0)):
        available = np.where(matched, 0, iou[:, j, None])
        k = available.argmax(0)
        correct[j] = available[k, range(len(IOUV))] >= IOUV
        matched[k, range(len(IOUV))] |= correct[j]
    return correct


def wbf(preds, iou_thr=0.55):
    """加权框融合：preds 为 [(boxes N×4, scores N, cls N)]，每个模型一组；同类且 IoU > 阈值的框按置信度加权平均"""
    n_models = len(preds)
    allb = [(b, s, c, m) for m, (bb, ss, cc) in enumerate(preds) for b, s, c in zip(bb, ss, cc)]
    allb.sort(key=lambda x: -x[1])
    clusters = []                                                     # 每个簇：[框列表, 置信度列表, 类别]
    for b, s, c, _ in allb:
        best, best_iou = None, iou_thr
        for cl in clusters:
            if cl[2] != c:
                continue
            fused = np.average(np.array(cl[0]), axis=0, weights=np.array(cl[1]))
            x0, y0 = max(fused[0], b[0]), max(fused[1], b[1]); x1, y1 = min(fused[2], b[2]), min(fused[3], b[3])
            inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            u = (fused[2] - fused[0]) * (fused[3] - fused[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            v = inter / u if u > 0 else 0
            if v > best_iou:
                best, best_iou = cl, v
        if best is None:
            clusters.append([[b], [s], c])
        else:
            best[0].append(b); best[1].append(s)
    boxes, scores, cls = [], [], []
    for bl, sl, c in clusters:
        boxes.append(np.average(np.array(bl), axis=0, weights=np.array(sl)))
        scores.append(float(np.mean(sl)) * min(len(sl), n_models) / n_models)   # 只有少数模型给出的框降低置信度
        cls.append(c)
    return np.array(boxes).reshape(-1, 4), np.array(scores), np.array(cls, float)


def predict(models, img, tta):
    out = []
    for m in models:
        r = m.predict(str(img), imgsz=IMGSZ, conf=0.001, iou=0.7, max_det=300, augment=tta, verbose=False)[0].boxes
        out.append((r.xyxy.cpu().numpy(), r.conf.cpu().numpy(), r.cls.cpu().numpy()))
    if len(out) == 1:
        return out[0]
    return wbf(out)


def score(models, split, tta):
    stats = []
    for img in sorted((SPLIT / split / "images").glob("*.png")):
        pb, pconf, pcls = predict(models, img, tta)
        order = np.argsort(-pconf); pb, pconf, pcls = pb[order], pconf[order], pcls[order]
        lines = [l.split() for l in (SPLIT / split / "labels" / f"{img.stem}.txt").read_text().splitlines() if l.strip()]
        tcls = np.array([int(l[0]) for l in lines], float)
        tb = np.array([[(float(cx) - float(w) / 2) * SIZE, (float(cy) - float(h) / 2) * SIZE,
                        (float(cx) + float(w) / 2) * SIZE, (float(cy) + float(h) / 2) * SIZE] for _, cx, cy, w, h in lines]).reshape(-1, 4)
        iou = (box_iou(torch.tensor(tb, dtype=torch.float32), torch.tensor(pb, dtype=torch.float32)).numpy()
               if len(tb) and len(pb) else np.zeros((len(tb), len(pb))))
        stats.append((match(pcls, tcls, iou), pconf, pcls, tcls))
    tp, conf, pc, tc = (np.concatenate([s[k] for s in stats]) for k in range(4))
    out = ap_per_class(tp, conf, pc, tc)
    p, r, ap, cls = out[2], out[3], out[5], out[6]
    return {"map50_95": float(ap.mean()), "map50": float(ap[:, 0].mean()), "precision": float(p.mean()), "recall": float(r.mean()),
            "per_class": {CLASSES[int(c)]: float(ap[k].mean()) for k, c in enumerate(cls)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()
    models = [YOLO(str(Path(r) / "train" / "weights" / "best.pt")) for r in args.runs]
    res = {"name": args.name, "runs": args.runs, "tta": args.tta,
           "select": score(models, "select", args.tta), "test": score(models, "test", args.tta)}
    out = RES / "ensemble"; out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / f"{args.name}.json", "w"), indent=1)
    print(json.dumps({k: (v if k in ("name", "tta") else {m: round(x, 4) for m, x in v.items() if m != "per_class"})
                      for k, v in res.items() if k != "runs"}), flush=True)


if __name__ == "__main__":
    main()
