"""300 张测试图够不够（2026-11-20）：用保留了权重的 Original 检测器（划分 A，640，种子 0），
对测试集按图像有放回重抽样 2000 次，看 mAP@50–95、召回率与各类 AP 只因"抽到哪 300 张测试图"而波动多少。
匹配规则与 Ultralytics 验证相同（按置信度贪心匹配，10 个 IoU 阈值），AP 用 ultralytics.utils.metrics.ap_per_class。
结果：results/experiments/bootstrap_test.json
"""
import json

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, box_iou

from common import CLASSES, DET, RES

WEIGHTS = RES / "detection_diag_img640" / "original" / "seed0" / "train" / "weights" / "best.pt"
TEST = DET.parent / "detection_A" / "test"
IOUV = np.linspace(0.5, 0.95, 10)
N_BOOT, SIZE = 2000, 256


def match(pred_cls, true_cls, iou):
    """与 BaseValidator.match_predictions（use_scipy=False）相同；pred 已按置信度降序"""
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


def metrics(per_img, idx):
    tp = np.concatenate([per_img[i][0] for i in idx]); conf = np.concatenate([per_img[i][1] for i in idx])
    pc = np.concatenate([per_img[i][2] for i in idx]); tc = np.concatenate([per_img[i][3] for i in idx])
    out = ap_per_class(tp, conf, pc, tc)
    p, r, ap, cls = out[2], out[3], out[5], out[6]
    per_cls = {CLASSES[int(c)]: float(ap[k].mean()) for k, c in enumerate(cls)}
    return float(ap.mean()), float(ap[:, 0].mean()), float(r.mean()), float(p.mean()), per_cls


def main():
    model = YOLO(str(WEIGHTS))
    imgs = sorted((TEST / "images").glob("*.png"))
    per_img = []
    for f in imgs:
        res = model.predict(str(f), imgsz=640, conf=0.001, iou=0.7, max_det=300, verbose=False)[0]
        b = res.boxes
        order = np.argsort(-b.conf.cpu().numpy())
        pb, pconf, pcls = b.xyxy.cpu().numpy()[order], b.conf.cpu().numpy()[order], b.cls.cpu().numpy()[order]
        lines = [l.split() for l in (TEST / "labels" / f"{f.stem}.txt").read_text().splitlines() if l.strip()]
        tcls = np.array([int(l[0]) for l in lines], float)
        tb = np.array([[(float(cx) - float(w) / 2) * SIZE, (float(cy) - float(h) / 2) * SIZE,
                        (float(cx) + float(w) / 2) * SIZE, (float(cy) + float(h) / 2) * SIZE] for _, cx, cy, w, h in lines]).reshape(-1, 4)
        iou = box_iou(torch.tensor(tb, dtype=torch.float32), torch.tensor(pb, dtype=torch.float32)).numpy() if len(tb) and len(pb) else np.zeros((len(tb), len(pb)))
        per_img.append((match(pcls, tcls, iou), pconf, pcls, tcls))
    full = metrics(per_img, range(len(imgs)))
    rng = np.random.default_rng(0)
    boot = [metrics(per_img, rng.integers(0, len(imgs), len(imgs))) for _ in range(N_BOOT)]
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float(np.std(v))]
    res = {"n_images": len(imgs), "n_boxes": int(sum(len(x[3]) for x in per_img)),
           "full": {"map50_95": full[0], "map50": full[1], "recall": full[2], "precision": full[3]},
           "ci95_and_sd": {"map50_95": ci([b[0] for b in boot]), "map50": ci([b[1] for b in boot]), "recall": ci([b[2] for b in boot]),
                           "precision": ci([b[3] for b in boot])},
           "per_class_ap50_95": {c: {"value": full[4][c], "ci95_and_sd": ci([b[4].get(c, np.nan) for b in boot])} for c in full[4]}}
    json.dump(res, open(RES / "bootstrap_test.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
