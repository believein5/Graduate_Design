"""泄漏诊断（只用于解释论文结果，不计入任务书指标）：用划分 A 测试图像自身的背景与缺陷位置构造生成任务（2026-11-24）
python build_leak_jobs.py
- 背景：每张测试裁剪在去缺陷照片（标注框外扩 8 px 用板模板覆盖，与 free_256 的做法相同）中的同一位置
- 类别与位置：该测试裁剪所选缺陷（meta.csv 的类别，标签中第一个该类框）；标签框 = 该真实框
输出 data/synthetic/jobs_Aleak/（jid.png、jid.txt、jobs.csv，格式同 make_jobs.py），之后
generate.py --method crg --jobs jobs_Aleak --tag leak_A 生成到 data/synthetic/crgleak_A/，检测配置名 crgleak_1x / crgleak_synonly
板 10～12 没有模板：按预处理的方法（该板全部照片逐像素中值）补建到 data/pcb_processed/templates/
"""
import csv
import shutil

import numpy as np
from PIL import Image

from common import CLASSES, FREE, PKU, PROC, ROOT, SYN, TRAIN, VAL, board, boxes, photo

MARGIN = 8
DET_A = ROOT / "data" / "detection_A"
OUT = SYN / "jobs_Aleak"


def template(b):
    p = PROC / "templates" / f"board_{b}.png"
    if p.exists():
        return np.asarray(Image.open(p).convert("RGB"))
    stack = np.stack([np.asarray(photo(s)) for s in TRAIN + VAL if board(s) == b])
    tpl = np.empty(stack.shape[1:], np.uint8)
    for r in range(0, stack.shape[1], 64):                       # 分块求中值，控制内存
        tpl[r:r + 64] = np.median(stack[:, r:r + 64], axis=0)
    Image.fromarray(tpl).save(p)
    print(f"built template board {b} from {len(stack)} photos")
    return tpl


def clean(stem):
    a = np.asarray(photo(stem)).copy()
    t = template(board(stem))
    H, W = a.shape[:2]
    mask = np.zeros((H, W), bool)
    for _, x0, y0, x1, y1 in boxes(stem):
        l, tp = max(0, int(x0 - MARGIN)), max(0, int(y0 - MARGIN))
        r, bt = min(W, int(x1 + MARGIN) + 1), min(H, int(y1 + MARGIN) + 1)
        a[tp:bt, l:r] = t[tp:bt, l:r]; mask[tp:bt, l:r] = True
    # 对齐检查：框外像素与模板的平均灰度差（训练照片中位数 0.20）
    diff = np.abs(np.asarray(photo(stem), np.int16) - t.astype(np.int16)).mean(2)[~mask].mean()
    return a, diff


def main():
    meta = [r for r in csv.DictReader(open(DET_A / "meta.csv")) if r["split"] == "test"]
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rows, diffs = [], []
    for i, m in enumerate(meta):
        jid, c = f"j{i:04d}", CLASSES.index(m["class"])
        l, t = int(m["left"]), int(m["top"])
        a, d = clean(m["source"]); diffs.append(d)
        Image.fromarray(a[t:t + FREE, l:l + FREE]).save(OUT / f"{jid}.png")
        lines = [list(map(float, x.split())) for x in (DET_A / "test" / "labels" / f"{m['name']}.txt").read_text().splitlines()]
        _, cx, cy, w, h = next(x for x in lines if int(x[0]) == c)
        bx = ((cx - w / 2) * FREE, (cy - h / 2) * FREE, (cx + w / 2) * FREE, (cy + h / 2) * FREE)
        (OUT / f"{jid}.txt").write_text(f"{c} {cx:.6f} {cy:.6f}\n")
        rows.append([jid, m["class"], board(m["source"]), m["name"], l, t, m["source"], *(round(v, 2) for v in bx), "", "", "", "", ""])
    with open(OUT / "jobs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["jid", "class", "board", "bg_file", "bg_left", "bg_top", "place_photo", "bx0", "by0", "bx1", "by1",
                    "src_photo", "sx0", "sy0", "sx1", "sy1"])
        w.writerows(rows)
    diffs = np.array(diffs)
    print(f"{len(rows)} jobs · photo-template diff outside boxes: median {np.median(diffs):.2f}, max {diffs.max():.2f}, "
          f"> 3: {(diffs > 3).sum()}")


if __name__ == "__main__":
    main()
