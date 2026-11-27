"""检查 C1（2026-11-20）：合成样本的标签框怎么画（论文未说明）。图像与 crg_A 完全相同（符号链接），只换标签：
  crgwbox_A  标签框 = 整个 64×64 生成窗口
  crgcbox_A  标签框 = 生成图与背景的实际差异区域的外接框（check_boxes.change_box；找不到差异时保留原标签）
原 crg_A 的标签框 = 放置位置真实缺陷的大小、以窗口中心为中心。box_check 显示 CRG 实际改动的外接框
只有原标签框面积的约 14%（中位数），所以原标签可能偏大。
"""
import csv

import numpy as np
from PIL import Image

from check_boxes import change_box
from common import FREE, PATCH, SYN, window_of

SRC, JOBS = SYN / "crg_A", SYN / "jobs_A"


def line(c, x0, y0, x1, y1):
    return f"{c} {(x0 + x1) / 2 / FREE:.6f} {(y0 + y1) / 2 / FREE:.6f} {(x1 - x0) / FREE:.6f} {(y1 - y0) / FREE:.6f}\n"


def main():
    outs = {k: SYN / f"{k}_A" for k in ("crgwbox", "crgcbox")}
    for d in outs.values():
        (d / "images").mkdir(parents=True, exist_ok=True); (d / "labels").mkdir(parents=True, exist_ok=True)
    found, ratio, rows = 0, [], []
    jids = [r["jid"] for r in csv.DictReader(open(JOBS / "jobs.csv"))][:300]
    for jid in jids:
        c, cx, cy, w, h = (SRC / "labels" / f"{jid}.txt").read_text().split()
        jx, jy = [float(v) * FREE for v in (JOBS / f"{jid}.txt").read_text().split()[1:3]]
        lx, ly = window_of(int(jx), int(jy))
        gen = np.asarray(Image.open(SRC / "images" / f"{jid}.png").convert("RGB"))
        bg = np.asarray(Image.open(JOBS / f"{jid}.png").convert("RGB"))
        cb, _, _ = change_box(gen, bg, lx, ly)
        orig = (float(cx) - float(w) / 2) * FREE, (float(cy) - float(h) / 2) * FREE, (float(cx) + float(w) / 2) * FREE, (float(cy) + float(h) / 2) * FREE
        if cb is not None:
            found += 1; ratio.append((cb[2] - cb[0]) * (cb[3] - cb[1]) / ((orig[2] - orig[0]) * (orig[3] - orig[1])))
        labels = {"crgwbox": line(c, lx, ly, lx + PATCH, ly + PATCH),
                  "crgcbox": line(c, *cb) if cb is not None else line(c, *orig)}
        for k, d in outs.items():
            img = d / "images" / f"{jid}.png"
            if not img.exists():
                img.symlink_to((SRC / "images" / f"{jid}.png").resolve())
            (d / "labels" / f"{jid}.txt").write_text(labels[k])
        rows.append([jid, c, cb is not None] + ([round(v, 1) for v in cb] if cb is not None else ["", "", "", ""]))
    with open(outs["crgcbox"] / "change_boxes.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["jid", "class", "found", "x0", "y0", "x1", "y1"]); wr.writerows(rows)
    print(f"change box found for {found}/{len(jids)} images; area vs original label: median {np.median(ratio):.2f}, "
          f"quartiles {np.percentile(ratio, 25):.2f}–{np.percentile(ratio, 75):.2f}", flush=True)


if __name__ == "__main__":
    main()
