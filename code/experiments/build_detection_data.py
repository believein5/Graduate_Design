"""P0：建立检测用真实数据 data/detection/{train,select,test}/{images,labels}

train : 训练板（01, 04–09）上 300 张含缺陷的 256×256 裁剪（每类 50）+ 50 张无缺陷裁剪（负样本）
select: 留出板 10–12 的选择半区 300 张（每类 50），只用于调参
test  : 留出板 10–12 的测试半区 300 张（每类 50），所有报告指标都在这里算
全部在 0.5× 照片上裁剪；窗口内所有标注框都写进标签（被截断的框保留可见 ≥ 50% 的）。
"""
import csv
import random
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image

from common import (CLASSES, DET, FREE, N_PER_SPLIT_CLASS, PROC, SELECT, TEST, TRAIN, boxes, photo, size_of,
                    write_sample, yolo_lines)

SEED = 0
MARGIN = 8          # 目标框距裁剪边缘至少 8 px


def defect_crops(split, photos, rng):
    """每类随机取 50 个不同的标注框，各裁一张包含它的 256 窗口"""
    by_class = {c: [] for c in range(len(CLASSES))}
    for s in photos:
        for k, b in enumerate(boxes(s)):
            by_class[b[0]].append((s, k))
    rows = []
    for c, items in by_class.items():
        pick = rng.sample(items, N_PER_SPLIT_CLASS)
        for j, (s, k) in enumerate(pick):
            w, h = size_of(s)
            _, x0, y0, x1, y1 = boxes(s)[k]
            # 窗口左上角的可行范围：框完全在窗口内且离边缘 ≥ MARGIN，再夹在图像内
            lo_x, hi_x = x1 + MARGIN - FREE, x0 - MARGIN
            lo_y, hi_y = y1 + MARGIN - FREE, y0 - MARGIN
            left = int(min(max(rng.uniform(lo_x, hi_x), 0), w - FREE))
            top = int(min(max(rng.uniform(lo_y, hi_y), 0), h - FREE))
            img = photo(s).crop((left, top, left + FREE, top + FREE))
            lines = yolo_lines(boxes(s), left, top)
            name = f"{split}_{CLASSES[c]}_{j:03d}"
            write_sample(img, lines, DET / split / "images" / f"{name}.png", DET / split / "labels" / f"{name}.txt")
            rows.append([name, CLASSES[c], s, left, top, len(lines)])
    return rows


def main():
    if DET.exists():
        shutil.rmtree(DET)
    rng = random.Random(SEED)
    meta = []
    for split, photos in [("train", TRAIN), ("select", SELECT), ("test", TEST)]:
        meta += [[split] + r for r in defect_crops(split, photos, rng)]
    # 训练集的 50 张无缺陷负样本：从已校验过的 free_256 中随机取
    free = list(csv.DictReader(open(PROC / "free_manifest.csv")))
    for j, r in enumerate(rng.sample(free, 50)):
        name = f"train_free_{j:03d}"
        img = Image.open(PROC / "free_256" / r["file"]).convert("RGB")
        write_sample(img, [], DET / "train" / "images" / f"{name}.png", DET / "train" / "labels" / f"{name}.txt")
        meta.append(["train", name, "free", r["source"], r["left"], r["top"], 0])
    with open(DET / "meta.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["split", "name", "class", "source", "left", "top", "n_boxes"]); w.writerows(meta)
    for split in ["train", "select", "test"]:
        n = [m for m in meta if m[0] == split]
        print(f"{split:6s} {len(n)} images · {sum(m[6] for m in n)} boxes · photos {len({m[3] for m in n})}")

    # 预览：每个划分每类 2 张，画出标签框
    fig, axes = plt.subplots(3, 12, figsize=(24, 6.6))
    for r, split in enumerate(["train", "select", "test"]):
        for k, (c, j) in enumerate([(c, j) for c in CLASSES for j in (0, 1)]):
            name = f"{split}_{c}_{j:03d}"; ax = axes[r, k]
            ax.imshow(Image.open(DET / split / "images" / f"{name}.png"))
            for line in (DET / split / "labels" / f"{name}.txt").read_text().splitlines():
                _, cx, cy, bw, bh = map(float, line.split())
                ax.add_patch(mpatches.Rectangle(((cx - bw / 2) * FREE, (cy - bh / 2) * FREE), bw * FREE, bh * FREE,
                                                fill=False, ec="red", lw=1))
            ax.set_axis_off(); ax.set_title(f"{split} · {c}", fontsize=7)
    plt.tight_layout(); fig.savefig(DET / "preview.png", dpi=90)


if __name__ == "__main__":
    main()
