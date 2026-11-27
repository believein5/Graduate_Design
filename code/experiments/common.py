"""实验共用的路径、数据读取与工具函数（复现 CRG 论文的下游检测实验）"""
import csv
import functools
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]                    # Graduate_Design/
REPO = ROOT / "third_party" / "CRG-DefectDiffuser"
PKU = ROOT / "data" / "PKU-Market-PCB-corrected"
PROC = ROOT / "data" / "pcb_processed"
DET = ROOT / "data" / "detection"                             # 检测用真实数据（train / select / test）
SYN = ROOT / "data" / "synthetic"                             # 各方法生成的合成数据
RES = ROOT / "results" / "experiments"
CKPT_64 = ROOT / "checkpoint" / "checkpoint-64-defect.pt"     # 作者的 DPM
CKPT_256 = ROOT / "checkpoint" / "checkpoint-256-defect-free.pt"  # 作者的 CSM

SCALE, FREE, PATCH = 0.5, 256, 64
CLASSES = ["missinghole", "mousebite", "opencircuit", "short", "spur", "spuriouscopper"]   # 与 pcb.json 的 id 一致
N_PER_SPLIT_CLASS = 50


def stems(split_file):
    return [Path(s).stem for s in (PKU / "ImageSets" / split_file).read_text().split()]


TRAIN = stems("train_board.txt")
VAL = stems("val_board.txt")
SELECT, TEST = VAL[0::2], VAL[1::2]                           # 与 train_walkthrough.ipynb 相同的两半


def board(stem):
    return stem[:2]


@functools.lru_cache(maxsize=32)
def photo(stem):
    """0.5× 照片（与预处理相同：双三次缩放）"""
    im = Image.open(PKU / "images" / f"{stem}.jpg").convert("RGB")
    return im.resize((round(im.width * SCALE), round(im.height * SCALE)), Image.BICUBIC)


def size_of(stem):
    with Image.open(PKU / "images" / f"{stem}.jpg") as im:
        return round(im.width * SCALE), round(im.height * SCALE)


def boxes(stem):
    """0.5× 图像上的标注框：[(类别 id, x0, y0, x1, y1)]"""
    w, h = size_of(stem)
    out = []
    for line in (PKU / "labels" / f"{stem}.txt").read_text().splitlines():
        c, cx, cy, bw, bh = map(float, line.split())
        out.append((int(c), (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def misaligned():
    """与板模板错位 ≥ 0.5 px 的照片（预处理第 8 步算出）；它们的标注位置不能映射到模板"""
    f = PROC / "photo_shift.csv"
    return {r["photo"] for r in csv.DictReader(open(f)) if float(r["shift_px"]) >= 0.5}


def yolo_lines(bxs, left, top, size=FREE, min_visible=0.5):
    """把落在裁剪窗口内的框转成 YOLO 行；被边界截断的框保留可见面积 ≥ min_visible 的部分"""
    lines = []
    for c, x0, y0, x1, y1 in bxs:
        a0, b0 = max(x0 - left, 0), max(y0 - top, 0)
        a1, b1 = min(x1 - left, size), min(y1 - top, size)
        if a1 <= a0 or b1 <= b0:
            continue
        if (a1 - a0) * (b1 - b0) < min_visible * (x1 - x0) * (y1 - y0):
            continue
        lines.append(f"{c} {(a0 + a1) / 2 / size:.6f} {(b0 + b1) / 2 / size:.6f} {(a1 - a0) / size:.6f} {(b1 - b0) / size:.6f}")
    return lines


def write_sample(img, lines, img_path, lbl_path):
    img_path.parent.mkdir(parents=True, exist_ok=True)
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(img_path)
    lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def window_of(cx, cy, size=FREE, patch=PATCH):
    """与 sample_generation.py 相同的 64 窗口位置（以缺陷中心为中心，夹在图像内）"""
    lx = int(max(0, min(cx - patch // 2, size - patch)))
    ly = int(max(0, min(cy - patch // 2, size - patch)))
    return lx, ly
