"""缺陷局部块：超大缺陷不二次缩放，保持 0.5× 取覆盖框最多的 64×64 窗口（2026-11-24）
python build_defect_maxcov.py
- 只保留划分 A 训练照片的块（defect_manifest.csv 中 922 个块来自划分 A 测试照片）
- 未超大的块直接复制 defect_64/ 的原文件；超大块（effective_scale < 0.5）以框中心为窗口中心重新裁剪
- 覆盖率（窗口内的框面积 / 框面积）< MIN_COV 的丢弃
输出：data/pcb_processed/defect_64_maxcov_trainA/、defect_maxcov_manifest.csv、results/experiments/defect_maxcov/sheet.jpg
"""
import csv
import shutil

import numpy as np
from PIL import Image, ImageDraw

from common import CLASSES, PATCH, PKU, PROC, RES, ROOT, SCALE, photo

MIN_COV = 0.8
SRC, DST = PROC / "defect_64", PROC / "defect_64_maxcov_trainA"
OUT = RES / "defect_maxcov"


def boxes_s(stem):
    """0.5× 像素坐标的框 [(类别 id, x0, y0, x1, y1)]"""
    W, H = Image.open(PKU / "images" / f"{stem}.jpg").size
    out = []
    for line in (PKU / "labels" / f"{stem}.txt").read_text().splitlines():
        c, cx, cy, w, h = line.split()
        cx, cy, w, h = float(cx) * W * SCALE, float(cy) * H * SCALE, float(w) * W * SCALE, float(h) * H * SCALE
        out.append((int(c), cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def find_box(r):
    """旧块对应的框：该类、中心在旧窗口内、长边 > 64 的框中离窗口中心最近的一个"""
    l, t, S = int(r["left"]), int(r["top"]), int(r["window"])
    c = CLASSES.index(r["class"])
    cand = [b for b in boxes_s(r["source"]) if b[0] == c and max(b[3] - b[1], b[4] - b[2]) > PATCH
            and l <= (b[1] + b[3]) / 2 <= l + S and t <= (b[2] + b[4]) / 2 <= t + S]
    assert cand, r["file"]
    return min(cand, key=lambda b: ((b[1] + b[3]) / 2 - l - S / 2) ** 2 + ((b[2] + b[4]) / 2 - t - S / 2) ** 2)


def maxcov_crop(img, b):
    """以框中心为中心的 64 窗口：框较长的一边被截断时，任何覆盖框内的位置覆盖率相同，取居中；返回 (块, left, top, 覆盖率)"""
    _, x0, y0, x1, y1 = b
    W, H = img.size
    l = int(round(min(max((x0 + x1) / 2 - PATCH / 2, 0), W - PATCH)))
    t = int(round(min(max((y0 + y1) / 2 - PATCH / 2, 0), H - PATCH)))
    inter = max(0, min(x1, l + PATCH) - max(x0, l)) * max(0, min(y1, t + PATCH) - max(y0, t))
    return img.crop((l, t, l + PATCH, t + PATCH)), l, t, inter / ((x1 - x0) * (y1 - y0))


def main():
    split = {r["photo"]: r["split"] for r in csv.DictReader(open(ROOT / "data" / "detection_A" / "photo_split.csv"))}
    rows = list(csv.DictReader(open(PROC / "defect_manifest.csv")))
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    OUT.mkdir(parents=True, exist_ok=True)

    # 检验：未超大块重新裁剪应与原文件只差 JPEG 压缩
    plain = [r for r in rows if float(r["effective_scale"]) == SCALE][:20]
    err = [np.abs(np.asarray(photo(r["source"]).crop((int(r["left"]), int(r["top"]), int(r["left"]) + PATCH, int(r["top"]) + PATCH)),
                             np.int16) - np.asarray(Image.open(SRC / r["file"]).convert("RGB"), np.int16)).mean() for r in plain]
    print(f"recrop check on {len(plain)} plain patches: mean abs diff {np.mean(err):.2f} (max {np.max(err):.2f}) grey levels")
    assert np.max(err) < 3

    man, tiles = [], []
    for r in rows:
        big = float(r["effective_scale"]) < SCALE
        keep_split = split[r["source"]] == "train"
        if not big:
            if keep_split:
                shutil.copy(SRC / r["file"], DST / r["file"])
                man.append([r["file"], r["class"], r["source"], r["left"], r["top"], "copy", 1.0])
            continue
        b = find_box(r)
        img = photo(r["source"])
        new, l, t, cov = maxcov_crop(img, b)
        status = "drop_cov" if cov < MIN_COV else ("keep" if keep_split else "not_trainA")
        if status == "keep":
            new.save(DST / r["file"], quality=95)
            man.append([r["file"], r["class"], r["source"], l, t, "maxcov", round(cov, 3)])
        # 检查图：旧块（二次缩放）| 新块，框画在新块上
        old = Image.open(SRC / r["file"]).convert("RGB")
        nb = new.copy()
        ImageDraw.Draw(nb).rectangle([b[1] - l, b[2] - t, b[3] - l, b[4] - t], outline=(230, 30, 30))
        tiles.append((r["class"], cov, status, old, nb))

    with open(PROC / "defect_maxcov_manifest.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "class", "source", "left", "top", "how", "box_coverage"]); w.writerows(man)

    # 每格：旧块、新块并排放大 2 倍；底色条：绿 = 进入训练集，灰 = 非训练照片，红 = 覆盖率不足被丢弃
    tiles.sort(key=lambda x: (x[0], x[1]))
    Z, cols = 2 * PATCH, 6
    cw, ch = 2 * Z + 6, Z + 8
    rows_n = -(-len(tiles) // cols)
    sheet = Image.new("RGB", (cols * cw + (cols - 1) * 10, rows_n * ch + (rows_n - 1) * 6), (255, 255, 255))
    col = {"keep": (40, 170, 70), "not_trainA": (170, 170, 170), "drop_cov": (220, 40, 40)}
    for k, (_, cov, st, old, nb) in enumerate(tiles):
        x, y = (k % cols) * (cw + 10), (k // cols) * (ch + 6)
        sheet.paste(old.resize((Z, Z), Image.NEAREST), (x, y))
        sheet.paste(nb.resize((Z, Z), Image.NEAREST), (x + Z + 6, y))
        ImageDraw.Draw(sheet).rectangle([x, y + Z + 2, x + cw - 1, y + Z + 6], fill=col[st])
    sheet.save(OUT / "sheet.jpg", quality=92)

    from collections import Counter
    print("oversize:", len(tiles), dict(Counter(t[2] for t in tiles)))
    print("dropped (coverage < %.1f):" % MIN_COV, [(t[0], round(t[1], 2)) for t in tiles if t[2] == "drop_cov"])
    print("train set:", len(man), dict(Counter(m[1] for m in man)), "· maxcov crops:", sum(m[5] == "maxcov" for m in man))


if __name__ == "__main__":
    main()
