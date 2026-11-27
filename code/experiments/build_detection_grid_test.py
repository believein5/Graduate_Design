"""检查 B2（2026-11-20）：按 256 网格切出的测试集，模仿论文图中线索（作者的真实缺陷裁剪落在 0.5× 照片的 256 网格上，
缺陷可在裁剪内任意位置，见 results/experiments/paper_figs/clues.md 第 7 条）
data/detection_A_grid/：train、select 与划分 A 完全相同（复制），test 用划分 A 的同一批 300 张测试照片：
- 0.5× 照片按 256 网格切块（左上角为 256 的整数倍，放不下的右侧、底部余边不用）
- 每张照片随机取一个标注框，取它中心所在的网格块；该框在块内可见面积需 ≥ 50%（否则换一个框）
- 块内所有框按训练数据的同一规则标注（可见 ≥ 50%）
"""
import csv
import random
import shutil

from common import CLASSES, DET, FREE, boxes, photo, size_of, write_sample, yolo_lines

SRC = DET.parent / "detection_A"
DST = DET.parent / "detection_A_grid"
SEED = 11


def visible(box, left, top):
    _, x0, y0, x1, y1 = box
    return max(0.0, min(x1, left + FREE) - max(x0, left)) * max(0.0, min(y1, top + FREE) - max(y0, top)) / ((x1 - x0) * (y1 - y0))


def main():
    rng = random.Random(SEED)
    test = [r["photo"] for r in csv.DictReader(open(SRC / "photo_split.csv")) if r["split"] == "test"]
    if DST.exists():
        shutil.rmtree(DST)
    for split in ("train", "select"):
        shutil.copytree(SRC / split, DST / split, ignore=shutil.ignore_patterns("*.cache"))
    k, rows, fallback = {c: 0 for c in CLASSES}, [], 0
    for s in test:
        w, h = size_of(s)
        nx, ny = int(w // FREE), int(h // FREE)
        bx = boxes(s)[:]
        rng.shuffle(bx)
        pick = None
        for box in bx:
            cx, cy = (box[1] + box[3]) / 2, (box[2] + box[4]) / 2
            i, j = int(cx // FREE), int(cy // FREE)
            if i < nx and j < ny and visible(box, i * FREE, j * FREE) >= 0.5:
                pick = (box, i * FREE, j * FREE); break
        if pick is None:                                                    # 没有框的中心落在完整网格块里：取可见面积最大的块
            fallback += 1
            pick = max(((b, i * FREE, j * FREE) for b in bx for i in range(nx) for j in range(ny)), key=lambda p: visible(*p))
        box, left, top = pick
        c = box[0]
        lines = yolo_lines(boxes(s), left, top)
        cut = sum(0 < visible(b, left, top) < 1 for b in boxes(s))
        name = f"test_{CLASSES[c]}_{k[CLASSES[c]]:03d}"; k[CLASSES[c]] += 1
        write_sample(photo(s).crop((left, top, left + FREE, top + FREE)), lines,
                     DST / "test" / "images" / f"{name}.png", DST / "test" / "labels" / f"{name}.txt")
        rows.append([name, CLASSES[c], s, left, top, len(lines), cut, round(visible(box, left, top), 3)])
    with open(DST / "test_meta.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["name", "class", "photo", "left", "top", "n_boxes", "n_cut", "main_visible"]); wr.writerows(rows)
    shutil.copy(SRC / "photo_split.csv", DST / "photo_split.csv")
    n = len(rows)
    print(f"grid test: {n} images · {sum(r[5] for r in rows)} labelled boxes · main defect cut by the edge in "
          f"{sum(r[7] < 1 for r in rows)} ({sum(r[7] < 1 for r in rows) / n:.0%}) · images with any cut box {sum(r[6] > 0 for r in rows)} · "
          f"fallback tiles {fallback} · per class {k}", flush=True)


if __name__ == "__main__":
    main()
