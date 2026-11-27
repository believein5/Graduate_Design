"""检查 B（2026-11-20）：更难的测试集——缺陷可以出现在裁剪的任意位置，包括被裁剪边缘截断
data/detection_A_edge/：train、select 与划分 A 完全相同（复制），test 用划分 A 的同一批 300 张测试照片重新裁剪：
- 每张照片随机取一个标注框，裁剪位置在"框中心落在窗口内"的范围内均匀抽取（划分 A 要求整个框在窗口内且离边缘 ≥ 8 px）
- 该框可见面积需 ≥ 50%（否则重抽）；窗口内其他框按同一规则（可见 ≥ 50%）标注，与训练数据的规则一致
用于检查：论文的 Original 召回率（0.869）低于我们（0.936），是否因为它的测试裁剪里有更多被截断的缺陷
"""
import csv
import random
import shutil

from common import CLASSES, DET, FREE, boxes, photo, size_of, write_sample, yolo_lines

SRC = DET.parent / "detection_A"
DST = DET.parent / "detection_A_edge"
SEED = 7


def visible(box, left, top):
    c, x0, y0, x1, y1 = box
    a = max(0.0, min(x1, left + FREE) - max(x0, left)) * max(0.0, min(y1, top + FREE) - max(y0, top))
    return a / ((x1 - x0) * (y1 - y0))


def main():
    rng = random.Random(SEED)
    test = [r["photo"] for r in csv.DictReader(open(SRC / "photo_split.csv")) if r["split"] == "test"]
    if DST.exists():
        shutil.rmtree(DST)
    for split in ("train", "select"):
        shutil.copytree(SRC / split, DST / split, ignore=shutil.ignore_patterns("*.cache"))
    k, rows = {c: 0 for c in CLASSES}, []
    for s in test:
        w, h = size_of(s)
        box = rng.choice(boxes(s))
        c, x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        for _ in range(1000):
            left = int(min(max(rng.uniform(cx - FREE + 1, cx), 0), w - FREE))
            top = int(min(max(rng.uniform(cy - FREE + 1, cy), 0), h - FREE))
            if visible(box, left, top) >= 0.5:
                break
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
    print(f"edge test: {n} images · {sum(r[5] for r in rows)} labelled boxes · main defect cut by the edge in "
          f"{sum(r[7] < 1 for r in rows)} images ({sum(r[7] < 1 for r in rows) / n:.0%}) · images with any cut box "
          f"{sum(r[6] > 0 for r in rows)}", flush=True)


if __name__ == "__main__":
    main()
