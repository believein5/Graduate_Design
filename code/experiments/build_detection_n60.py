"""划分 A 的每类 60 张训练集（2026-11-25，数据量趋势实验）：data/detection_A_n60/
= detection_A/train 的全部 350 张（每类 50 张缺陷 + 50 张无缺陷），再加每类 10 张新的缺陷裁剪和 10 张新的无缺陷背景块，
所以 10 ⊂ 20 ⊂ 30 ⊂ 40 ⊂ 50 ⊂ 60；select / test 与划分 A 相同。
- 新的缺陷裁剪只取划分 A 训练照片中尚未出现在任何已有训练裁剪里的标注框，裁剪规则与 build_detection_data.defect_crops 相同
- 新的无缺陷背景块只取来源为划分 A 训练照片、且未被已有训练集使用的 free_256 背景块
  （已有的 50 张无缺陷图中有 25 张来自测试照片、8 张来自选择照片，见 2026-11-25 记录；这里不再这样取）
python build_detection_n60.py
"""
import csv
import random
import shutil

from PIL import Image

from common import CLASSES, FREE, PROC, ROOT, boxes, photo, size_of, write_sample, yolo_lines

SRC, DST = ROOT / "data" / "detection_A", ROOT / "data" / "detection_A_n60"
EXTRA, MARGIN, SEED = 10, 8, 60


def inside(b, left, top):
    _, x0, y0, x1, y1 = b
    return x0 >= left and y0 >= top and x1 <= left + FREE and y1 <= top + FREE


def main():
    split = {r["photo"]: r["split"] for r in csv.DictReader(open(SRC / "photo_split.csv"))}
    meta = list(csv.DictReader(open(SRC / "meta.csv")))
    train_meta = [r for r in meta if r["split"] == "train"]
    used = {}                                                  # 照片 -> 已有训练裁剪窗口
    for r in train_meta:
        if r["class"] != "free":
            used.setdefault(r["source"], []).append((int(r["left"]), int(r["top"])))
    rng = random.Random(SEED)

    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC / "train", DST / "train", ignore=shutil.ignore_patterns("*.cache"))
    for s in ("select", "test"):
        shutil.copytree(SRC / s, DST / s, ignore=shutil.ignore_patterns("*.cache"))
    new = []
    photos = sorted(p for p, v in split.items() if v == "train")
    for ci, c in enumerate(CLASSES):
        cand = [(s, k) for s in photos for k, b in enumerate(boxes(s))
                if b[0] == ci and not any(inside(b, l, t) for l, t in used.get(s, []))]
        for j, (s, k) in enumerate(rng.sample(cand, EXTRA)):
            w, h = size_of(s)
            _, x0, y0, x1, y1 = boxes(s)[k]
            left = int(min(max(rng.uniform(x1 + MARGIN - FREE, x0 - MARGIN), 0), w - FREE))
            top = int(min(max(rng.uniform(y1 + MARGIN - FREE, y0 - MARGIN), 0), h - FREE))
            lines = yolo_lines(boxes(s), left, top)
            name = f"train_{c}_{50 + j:03d}"
            write_sample(photo(s).crop((left, top, left + FREE, top + FREE)), lines,
                         DST / "train" / "images" / f"{name}.png", DST / "train" / "labels" / f"{name}.txt")
            new.append(["train", name, c, s, left, top, len(lines)])
        print(f"{c}: {len(cand)} unused boxes in train photos, added {EXTRA}")
    used_free = {(r["source"], r["left"], r["top"]) for r in train_meta if r["class"] == "free"}
    free = [r for r in csv.DictReader(open(PROC / "free_manifest.csv"))
            if split[r["source"]] == "train" and (r["source"], r["left"], r["top"]) not in used_free]
    for j, r in enumerate(rng.sample(free, EXTRA)):
        name = f"train_free_{50 + j:03d}"
        write_sample(Image.open(PROC / "free_256" / r["file"]).convert("RGB"), [],
                     DST / "train" / "images" / f"{name}.png", DST / "train" / "labels" / f"{name}.txt")
        new.append(["train", name, "free", r["source"], r["left"], r["top"], 0])
    with open(DST / "meta_added.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["split", "name", "class", "source", "left", "top", "n_boxes"]); w.writerows(new)
    assert all(split[r[3]] == "train" for r in new)
    n = len(list((DST / "train" / "images").glob("*.png")))
    print(f"{DST.name}: train {n} images ({n - 350} added, all from split-A train photos)")


if __name__ == "__main__":
    main()
