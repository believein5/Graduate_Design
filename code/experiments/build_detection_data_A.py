"""划分 A（论文式，按照片划分）：data/detection_A/{train,select,test}

与 build_detection_data.py 的区别只在照片的划分方式：
- 全部 10 块板的 693 张照片按照片随机分为 test / select / train 三组互不相交的照片
- test 300 张（每类 50，各类来自不同照片）、select 300 张、train 300 张缺陷 + 50 张无缺陷
- 测试照片不提供任何训练裁剪，但它们的板在训练中出现过（与论文的划分一致）
其余（0.5×、256 裁剪、框距边缘 ≥ 8 px、窗口内所有框都标注）与原来相同。
"""
import csv
import random
import shutil

from PIL import Image

import build_detection_data as B
from common import CLASSES, PROC, PKU, ROOT, boxes, write_sample

DET_A = ROOT / "data" / "detection_A"
SEED = 0
TEST_PHOTOS_PER_CLASS = 50      # 每类 50 张测试照片：每张照片只有一种缺陷，所以测试照片按类别抽
SELECT_PHOTOS_PER_CLASS = 25


def one_crop_per_photo(photos, rng):
    """每张照片随机取一个标注框，裁一张包含它的 256 窗口（规则同 B.defect_crops）"""
    from common import FREE, photo, size_of, yolo_lines
    rows, k = [], {c: 0 for c in CLASSES}
    for s in photos:
        w, h = size_of(s)
        c, x0, y0, x1, y1 = rng.choice(boxes(s))
        left = int(min(max(rng.uniform(x1 + B.MARGIN - FREE, x0 - B.MARGIN), 0), w - FREE))
        top = int(min(max(rng.uniform(y1 + B.MARGIN - FREE, y0 - B.MARGIN), 0), h - FREE))
        lines = yolo_lines(boxes(s), left, top)
        name = f"test_{CLASSES[c]}_{k[CLASSES[c]]:03d}"; k[CLASSES[c]] += 1
        write_sample(photo(s).crop((left, top, left + FREE, top + FREE)), lines,
                     DET_A / "test" / "images" / f"{name}.png", DET_A / "test" / "labels" / f"{name}.txt")
        rows.append([name, CLASSES[c], s, left, top, len(lines)])
    return rows


def main():
    rng = random.Random(SEED)
    photos = sorted(p.stem for p in (PKU / "images").glob("*.jpg"))
    by_cls = {c: [s for s in photos if boxes(s) and boxes(s)[0][0] == i] for i, c in enumerate(CLASSES)}
    test, select = [], []
    for c, ss in by_cls.items():
        ss = ss[:]; rng.shuffle(ss)
        test += ss[:TEST_PHOTOS_PER_CLASS]; select += ss[TEST_PHOTOS_PER_CLASS:TEST_PHOTOS_PER_CLASS + SELECT_PHOTOS_PER_CLASS]
    train = [s for s in photos if s not in set(test) | set(select)]
    print(f"photos: test {len(test)} · select {len(select)} · train {len(train)} · boards in test {sorted({s[:2] for s in test})}")

    if DET_A.exists():
        shutil.rmtree(DET_A)
    B.DET, B.MARGIN = DET_A, 8                                  # 复用原脚本的裁剪函数，只改输出目录
    import common
    common.DET = DET_A
    meta = []
    for split, ph in [("train", train), ("select", select)]:
        meta += [[split] + r for r in B.defect_crops(split, ph, rng)]
    meta += [["test"] + r for r in one_crop_per_photo(test, rng)]        # 测试：每张测试照片恰好一张裁剪（300 张不同的图）
    free = [r for r in csv.DictReader(open(PROC / "free_manifest.csv"))]
    for j, r in enumerate(rng.sample(free, 50)):
        name = f"train_free_{j:03d}"
        write_sample(Image.open(PROC / "free_256" / r["file"]).convert("RGB"), [],
                     DET_A / "train" / "images" / f"{name}.png", DET_A / "train" / "labels" / f"{name}.txt")
        meta.append(["train", name, "free", r["source"], r["left"], r["top"], 0])
    with open(DET_A / "meta.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["split", "name", "class", "source", "left", "top", "n_boxes"]); w.writerows(meta)
    with open(DET_A / "photo_split.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["photo", "split"])
        w.writerows([(s, "test") for s in test] + [(s, "select") for s in select] + [(s, "train") for s in train])
    for split in ["train", "select", "test"]:
        n = [m for m in meta if m[0] == split]
        print(f"{split:6s} {len(n)} images · {sum(m[6] for m in n)} boxes · photos {len({m[3] for m in n})} · "
              f"boards {sorted({m[3][:2] for m in n})}")
    assert not ({m[3] for m in meta if m[0] == 'test'} & {m[3] for m in meta if m[0] != 'test' and m[2] != 'free'})


if __name__ == "__main__":
    main()
