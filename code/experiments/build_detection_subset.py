"""划分 A 训练集的子集：每类 n 张缺陷裁剪 + n 张无缺陷裁剪（与原来 50 + 50 的比例相同），select / test 与划分 A 完全相同
用于检查真实数据更少时扩增是否更有用（detect.py --split A10 / A20）。每类按固定随机顺序取前 n 张，所以 10 ⊂ 20 ⊂ 50。

python build_detection_subset.py --n 10 20   → data/detection_A_n10/、data/detection_A_n20/
"""
import argparse
import csv
import random
import shutil
from collections import Counter

from common import DET

SRC = DET.parent / "detection_A"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[10, 20])
    args = ap.parse_args()
    order = {}
    for r in csv.DictReader(open(SRC / "meta.csv")):
        if r["split"] == "train":
            order.setdefault(r["class"], []).append(r["name"])
    rng = random.Random(0)
    for c in order:
        order[c] = sorted(order[c]); rng.shuffle(order[c])
    for n in args.n:
        dst = DET.parent / f"detection_A_n{n}"
        if dst.exists():
            shutil.rmtree(dst)
        for sub in ("images", "labels"):
            (dst / "train" / sub).mkdir(parents=True)
        count = Counter()
        for c, names in order.items():
            for name in names[:n]:
                shutil.copy(SRC / "train" / "images" / f"{name}.png", dst / "train" / "images")
                shutil.copy(SRC / "train" / "labels" / f"{name}.txt", dst / "train" / "labels")
                count[c] += 1
        for split in ("select", "test"):
            shutil.copytree(SRC / split, dst / split, ignore=shutil.ignore_patterns("*.cache"))
        print(f"{dst.name}: train {sum(count.values())} crops {dict(count)}", flush=True)


if __name__ == "__main__":
    main()
