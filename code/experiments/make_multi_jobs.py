"""检查 C2（2026-11-20）：每张合成图 2 个缺陷（论文未说明每张图几个缺陷）
在 jobs_A 的每个任务上找第二个放置位置：同板、同类、来自划分 A 训练照片的真实缺陷，完整落在背景裁剪内，
且它的 64 窗口与第一个缺陷的窗口不重叠（真实照片里同一类缺陷通常有多处）。
写出 data/synthetic/jobs_A_multi/：背景 = crg_A 已生成的图（已含第一个缺陷），任务名 j1xxx（种子与第一个缺陷不同）。
之后：generate.py --method crg --jobs jobs_A_multi --tag multi_A 生成第二个缺陷；merge 子命令合并成 crgmd_A（300 张）。
"""
import csv
import random
import shutil
import sys

from common import FREE, PATCH, SYN, TRAIN, board, boxes, misaligned, window_of
from make_jobs import EDGE, MAX_SIDE

SRC, JOBS, OUT = SYN / "crg_A", SYN / "jobs_A", SYN / "jobs_A_multi"
DET_A = SYN.parent / "detection_A"


def make():
    rng = random.Random(3)
    allowed = {r["photo"] for r in csv.DictReader(open(DET_A / "photo_split.csv")) if r["split"] == "train"}
    bad = misaligned()
    real = {}
    for s in TRAIN:
        if s in allowed and s not in bad:
            for c, x0, y0, x1, y1 in boxes(s):
                if max(x1 - x0, y1 - y0) <= MAX_SIDE:
                    real.setdefault((board(s), c), []).append((round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)))
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rows, jobs = [], list(csv.DictReader(open(JOBS / "jobs.csv")))[:300]
    from common import CLASSES
    for j in jobs:
        c, b, l, t = CLASSES.index(j["class"]), j["board"], int(j["bg_left"]), int(j["bg_top"])
        fx, fy = (float(j["bx0"]) + float(j["bx1"])) / 2, (float(j["by0"]) + float(j["by1"])) / 2
        flx, fly = window_of(int(fx), int(fy))
        cand = []
        for x0, y0, x1, y1 in set(real.get((b, c), [])):
            bx = (x0 - l, y0 - t, x1 - l, y1 - t)
            if bx[0] < EDGE or bx[2] > FREE - EDGE or bx[1] < EDGE or bx[3] > FREE - EDGE:
                continue
            lx, ly = window_of(int((bx[0] + bx[2]) / 2), int((bx[1] + bx[3]) / 2))
            if abs(lx - flx) >= PATCH or abs(ly - fly) >= PATCH:                    # 两个 64 窗口不重叠
                cand.append(bx)
        if not cand:
            continue
        bx = rng.choice(sorted(cand))
        jid = f"j1{j['jid'][2:]}"
        shutil.copy(SRC / "images" / f"{j['jid']}.png", OUT / f"{jid}.png")
        (OUT / f"{jid}.txt").write_text(f"{c} {(bx[0] + bx[2]) / 2 / FREE:.6f} {(bx[1] + bx[3]) / 2 / FREE:.6f}\n")
        rows.append([jid, j["class"], b, j["bg_file"], l, t, j["place_photo"], *bx, j["jid"]])
    with open(OUT / "jobs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["jid", "class", "board", "bg_file", "bg_left", "bg_top", "place_photo", "bx0", "by0", "bx1", "by1", "first_jid"])
        w.writerows(rows)
    from collections import Counter
    print(f"{len(rows)} of {len(jobs)} images get a second defect; per class {dict(Counter(r[1] for r in rows))}", flush=True)


def merge():
    """crgmd_A：有第二个缺陷的用 crgmulti_A 的图、两行标签；其余沿用 crg_A（单缺陷）"""
    dst = SYN / "crgmd_A"
    (dst / "images").mkdir(parents=True, exist_ok=True); (dst / "labels").mkdir(parents=True, exist_ok=True)
    second = {r["first_jid"]: r["jid"] for r in csv.DictReader(open(OUT / "jobs.csv"))}
    multi, n2 = SYN / "crgmulti_A", 0
    for j in [r["jid"] for r in csv.DictReader(open(JOBS / "jobs.csv"))][:300]:
        k = second.get(j)
        two = k is not None and (multi / "labels" / f"{k}.txt").exists()
        src_img = multi / "images" / f"{k}.png" if two else SRC / "images" / f"{j}.png"
        lines = (SRC / "labels" / f"{j}.txt").read_text() + ((multi / "labels" / f"{k}.txt").read_text() if two else "")
        img = dst / "images" / f"{j}.png"
        if img.is_symlink() or img.exists():
            img.unlink()
        img.symlink_to(src_img.resolve())
        (dst / "labels" / f"{j}.txt").write_text(lines)
        n2 += two
    print(f"crgmd_A: 300 images, {n2} with two defects", flush=True)


if __name__ == "__main__":
    {"make": make, "merge": merge}[sys.argv[1]]()
