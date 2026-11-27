"""生成任务列表 data/synthetic/jobs/：900 个（背景, 缺陷类别, 位置）组合，所有方法共用

- 背景：free_256/ 中已校验的无缺陷裁剪（训练板）
- 位置：同一块板上某张训练照片里该类别真实缺陷的位置。同板照片彼此对齐，所以缺陷落在这块板上
  该类缺陷真实出现的地方（短路在走线之间、缺孔在焊盘上）
- 类别按 i % 6 轮换：前 300 / 600 / 900 个任务分别是 1× / 2× / 3×，每类均衡
- copy-paste 基线的缺陷来源：同类别、同板另一张照片中的真实缺陷
每个任务写一张背景图和一行 `class cx cy`（sample_generation.py 的输入格式），以及 jobs.csv。
"""
import argparse
import csv
import random
import shutil

from PIL import Image

from common import CLASSES, FREE, PROC, SYN, TRAIN, board, boxes, misaligned

N_JOBS = 900
MAX_SIDE = 56        # 缺陷框长边上限：保证完整落在 64×64 窗口内
EDGE = 4             # 缺陷框距背景裁剪边缘至少 4 px
SEED = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="jobs", help="data/synthetic/ 下的输出目录名")
    ap.add_argument("--n", type=int, default=N_JOBS)
    ap.add_argument("--photo-split", default="", help="photo_split.csv：放置位置、copy-paste 来源与背景只取其中 split=train 的照片")
    args = ap.parse_args()
    rng = random.Random(SEED)
    bad = misaligned()
    allowed = None
    if args.photo_split:
        allowed = {r["photo"] for r in csv.DictReader(open(args.photo_split)) if r["split"] == "train"}
    real = {}                                          # (板, 类别) -> [(照片, x0, y0, x1, y1)]，0.5× 坐标
    for s in TRAIN:
        if allowed is not None and s not in allowed:
            continue
        for c, x0, y0, x1, y1 in boxes(s):
            if max(x1 - x0, y1 - y0) <= MAX_SIDE:
                real.setdefault((board(s), c), []).append((s, x0, y0, x1, y1))
    free = list(csv.DictReader(open(PROC / "free_manifest.csv")))
    if allowed is not None:
        free = [r for r in free if r["source"] in allowed]
    used = set()
    out = SYN / args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rows = []
    for i in range(args.n):
        c = i % len(CLASSES)
        for _ in range(1000):
            bg = rng.choice(free)
            b, l, t = bg["source"][:2], int(bg["left"]), int(bg["top"])
            cand = [p for p in real.get((b, c), []) if p[0] not in bad
                    and p[1] - l >= EDGE and p[3] - l <= FREE - EDGE and p[2] - t >= EDGE and p[4] - t <= FREE - EDGE]
            fresh = [p for p in cand if p not in used]
            if fresh or cand:
                place = rng.choice(fresh or cand)
                break
        used.add(place)
        s, x0, y0, x1, y1 = place
        bx = (x0 - l, y0 - t, x1 - l, y1 - t)                      # 背景裁剪内的放置框
        # copy-paste 的来源缺陷：同类、同板、另一张照片
        others = [p for p in real[(b, c)] if p[0] != s]
        src = rng.choice(others or [p for p in real[(b, c)] if p != place] or [place])   # 优先取另一张照片；只剩一张时取同照片的另一个缺陷
        jid = f"j{i:04d}"
        Image.open(PROC / "free_256" / bg["file"]).convert("RGB").save(out / f"{jid}.png")
        cx, cy = (bx[0] + bx[2]) / 2 / FREE, (bx[1] + bx[3]) / 2 / FREE
        (out / f"{jid}.txt").write_text(f"{c} {cx:.6f} {cy:.6f}\n")
        rows.append([jid, CLASSES[c], b, bg["file"], l, t, s, *(round(v, 2) for v in bx), src[0], *(round(v, 2) for v in src[1:])])
    with open(out / "jobs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["jid", "class", "board", "bg_file", "bg_left", "bg_top", "place_photo", "bx0", "by0", "bx1", "by1",
                    "src_photo", "sx0", "sy0", "sx1", "sy1"])
        w.writerows(rows)
    print(f"{len(rows)} jobs · distinct placements {len(used)} · boards {sorted({r[2] for r in rows})}")


if __name__ == "__main__":
    main()
