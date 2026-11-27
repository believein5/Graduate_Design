"""论文 Fig. 6（第 9 页）与 Fig. 8（第 12 页）的线索提取（2026-11-20）
1) 600 dpi 渲染页中找出每个 256×256 图块的位置与图上的黄色框（矢量叠加，不在位图里），换算成 256 像素坐标
2) 每个图块对应 pdfimages 抽出的原始 256×256 位图（按缩小后的像素差最小匹配）
3) 位图在 PKU 照片上做模板匹配（每块板一张对齐照片，0.5× 及其附近的比例），得到来源板、位置、比例
输出 results/experiments/paper_figs/{tiles.csv, summary.txt}
"""
import csv
import glob
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from common import PKU, RES, board, misaligned

FIG = Path("/tmp/figs")
OUT = RES / "paper_figs"
Image.MAX_IMAGE_PIXELS = None


def tiles_and_boxes(page):
    im = np.asarray(Image.open(FIG / f"r600-{page}.png").convert("RGB")).astype(np.int16)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    yellow = (r > 200) & (g > 200) & (b < 90)
    # 图块：明显偏绿或很暗（掩码插图）的大块区域，先去掉黄色
    content = ((g - r > 25) | (im.sum(-1) < 120)) & ~yellow
    content = ndimage.binary_closing(content, iterations=4)
    lab, n = ndimage.label(content)
    tiles = []
    for sl in ndimage.find_objects(lab):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if h > 250 and w > 250 and 0.85 < w / h < 1.18:
            tiles.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))
    ylab, yn = ndimage.label(ndimage.binary_dilation(yellow, iterations=1))
    boxes = []
    for sl in ndimage.find_objects(ylab):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if h > 8 and w > 8:
            boxes.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))
    return im.astype(np.uint8), sorted(tiles, key=lambda t: (round(t[1] / 200), t[0])), boxes


def match_raster(tile_img, rasters):
    small = cv2.resize(tile_img, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    best = min(rasters.items(), key=lambda kv: np.abs(kv[1][1] - small).mean())
    return best[0], float(np.abs(best[1][1] - small).mean())


def board_refs():
    bad = misaligned()
    stems = sorted(p.stem for p in (PKU / "images").glob("*.jpg"))
    refs = {}
    for s in stems:
        b = board(s)
        if b not in refs and s not in bad:
            refs[b] = (s, cv2.cvtColor(cv2.imread(str(PKU / "images" / f"{s}.jpg")), cv2.COLOR_BGR2GRAY))
    return refs


def locate(raster, refs, scales):
    g = cv2.cvtColor(raster, cv2.COLOR_RGB2GRAY)
    best = (-1, None, None, None)
    for sc in scales:
        for b, (s, full) in refs.items():
            ph = cv2.resize(full, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(ph, g, cv2.TM_CCOEFF_NORMED)
            _, v, _, loc = cv2.minMaxLoc(res)
            if v > best[0]:
                best = (v, b, loc, sc)
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rasters = {}
    for f in sorted(glob.glob(str(FIG / "crg" / "f-0[01][92]-*.png"))):
        im = np.asarray(Image.open(f).convert("RGB"))
        if im.shape[:2] == (256, 256):
            rasters[f] = (im, cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32))
    refs = board_refs()
    rows, lines = [], []
    for page, fig in (("09", "Fig6"), ("12", "Fig8")):
        im, tiles, boxes = tiles_and_boxes(page)
        lines.append(f"{fig}: {len(tiles)} tiles, {len(boxes)} yellow boxes")
        for k, (x0, y0, x1, y1) in enumerate(tiles):
            f, err = match_raster(im[y0:y1, x0:x1], rasters)
            sx, sy = 256 / (x1 - x0), 256 / (y1 - y0)
            ybs = [((bx0 - x0) * sx, (by0 - y0) * sy, (bx1 - x0) * sx, (by1 - y0) * sy)
                   for bx0, by0, bx1, by1 in boxes if bx0 >= x0 - 5 and bx1 <= x1 + 5 and by0 >= y0 - 5 and by1 <= y1 + 5]
            raster = rasters[f][0]
            is_mask = raster.mean() < 40
            v, b, loc, sc = locate(raster, refs, [0.5]) if not is_mask else (np.nan, "", (0, 0), np.nan)
            if not is_mask and v < 0.8:
                v, b, loc, sc = locate(raster, refs, [0.40, 0.45, 0.48, 0.52, 0.55, 0.60, 0.70, 1.0])
            rows.append(dict(fig=fig, tile=k, tile_x=x0, tile_y=y0, raster=Path(f).name, raster_err=round(err, 1), is_mask=is_mask,
                             match_r=round(v, 3) if v == v else "", board=b, scale=sc, left=loc[0], top=loc[1],
                             n_yellow=len(ybs), yellow=";".join(",".join(f"{c:.0f}" for c in yb) for yb in ybs)))
    with open(OUT / "tiles.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__" and not __import__("sys").argv[1:]:
    main()


# ---------- 第 2 步：与无缺陷板模板比对，找生成缺陷的位置，并和黄色框、真实缺陷位置对照 ----------
ROWS6 = {703: "DDIM", 1235: "RePaint", 1767: "CoPaint", 2299: "ControlNet", 2831: "AnomalyDiffusion", 3363: "DefFiller", 3895: "CRG"}
COLS6 = [1474, 2008, 2538, 3070, 3602, 4134]
ROWS8 = {588: 0, 944: 1, 1299: 2, 1655: 3, 2010: 4, 2365: 5}
CLASSES = ["missinghole", "mousebite", "opencircuit", "short", "spur", "spuriouscopper"]


def median_templates():
    from common import boxes, photo
    bad = misaligned()
    stems = sorted(p.stem for p in (PKU / "images").glob("*.jpg"))
    temps, real = {}, {}
    for b in sorted({board(s) for s in stems}):
        ph = [s for s in stems if board(s) == b and s not in bad]
        imgs = [np.asarray(photo(s), np.uint8) for s in ph[:15]]
        h = min(i.shape[0] for i in imgs); w = min(i.shape[1] for i in imgs)
        temps[b] = np.median(np.stack([i[:h, :w] for i in imgs]), 0).astype(np.uint8)
        real[b] = [(c, x0, y0, x1, y1, s) for s in ph for c, x0, y0, x1, y1 in boxes(s)]
    return temps, real


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h)


def stage2():
    temps, real = median_templates()
    rows = list(csv.DictReader(open(OUT / "tiles.csv")))
    out = []
    for r in rows:
        f = FIG / "crg" / r["raster"]
        raster = np.asarray(Image.open(f).convert("RGB"))
        if (raster < 30).all(-1).mean() > 0.85:                                    # 掩码插图
            lab, n = ndimage.label(raster.mean(-1) > 128)
            comps = [(sl[1].start, sl[0].start, sl[1].stop, sl[0].stop) for sl in ndimage.find_objects(lab)]
            out.append({**r, "kind": "mask", "mask_components": ";".join(",".join(map(str, c)) for c in comps)}); continue
        g = cv2.cvtColor(raster, cv2.COLOR_RGB2GRAY)
        best = (-1, None, None)
        for b, t in temps.items():
            res = cv2.matchTemplate(cv2.cvtColor(t, cv2.COLOR_RGB2GRAY), g, cv2.TM_CCOEFF_NORMED)
            _, v, _, loc = cv2.minMaxLoc(res)
            if v > best[0]:
                best = (v, b, loc)
        v, b, (l, t0) = best
        tpl = temps[b][t0:t0 + 256, l:l + 256].astype(np.int16)
        d = ndimage.uniform_filter(np.abs(raster.astype(np.int16) - tpl).max(-1).astype(np.float32), 3)
        m = d > 40
        lab, n = ndimage.label(m)
        comps = []
        for i, sl in enumerate(ndimage.find_objects(lab), 1):
            if (lab[sl] == i).sum() >= 12:
                comps.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))
        yb = [tuple(map(float, y.split(","))) for y in r["yellow"].split(";") if y]
        col = int(np.argmin([abs(int(r["tile_x"]) - c) for c in COLS6])) if r["fig"] == "Fig6" else ROWS8[min(ROWS8, key=lambda y: abs(y - int(r["tile_y"])))]
        cls = CLASSES.index(CLASSES[col])
        notes = []
        for y in yb:
            gb = (y[0] + l, y[1] + t0, y[2] + l, y[3] + t0)                       # 黄色框换算到 0.5× 板坐标
            same = [(iou(gb, rb[1:5]), rb) for rb in real[b] if rb[0] == cls]
            best_real = max(same, default=(0, None))
            chg = max((iou(y, c) for c in comps), default=0)
            notes.append(f"yellow {y[2]-y[0]:.0f}x{y[3]-y[1]:.0f} · IoU with changed region {chg:.2f} · best IoU with a real "
                         f"{CLASSES[cls]} box on board {b}: {best_real[0]:.2f}"
                         + (f" ({best_real[1][5]}, {best_real[1][3]-best_real[1][1]:.0f}x{best_real[1][4]-best_real[1][2]:.0f})" if best_real[1] else ""))
        # 背景窗口里有没有该板上的真实缺陷位置（不同照片的缺陷在同一板的不同位置）
        inside = sum(1 for rb in real[b] if rb[1] >= l and rb[3] <= l + 256 and rb[2] >= t0 and rb[4] <= t0 + 256)
        out.append({**r, "kind": "image", "t_board": b, "t_r": round(v, 3), "t_left": l, "t_top": t0,
                    "row": ROWS6.get(min(ROWS6, key=lambda y: abs(y - int(r["tile_y"])))) if r["fig"] == "Fig6" else "ablation",
                    "class": CLASSES[cls], "changed": ";".join(",".join(map(str, c)) for c in comps),
                    "real_boxes_in_window": inside, "notes": " | ".join(notes)})
    keys = sorted({k for o in out for k in o}, key=lambda k: list(out[-1]).index(k) if k in out[-1] else 99)
    with open(OUT / "tiles2.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(out)
    for o in out:
        if o["kind"] == "image":
            print(f"{o['fig']} {o.get('row',''):16s} {o['class']:14s} board {o['t_board']} r {o['t_r']} at ({o['t_left']},{o['t_top']}) "
                  f"real boxes in window {o['real_boxes_in_window']:3d} | {o['notes']}")
        else:
            print(f"{o['fig']} tile {o['tile']} mask components {o['mask_components']} · yellow {o['yellow']}")


if __name__ == "__main__" and __import__("sys").argv[1:] == ["stage2"]:
    stage2()
