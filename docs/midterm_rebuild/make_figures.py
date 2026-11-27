# 生成中期报告插图（图内不含文字：本机无宋体，图内说明全部写在 Word 图题中）
# 用法：python -I make_figures.py <Graduate_Design 目录> <输出 figures 目录>
import sys, json, random
from pathlib import Path
from PIL import Image, ImageDraw

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
PKU = ROOT / "data" / "PKU-Market-PCB-corrected"
CRG = ROOT / "third_party" / "CRG-DefectDiffuser"
SAMPLES = ROOT / "results" / "checkpoint_samples"
WHITE = (255, 255, 255)
sizes = {}


def save(img, name, dpi=300):
    img.save(OUT / f"{name}.jpg", quality=95, dpi=(dpi, dpi))
    sizes[name] = {"px": list(img.size), "pt": [img.size[0] * 72 / dpi, img.size[1] * 72 / dpi]}


# ---------- 从 matplotlib 网格图中切出各个子图 ----------
def bands(profile, thr):
    out, start = [], None
    for i, v in enumerate(profile + [0]):
        if v > thr and start is None:
            start = i
        elif v <= thr and start is not None:
            if i - start > 20:
                out.append((start, i))
            start = None
    return out


def tiles(path):
    """按非白像素投影找到各子图（含 1 px 黑色边框），返回按行排列的图块列表"""
    im = Image.open(path).convert("RGB")
    W, H = im.size
    px = im.load()
    nonwhite = [[min(px[x, y]) < 235 for x in range(W)] for y in range(H)]
    rows = bands([sum(r) / W for r in nonwhite], 0.45)
    grid = []
    for y0, y1 in rows:
        ym = (y0 + y1) // 2
        cols = bands([sum(nonwhite[y][x] for y in range(y0, y1)) / (y1 - y0) for x in range(W)], 0.9)
        # 去掉边框：向内收 2 px
        grid.append([im.crop((x0 + 2, y0 + 2, x1 - 2, y1 - 2)) for x0, x1 in cols])
    return grid


def compose(grid, cell, gap, pad=0):
    """把图块重新排成等尺寸网格（白色间隔），不含任何文字"""
    nr, nc = len(grid), max(len(r) for r in grid)
    out = Image.new("RGB", (nc * cell + (nc - 1) * gap + 2 * pad, nr * cell + (nr - 1) * gap + 2 * pad), WHITE)
    for r, row in enumerate(grid):
        for c, t in enumerate(row):
            out.paste(t.resize((cell, cell), Image.NEAREST), (pad + c * (cell + gap), pad + r * (cell + gap)))
    return out


g64 = tiles(SAMPLES / "checkpoint-64-defect_64.png")
assert len(g64) == 6 and all(len(r) == 8 for r in g64), [len(r) for r in g64]
save(compose(g64, 192, 12), "fig_64_samples")

gproc = tiles(SAMPLES / "checkpoint-64-defect_64_process_short.png")
assert len(gproc) == 4 and all(len(r) == 8 for r in gproc), [len(r) for r in gproc]
save(compose(gproc, 192, 12), "fig_64_process")

g256 = tiles(SAMPLES / "checkpoint-256-defect-free_256.png")
assert len(g256) == 2 and all(len(r) == 4 for r in g256), [len(r) for r in g256]
save(compose(g256, 512, 16), "fig_256_samples")

# ---------- 预处理示例：直接取 data/pcb_processed 中已构建的训练数据（按清单文件定位） ----------
import csv
import math
import numpy as np
from matplotlib import colormaps

PROC = ROOT / "data" / "pcb_processed"
SCALE, PATCH, FREE, FREE_MARGIN, EDGE_MARGIN, SEED = 0.5, 64, 256, 8, 32, 0     # 与 preprocess_walkthrough.ipynb 一致
EXAMPLE = "01_short_17"
TRAIN = [Path(s).stem for s in (PKU / "ImageSets" / "train_board.txt").read_text().split()]
CLASSES = [n.replace("_", "") for n in (PKU / "classes.txt").read_text().split()]
dman = list(csv.DictReader(open(PROC / "defect_manifest.csv")))
fman = list(csv.DictReader(open(PROC / "free_manifest.csv")))


def board(stem):
    return stem[:2]


def load(stem):
    img = Image.open(PKU / "images" / f"{stem}.jpg").convert("RGB")
    W, H = img.size
    boxes = []
    for line in (PKU / "labels" / f"{stem}.txt").read_text().splitlines():
        c, cx, cy, w, h = line.split()
        c, cx, cy, w, h = int(c), float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
        boxes.append((c, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return img, boxes


def scaled(stem):
    img, boxes = load(stem)
    img_s = img.resize((round(img.width * SCALE), round(img.height * SCALE)), Image.BICUBIC)
    return img_s, [(c, x0 * SCALE, y0 * SCALE, x1 * SCALE, y1 * SCALE) for c, x0, y0, x1, y1 in boxes]


def clean_photo(stem):
    """与 notebook 的 clean_photo 相同：标注框外扩 FREE_MARGIN 内的像素用板模板替换"""
    img_s, boxes_s = scaled(stem)
    a = np.asarray(img_s).copy()
    t = np.asarray(Image.open(PROC / "templates" / f"board_{board(stem)}.png").convert("RGB"))
    H, W = a.shape[:2]
    for _, x0, y0, x1, y1 in boxes_s:
        l, tp = max(0, int(x0 - FREE_MARGIN)), max(0, int(y0 - FREE_MARGIN))
        r, b = min(W, int(x1 + FREE_MARGIN) + 1), min(H, int(y1 + FREE_MARGIN) + 1)
        a[tp:b, l:r] = t[tp:b, l:r]
    return Image.fromarray(a), boxes_s


# 图：缩放后示例图像上的标注框（红）与缺陷局部块截取窗口（黄）
img_s, boxes_s = scaled(EXAMPLE)
over = img_s.copy()
d = ImageDraw.Draw(over)
for r in (r for r in dman if r["source"] == EXAMPLE):
    l, t, S = int(r["left"]), int(r["top"]), int(r["window"])
    d.rectangle((l, t, l + S, t + S), outline=(255, 220, 0), width=4)
for _, x0, y0, x1, y1 in boxes_s:
    d.rectangle((x0, y0, x1, y1), outline=(255, 0, 0), width=3)
save(over, "fig_pre_overview", dpi=int(over.width / (425 / 72)))      # 整版宽

# 图：六类缺陷各取该类第一张训练图像的局部块（已构建的训练文件）
rows = []
for name in CLASSES:
    stem = next(r["source"] for r in dman if r["class"] == name)
    rows.append([Image.open(PROC / "defect_64" / r["file"]).convert("RGB")
                 for r in dman if r["source"] == stem][:5])
save(compose(rows, 192, 12), "fig_pre_patches")

# 图：二次缩放的局部块，每个涉及的类别取第一个；上行为放大的截取窗口，下行为保存的 64×64 局部块
big = [r for r in dman if r["window"] != str(PATCH)]
picked = [next(r for r in big if r["class"] == c) for c in ("spuriouscopper", "short", "spur", "missinghole")]
top_row, bottom_row = [], []
for r in picked:
    l, t, S = int(r["left"]), int(r["top"]), int(r["window"])
    top_row.append(scaled(r["source"])[0].crop((l, t, l + S, t + S)).resize((192, 192), Image.LANCZOS))
    bottom_row.append(Image.open(PROC / "defect_64" / r["file"]).convert("RGB"))
save(compose([top_row, bottom_row], 192, 12), "fig_pre_rescale")
print("rescale examples:", [(r["class"], r["source"], r["window"], r["effective_scale"]) for r in picked])

# 图：示例图像中各缺陷的去除结果；上行原图，中行板模板，下行去除缺陷后（以框中心取 96×96）
clean, _ = clean_photo(EXAMPLE)
tpl = Image.open(PROC / "templates" / f"board_{board(EXAMPLE)}.png").convert("RGB")
grid = [[], [], []]
for _, x0, y0, x1, y1 in boxes_s:
    cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
    win = (cx - 48, cy - 48, cx + 48, cy + 48)
    for k, im in enumerate((img_s, tpl, clean)):
        grid[k].append(im.crop(win))
save(compose(grid, 192, 12), "fig_pre_clean")


# 图：电路板 01 上背景块的覆盖次数。左：原方法（随机放窗口，靠近标注框的舍弃，每张 6 个）；
# 右：新方法，即 free_manifest.csv 中的实际位置。两图色标相同
def boxes_only(stem):
    """缩放后的标注框，不解码整幅图像"""
    with Image.open(PKU / "images" / f"{stem}.jpg") as im:
        W, H = im.size
    out = []
    for line in (PKU / "labels" / f"{stem}.txt").read_text().splitlines():
        _, cx, cy, w, h = map(float, line.split())
        out.append(((cx - w / 2) * W * SCALE, (cy - h / 2) * H * SCALE, (cx + w / 2) * W * SCALE, (cy + h / 2) * H * SCALE))
    return out, (round(W * SCALE), round(H * SCALE))


st0 = [s for s in TRAIN if board(s) == "01"]
W0, H0 = boxes_only(st0[0])[1]
rng = random.Random(SEED)
per_image = math.ceil(3000 / len(TRAIN))
old_plan = {}
for s in st0:                                    # 与 notebook 的 rejection_plan 相同
    bx, (W, H) = boxes_only(s)
    acc = []
    for _ in range(200):
        if len(acc) == per_image:
            break
        l = rng.randint(EDGE_MARGIN, W - FREE - EDGE_MARGIN)
        t = rng.randint(EDGE_MARGIN, H - FREE - EDGE_MARGIN)
        if not any(x0 - FREE_MARGIN < l + FREE and x1 + FREE_MARGIN > l and y0 - FREE_MARGIN < t + FREE
                   and y1 + FREE_MARGIN > t for x0, y0, x1, y1 in bx):
            acc.append((l, t))
    old_plan[s] = acc
new_plan = {}
for r in fman:
    new_plan.setdefault(r["source"], []).append((int(r["left"]), int(r["top"])))


def coverage(plan):
    cov = np.zeros((H0, W0), np.int32)
    for s in st0:
        for l, t in plan.get(s, []):
            cov[t:t + FREE, l:l + FREE] += 1
    return cov


cov_old, cov_new = coverage(old_plan), coverage(new_plan)
inner = np.zeros((H0, W0), bool)                 # 中部区域：离有效区域边界超过 FREE 的像素
inner[EDGE_MARGIN + FREE:H0 - EDGE_MARGIN - FREE, EDGE_MARGIN + FREE:W0 - EDGE_MARGIN - FREE] = True
for name, cov, plan in (("old", cov_old, old_plan), ("new", cov_new, new_plan)):
    v = cov[inner]
    print(f"coverage {name}: crops {sum(len(plan.get(s, [])) for s in st0)} cv {v.std() / v.mean():.3f} "
          f"range {v.min()}–{v.max()}")
vmax = max(cov_old.max(), cov_new.max())
cmap = colormaps["viridis"]
maps = [Image.fromarray((cmap(c / vmax)[:, :, :3] * 255).astype(np.uint8)) for c in (cov_old, cov_new)]
gap = 24
pair = Image.new("RGB", (2 * W0 + gap, H0), WHITE)
pair.paste(maps[0], (0, 0))
pair.paste(maps[1], (W0 + gap, 0))
save(pair, "fig_pre_coverage", dpi=int(pair.width / (425 / 72)))

# 图：一张训练图像去除缺陷后截取的背景块（已构建的训练文件）
stem0 = fman[0]["source"]
save(compose([[Image.open(PROC / "free_256" / r["file"]).convert("RGB") for r in fman if r["source"] == stem0]],
             384, 16), "fig_pre_free")
print("free example:", stem0, sum(r["source"] == stem0 for r in fman))

# ---------- CRG 协同引导生成：原图窗口 / 生成窗口 ----------
order = [("missinghole", 33, 99, (1, 67)), ("mousebite", 100, 151, (68, 119)),
         ("opencircuit", 112, 135, (80, 103)), ("short", 117, 142, (85, 110)),
         ("spur", 117, 187, (85, 155)), ("spuriouscopper", 111, 158, (79, 126))]   # 窗口左上角取自运行日志
src = Image.open(CRG / "data" / "sample" / "sample.jpg").convert("RGB")
top, bottom = [], []
for name, cx, cy, (l, t) in order:
    top.append(src.crop((l, t, l + 64, t + 64)))
    gen = Image.open(CRG / "images" / "samples" / f"sample_{name}_cx{cx}_cy{cy}_1.png").convert("RGB")
    bottom.append(gen.crop((l, t, l + 64, t + 64)))
save(compose([top, bottom], 192, 12), "fig_crg_pairs")

ov = src.resize((768, 768), Image.NEAREST)
d = ImageDraw.Draw(ov)
for name, cx, cy, (l, t) in order:
    d.rectangle((l * 3, t * 3, (l + 64) * 3, (t + 64) * 3), outline=(255, 220, 0), width=3)
save(ov, "fig_crg_input")

json.dump(sizes, open(OUT / "sizes.json", "w"), indent=1)
print(json.dumps(sizes, indent=1))
