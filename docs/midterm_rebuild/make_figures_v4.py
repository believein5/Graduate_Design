# 中期报告第四轮新增插图
# 用法：python -I make_figures_v4.py <Graduate_Design 目录> <输出 figures 目录>
# fig_kid：训练过程中各保存权重的 KID（results/comparison/checkpoint_selection.json），(a) 缺陷先验模型 (b) 背景合成模型；
#          图内中文用 Noto Serif CJK SC（宋体风格），数字与字母用 Times New Roman，按插入宽度下 10.5 磅（五号）排版
# fig_crg：划分 A 的任务 j0000–j0005 的无缺陷背景与 CRG 生成结果（作者权重），取法与 fig_methods 相同
import sys, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import font_manager
from PIL import Image, ImageDraw

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SIZES = json.load(open(OUT / "sizes.json"))
DPI = 300

# ---------- fig_kid ----------
for f in ["/tmp/lo_fonts/ttf/times.ttf", "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"]:
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Times New Roman", "Noto Serif CJK SC"], "font.size": 10.5,
                     "mathtext.fontset": "stix", "axes.unicode_minus": False})
sel = json.load(open(ROOT / "results" / "comparison" / "checkpoint_selection.json"))
RUNS = [("ours · paper data (50/class)", "（a）"), ("ours · paper data (3,000 crops)", "（b）")]
W_PT = 420
fig, axes = plt.subplots(1, 2, figsize=(W_PT / 72, 2.75))
for ax, (key, sub) in zip(axes, RUNS):
    rows = sel["selection"][key]
    for w, lab, mk, ls in (("model", "原始权重", "o", "-"), ("ema", "滑动平均权重", "s", "--")):
        r = sorted((x for x in rows if x["weights"] == w), key=lambda x: x["step"])
        ax.plot([x["step"] / 1e4 for x in r], [x["kid_x1e3"] for x in r], ls, marker=mk, ms=2.5, lw=0.9, color="k",
                mfc="k" if w == "model" else "white", label=lab)
    best = Path(sel["best"][key]).name
    b = next(x for x in rows if x["file"] == best)
    ax.plot([b["step"] / 1e4], [b["kid_x1e3"]], "o", ms=9, mfc="none", mec="k", mew=1.0, label="选中的权重")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" if f"{v:g}"[0] in "25" else ""))
    ax.set_xlabel("训练迭代次数/10⁴")
    ax.set_ylabel("KID×10³")
    ax.grid(which="major", lw=0.3, color="0.75")
    ax.tick_params(direction="in", which="both", width=0.6)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    ax.text(0.5, -0.33, sub, transform=ax.transAxes, ha="center", va="top")
axes[0].legend(frameon=False, fontsize=9, loc="upper right")
fig.tight_layout(w_pad=1.5)
fig.subplots_adjust(bottom=0.3)
fig.savefig(OUT / "fig_kid.jpg", dpi=DPI, pil_kwargs={"quality": 95})
w, h = Image.open(OUT / "fig_kid.jpg").size
SIZES["fig_kid"] = {"px": [w, h], "pt": [w * 72 / DPI, h * 72 / DPI]}

# ---------- fig_crg ----------
SYN = ROOT / "data" / "synthetic"
ROWS = [None, "crg_A"]                                   # 2026-11-25：只保留背景与作者权重的生成结果
JIDS = [f"j{i:04d}" for i in range(6)]
CELL, GAP, CROP, SIZE = 160, 6, 128, 256


def tile(row, jid):
    img = Image.open((SYN / "jobs_A" / f"{jid}.png") if row is None else (SYN / row / "images" / f"{jid}.png")).convert("RGB")
    _, cx, cy, bw, bh = (float(v) for v in (SYN / "crg_A" / "labels" / f"{jid}.txt").read_text().split())
    cx, cy, bw, bh = cx * SIZE, cy * SIZE, bw * SIZE, bh * SIZE
    x0 = int(min(max(cx - CROP / 2, 0), SIZE - CROP)); y0 = int(min(max(cy - CROP / 2, 0), SIZE - CROP))
    t = img.crop((x0, y0, x0 + CROP, y0 + CROP)).resize((CELL, CELL), Image.LANCZOS)
    k = CELL / CROP
    ImageDraw.Draw(t).rectangle([(cx - bw / 2 - x0) * k, (cy - bh / 2 - y0) * k, (cx + bw / 2 - x0) * k, (cy + bh / 2 - y0) * k],
                                outline=(230, 30, 30), width=2)
    return t


sheet = Image.new("RGB", (len(JIDS) * CELL + (len(JIDS) - 1) * GAP, len(ROWS) * CELL + (len(ROWS) - 1) * GAP), (255, 255, 255))
for r, row in enumerate(ROWS):
    for c, jid in enumerate(JIDS):
        sheet.paste(tile(row, jid), (c * (CELL + GAP), r * (CELL + GAP)))
sheet.save(OUT / "fig_crg.jpg", quality=95, dpi=(DPI, DPI))
SIZES["fig_crg"] = {"px": list(sheet.size), "pt": [sheet.size[0] * 72 / DPI, sheet.size[1] * 72 / DPI]}

json.dump(SIZES, open(OUT / "sizes.json", "w"), indent=1)
print({k: SIZES[k] for k in ("fig_kid", "fig_crg")})
