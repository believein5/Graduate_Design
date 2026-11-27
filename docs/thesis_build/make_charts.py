# 论文第4章补充图（2026-11-27）：
#   fig_trend：不同真实数据量下的 mAP@50-95（测试集，3 个随机种子；只用真实数据与双扩散协同引导画出 ±1 标准差），其他方法为细灰线
#   fig_ablation：消融实验各设置在同一批生成任务上的缺陷窗口（每类一个任务）
# 数字直接读 results/experiments/detection_*/<方法>/seed*/metrics.json；图像读 data/synthetic/<设置>_A/images
# 图内文字小五号（9 pt），中文 FandolSong，数字与字母 Times New Roman；黑白打印时靠线型和标记区分
# 用法：python -I make_charts.py <Graduate_Design 目录> [--jids j0006,j0007,...]
import csv, json, statistics as st, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw

ROOT = Path(sys.argv[1])
OUT = ROOT / "docs" / "thesis_build" / "figures"
RES = ROOT / "results" / "experiments"
SYN = ROOT / "data" / "synthetic"
FONTS = ROOT / "tools" / "libreoffice" / "fonts"
for f in (FONTS / "ttf" / "times.ttf", FONTS / "fandol_m" / "FandolSong-Regular.otf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": ["Times New Roman", "FandolSong"], "font.size": 9, "mathtext.fontset": "stix",
                     "axes.unicode_minus": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
INK, MUTED, GRID, ACCENT = "#1f1f1f", "#9a9a9a", "#e3e3e3", "#2a78d6"


def save_size(name, im_px, width_pt):
    sizes = json.load(open(OUT / "sizes.json"))
    sizes[name] = {"px": list(im_px), "pt": [width_pt, width_pt * im_px[1] / im_px[0]]}
    json.dump(sizes, open(OUT / "sizes.json", "w"), indent=1)


# ===================== 趋势图 =====================
NS = [10, 20, 30, 40, 50, 60]
DIRS = {10: "detection_A10_img640", 20: "detection_A20_img640", 30: "detection_A30_img640", 40: "detection_A40_img640",
        50: "detection_A_img640", 60: "detection_A60_img640"}
OTHERS = [("dpmpaste_1x", "DDIM微调"), ("repaint_1x", "RePaint"), ("copaint_1x", "CoPaint"), ("controlnet_1x", "ControlNet"),
          ("deffiller_1x", "DefFiller"), ("anomalydiffusion_1x", "AnomalyDiffusion")]


def series(m):
    xs, mu, sd = [], [], []
    for n in NS:
        p = [RES / DIRS[n] / m / f"seed{s}" / "metrics.json" for s in range(3)]
        if all(x.exists() for x in p):
            v = [json.load(open(x))["map50_95"] for x in p]
            xs.append(n); mu.append(st.mean(v)); sd.append(st.stdev(v))
    return xs, mu, sd


W_PT = 380
fig, ax = plt.subplots(figsize=(W_PT / 72, 230 / 72), dpi=300)
for m, _ in OTHERS:
    x, y, _ = series(m)
    ax.plot(x, y, color=MUTED, lw=0.8, marker="o", ms=2.2, zorder=1)
x, y, e = series("original")
ax.errorbar(x, y, yerr=e, color=INK, lw=1.4, ls="--", marker="s", ms=4, capsize=2, elinewidth=0.8, zorder=3, label="只用真实数据")
x, y, e = series("crg_1x")
ax.errorbar(x, y, yerr=e, color=ACCENT, lw=1.6, marker="o", ms=4.5, capsize=2, elinewidth=0.8, zorder=4, label="双扩散协同引导")
ax.plot([], [], color=MUTED, lw=0.8, marker="o", ms=2.2, label="其他生成方法")
ax.set_xticks(NS); ax.set_xlim(7, 63)
ax.set_xlabel("每类真实图像数"); ax.set_ylabel("mAP@50-95")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
h, l = ax.get_legend_handles_labels(); o = [l.index(k) for k in ("双扩散协同引导", "只用真实数据", "其他生成方法")]
ax.legend([h[i] for i in o], [l[i] for i in o], frameon=False, loc="lower right")
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig_trend.jpg", dpi=300, pil_kwargs={"quality": 95})
im = Image.open(OUT / "fig_trend.jpg"); save_size("fig_trend", im.size, W_PT); print("fig_trend", im.size)

# ===================== 消融图 =====================
jids = sys.argv[sys.argv.index("--jids") + 1].split(",") if "--jids" in sys.argv else \
    ["j0006", "j0007", "j0020", "j0021", "j0010", "j0011"]
ROWS = [("jobs_A", "无缺陷背景"), ("crg_A", "完整方法"), ("crgscan_abl_nodef_A", "去掉缺陷损失"),
        ("crgscan_abl_nobg_A", "去掉窗口内背景损失"), ("crgscan_abl_noout_A", "去掉窗口外背景损失"),
        ("crgscan_abl_noema_A", "去掉掩码滑动平均"), ("ecics_A", "去掉注意力掩码")]
CLS = {"missinghole": "缺孔", "mousebite": "鼠咬", "opencircuit": "开路", "short": "短路", "spur": "毛刺", "spuriouscopper": "多余铜箔"}
jobs = {j["jid"]: j for j in csv.DictReader(open(SYN / "jobs_A" / "jobs.csv"))}
sys.path.insert(0, str(ROOT / "code" / "experiments"))
from common import FREE, window_of                                     # noqa: E402

C = 96                                         # 以 64×64 窗口为中心截取 96×96，保留一圈背景以看出衔接
W2_PT = 400
fig, axes = plt.subplots(len(ROWS), len(jids), figsize=(W2_PT / 72, W2_PT / 72 * len(ROWS) / len(jids) * 0.92), dpi=300)
for c, jid in enumerate(jids):
    lab = open(SYN / "jobs_A" / f"{jid}.txt").read().split()
    lx, ly = window_of(int(float(lab[1]) * FREE), int(float(lab[2]) * FREE))
    x0 = min(max(lx + 32 - C // 2, 0), FREE - C); y0 = min(max(ly + 32 - C // 2, 0), FREE - C)
    for r, (d, name) in enumerate(ROWS):
        p = SYN / d / (f"{jid}.png" if d == "jobs_A" else f"images/{jid}.png")
        img = Image.open(p).convert("RGB").crop((x0, y0, x0 + C, y0 + C))
        ImageDraw.Draw(img).rectangle([lx - x0, ly - y0, lx - x0 + 63, ly - y0 + 63], outline=(255, 215, 0), width=1)
        a = axes[r, c]; a.imshow(img, interpolation="nearest"); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_linewidth(0.4); s.set_color("#94a3b8")
        if c == 0:
            a.set_ylabel(name, rotation=0, ha="right", va="center", labelpad=4)
        if r == 0:
            a.set_title(CLS[jobs[jid]["class"]], pad=3)
fig.subplots_adjust(left=0.235, right=0.995, top=0.95, bottom=0.005, wspace=0.05, hspace=0.06)
fig.savefig(OUT / "fig_ablation.jpg", dpi=300, pil_kwargs={"quality": 95})
im = Image.open(OUT / "fig_ablation.jpg"); save_size("fig_ablation", im.size, W2_PT); print("fig_ablation", im.size, jids)
