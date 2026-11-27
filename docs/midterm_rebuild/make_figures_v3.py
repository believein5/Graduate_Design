# 中期报告第三轮新增插图（图内不含文字，说明写在 Word 图题中）
# 用法：python -I make_figures_v3.py <Graduate_Design 目录> <输出 figures 目录>
# fig_methods：划分 A 的任务 j0000–j0005（依次为缺孔、鼠咬、开路、短路、毛刺、多余铜箔）在各方法下的生成结果，
#              以标签框中心取 128×128 区域，红框为标签框；行依次为无缺陷背景、DDIM、RePaint、CoPaint、ControlNet、
#              DefFiller、CRG（作者权重）、CRG（自训练权重）
import sys, json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SYN = ROOT / "data" / "synthetic"
ROWS = [None, "dpmpaste_A", "repaint_A", "copaint_A", "controlnet_A", "deffiller_A", "crg_A", "crgours_A"]
JIDS = [f"j{i:04d}" for i in range(6)]
CELL, GAP, CROP, SIZE = 160, 6, 128, 256
WHITE = (255, 255, 255)


def tile(row, jid):
    img = Image.open((SYN / "jobs_A" / f"{jid}.png") if row is None else (SYN / row / "images" / f"{jid}.png")).convert("RGB")
    _, cx, cy, w, h = (float(v) for v in (SYN / "crg_A" / "labels" / f"{jid}.txt").read_text().split())
    cx, cy, w, h = cx * SIZE, cy * SIZE, w * SIZE, h * SIZE
    x0 = int(min(max(cx - CROP / 2, 0), SIZE - CROP)); y0 = int(min(max(cy - CROP / 2, 0), SIZE - CROP))
    t = img.crop((x0, y0, x0 + CROP, y0 + CROP)).resize((CELL, CELL), Image.LANCZOS)
    k = CELL / CROP
    ImageDraw.Draw(t).rectangle([(cx - w / 2 - x0) * k, (cy - h / 2 - y0) * k, (cx + w / 2 - x0) * k, (cy + h / 2 - y0) * k],
                                outline=(230, 30, 30), width=2)
    return t


W = len(JIDS) * CELL + (len(JIDS) - 1) * GAP
H = len(ROWS) * CELL + (len(ROWS) - 1) * GAP
sheet = Image.new("RGB", (W, H), WHITE)
for r, row in enumerate(ROWS):
    for c, jid in enumerate(JIDS):
        sheet.paste(tile(row, jid), (c * (CELL + GAP), r * (CELL + GAP)))
dpi = 300
sheet.save(OUT / "fig_methods.jpg", quality=95, dpi=(dpi, dpi))
sizes = json.load(open(OUT / "sizes.json"))
sizes["fig_methods"] = {"px": list(sheet.size), "pt": [sheet.size[0] * 72 / dpi, sheet.size[1] * 72 / dpi]}
json.dump(sizes, open(OUT / "sizes.json", "w"), indent=1)
print(sizes["fig_methods"])
