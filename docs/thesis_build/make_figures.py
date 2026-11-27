# 毕业论文插图：收集到 docs/thesis_build/figures/，并写 sizes.json（像素与 300 dpi 下的磅数）
# 用法：python -I make_figures.py <Graduate_Design 目录>
# - op_*：开题报告 .docx 中的插图（PCB 图像、六类缺陷、技术路线、DDPM 示意）
# - 中期报告的插图直接复制
# - fig_methods_thesis：各方法在同一批生成任务上的结果（同 make_figures_v3.py，去掉自训练权重一行）
import json, shutil, sys, zipfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(sys.argv[1])
OUT = ROOT / "docs" / "thesis_build" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
MID = ROOT / "docs" / "midterm_rebuild" / "figures"
sizes = {}


def add(name, img_path):
    im = Image.open(img_path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    dst = OUT / f"{name}.jpg"
    im.save(dst, quality=95)
    sizes[name] = {"px": list(im.size)}


with zipfile.ZipFile(ROOT / "report" / "2023313823-陈斯萍-基于扩散模型的印刷电路板缺陷图像生成方法研究.docx") as z:
    for name, member in [("op_pcb", "word/media/fig1.jpg"), ("op_defects", "word/media/fig4.jpg"),
                         ("op_route", "word/media/fig3.jpg"), ("op_ddpm", "word/media/fig5.jpg")]:
        with z.open(member) as f:
            tmp = OUT / f"_{name}.jpg"
            tmp.write_bytes(f.read())
        add(name, tmp)
        tmp.unlink()

for name in ["fig_pre_overview", "fig_pre_patches", "fig_pre_rescale", "fig_pre_clean", "fig_pre_coverage", "fig_pre_free",
             "fig_64_samples", "fig_64_process", "fig_256_samples", "fig_crg"]:
    add(name, MID / f"{name}.jpg")

SYN = ROOT / "data" / "synthetic"
ROWS = [None, "dpmpaste_A", "repaint_A", "copaint_A", "controlnet_A", "deffiller_A", "crg_A"]
JIDS = [f"j{i:04d}" for i in range(6)]
CELL, GAP, CROP, SIZE = 160, 6, 128, 256


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


sheet = Image.new("RGB", (len(JIDS) * CELL + (len(JIDS) - 1) * GAP, len(ROWS) * CELL + (len(ROWS) - 1) * GAP), (255, 255, 255))
for r, row in enumerate(ROWS):
    for c, jid in enumerate(JIDS):
        sheet.paste(tile(row, jid), (c * (CELL + GAP), r * (CELL + GAP)))
sheet.save(OUT / "fig_methods_thesis.jpg", quality=95)
sizes["fig_methods_thesis"] = {"px": list(sheet.size)}

json.dump(sizes, open(OUT / "sizes.json", "w"), indent=1)
print({k: v["px"] for k, v in sizes.items()})
