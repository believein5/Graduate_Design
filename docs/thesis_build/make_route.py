# 图1-3 本文技术路线图（2026-11-26，按用户提供的草图 source/route_draft_user.png 重画）
# 与正文第2～5章对应；不出现原论文的方法名和缩写，不写原论文的结果；结果框写任务书指标。
# 图内中文：标题黑体（FandolHei），其余宋体（FandolSong）；数字与字母 Times New Roman（指南 §2.13.1）。
# 用法：python -I make_route.py <Graduate_Design 目录>   需要 rsvg-convert，以及 FONTCONFIG_FILE 指向含上述字体的配置
import json, os, subprocess, sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(sys.argv[1])
MID = "--mid" in sys.argv          # 中期报告版：栏标题不写论文章号，输出 route_mid 到 midterm_rebuild/figures
OUT = ROOT / "docs" / ("midterm_rebuild" if MID else "thesis_build") / "figures"
NAME = "route_mid" if MID else "route"
RSVG = os.environ.get("RSVG", "rsvg-convert")
SONG, HEI, TNR = "FandolSong", "FandolHei", "Times New Roman"
W, H = 1500, 900
INK, LINE = "#1f2937", "#475569"
COL = [("#e8f0fb", "#2f5d9c"), ("#e7f4f1", "#2a7a6b"), ("#f3ecf8", "#6b4c9a")]   # 三栏底色 / 边框

svg = []


def rect(x, y, w, h, fill="#ffffff", stroke=LINE, sw=1.6, r=8, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=21, font=SONG, anchor="middle", color=INK, weight="normal"):
    """s 中 _{...} 为下标、^{...} 为上标；拉丁字母和数字用 Times New Roman（斜体用 *x*）"""
    import re
    parts = []
    for tok in re.split(r"(\*[^*]+\*|_\{[^}]*\}|\^\{[^}]*\})", s):
        if not tok:
            continue
        if tok.startswith("*"):
            parts.append(f'<tspan font-family="{TNR}" font-style="italic">{escape(tok[1:-1])}</tspan>')
        elif tok.startswith("_{"):
            parts.append(f'<tspan font-family="{TNR}" font-size="{size * 0.68:.0f}" baseline-shift="sub">{escape(tok[2:-1])}</tspan>')
        elif tok.startswith("^{"):
            parts.append(f'<tspan font-family="{TNR}" font-size="{size * 0.68:.0f}" baseline-shift="super">{escape(tok[2:-1])}</tspan>')
        else:
            for seg in re.split(r"([A-Za-z0-9@.,:;%×=+\-/()<>←→≥ ]+)", tok):
                if not seg:
                    continue
                f = TNR if re.fullmatch(r"[A-Za-z0-9@.,:;%×=+\-/()<>←→≥ ]+", seg) else font
                parts.append(f'<tspan font-family="{f}">{escape(seg)}</tspan>')
    svg.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}" font-weight="{weight}" '
               f'xml:space="preserve">{"".join(parts)}</text>')


def box(x, y, w, h, lines, size=20, fill="#ffffff", stroke=LINE, font=SONG, bold_first=False, dash=None):
    rect(x, y, w, h, fill, stroke, dash=dash)
    n = len(lines)
    lh = size * 1.32
    y0 = y + h / 2 - (n - 1) * lh / 2 + size * 0.35
    for k, s in enumerate(lines):
        text(x + w / 2, y0 + k * lh, s, size, HEI if (bold_first and k == 0) else font)


def arrow(x1, y1, x2, y2, color=LINE, sw=2.2):
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" marker-end="url(#ah)"/>')


def column(x, w, title, k):
    fill, stroke = COL[k]
    rect(x, 20, w, H - 40, fill, stroke, sw=2.2, r=14)
    rect(x, 20, w, 62, stroke, stroke, sw=2.2, r=14)
    rect(x, 60, w, 22, stroke, stroke, sw=0, r=0)           # 标题栏下沿取直
    text(x + w / 2, 62, title, 27, HEI, color="#ffffff")


svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
           '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
           f'<path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/></marker></defs>'
           '<rect width="100%" height="100%" fill="#ffffff"/>')

# ---------- 第一栏：数据集与预处理（第2章）----------
X1, W1 = 20, 430
column(X1, W1, "数据集与预处理" if MID else "数据集与预处理（第2章）", 0)
cx = X1 + W1 / 2
box(X1 + 25, 105, W1 - 50, 78, ["PKU-Market-PCB数据集", "693张图像，10块电路板，6类缺陷"], 20, bold_first=True)
arrow(cx, 183, cx, 208)
box(X1 + 25, 208, W1 - 50, 78, ["数据划分", "扩散模型按电路板；检测按照片"], 20, bold_first=True)
arrow(cx, 286, cx, 311)
box(X1 + 25, 311, W1 - 50, 50, ["尺度统一：整幅图像缩小为0.5倍"], 20)
arrow(cx - 95, 361, cx - 95, 393); arrow(cx + 95, 361, cx + 95, 393)
LW = (W1 - 70) / 2
box(X1 + 25, 393, LW, 150, ["缺陷局部块", "64×64", "以缺陷框为中心", "共2185个"], 19, bold_first=True)
box(X1 + 45 + LW, 393, LW, 150, ["无缺陷背景块", "256×256", "模板去除缺陷", "网格截取并检查"], 19, bold_first=True)
arrow(cx - 95, 543, cx - 95, 575); arrow(cx + 95, 543, cx + 95, 575)
box(X1 + 25, 575, LW, 98, ["缺陷先验模型", "以缺陷类别为条件"], 19, fill="#dbe7f7", bold_first=True)
box(X1 + 45 + LW, 575, LW, 98, ["背景合成模型", "无条件"], 19, fill="#dbe7f7", bold_first=True)
box(X1 + 25, 700, W1 - 50, 100, ["检测数据", "训练每类10～60张 + 50张无缺陷", "选择集300张；测试集300张"], 19, bold_first=True)
text(cx, 838, "两个扩散模型采用公开的预训练权重", 18, SONG, color="#475569")

# ---------- 第二栏：协同引导生成（第3章）----------
X2, W2 = 500, 520
column(X2, W2, "双扩散模型协同引导生成" if MID else "双扩散模型协同引导生成（第3章）", 1)
cx2 = X2 + W2 / 2
box(X2 + 25, 105, W2 - 50, 78, ["输入：无缺陷图像*x*_{free}、缺陷类别、缺陷位置", "以缺陷位置为中心确定64×64窗口"], 20)
arrow(cx2, 183, cx2, 208)
box(X2 + 25, 208, W2 - 50, 52, ["加噪到中等噪声水平，再逐步去噪"], 20)
arrow(cx2, 260, cx2, 290)
rect(X2 + 20, 290, W2 - 40, 410, "#f6fbfa", COL[1][1], sw=1.6, r=10, dash="7,5")
text(cx2, 318, "每个去噪步", 20, HEI, color=COL[1][1])
HW = (W2 - 90) / 2
box(X2 + 35, 335, HW, 92, ["缺陷先验模型", "估计窗口内缺陷"], 19, fill="#ffffff", bold_first=True)
box(X2 + 55 + HW, 335, HW, 92, ["背景合成模型", "估计完整图像"], 19, fill="#ffffff", bold_first=True)
arrow(X2 + 35 + HW / 2, 427, X2 + 35 + HW / 2, 452); arrow(X2 + 55 + 1.5 * HW, 427, X2 + 55 + 1.5 * HW, 452)
box(X2 + 35, 452, W2 - 70, 70, ["注意力掩码：由两者差异计算", "锐化、中心加权、连通域筛选、滑动平均"], 18, fill="#fdf3e1", stroke="#c98a2b")
arrow(cx2, 522, cx2, 547)
box(X2 + 35, 547, W2 - 70, 70, ["引导损失：缺陷损失 + 窗口内、外背景损失", "缺陷跟随先验，其余区域保持不变"], 18, fill="#f1ebf8", stroke="#7a5aa8")
arrow(cx2, 617, cx2, 642)
box(X2 + 35, 642, W2 - 70, 46, ["对*x*_{t}梯度下降，再做一步DDIM更新"], 18)
svg.append(f'<path d="M {X2 + 35} 665 C {X2 + 5} 665, {X2 + 5} 381, {X2 + 35} 381" fill="none" stroke="{LINE}" stroke-width="2" marker-end="url(#ah)"/>')
arrow(cx2, 700, cx2, 728)
box(X2 + 25, 728, W2 - 50, 72, ["输出：带缺陷的生成图像", "标签框取该位置真实缺陷框的大小"], 20, fill="#dcefea", bold_first=True)
text(cx2, 838, "生成质量：KID、FID、生成精确率/召回率", 18, SONG, color="#475569")

# ---------- 第三栏：检测实验与性能提升（第4、5章）----------
X3, W3 = 1070, 410
column(X3, W3, "检测验证与性能提升" if MID else "检测实验（第4、5章）", 2)
cx3 = X3 + W3 / 2
box(X3 + 25, 105, W3 - 50, 78, ["生成数据加入训练集", "1倍、2倍、3倍扩增"], 20, bold_first=True)
arrow(cx3, 183, cx3, 208)
box(X3 + 25, 208, W3 - 50, 78, ["检测模型YOLO11n", "输入640，200 epoch，3个随机种子"], 20, bold_first=True)
arrow(cx3, 286, cx3, 311)
box(X3 + 25, 311, W3 - 50, 104, ["对比与分析", "其他生成方法；真实数据量", "（每类10～60张）；消融实验"], 19, bold_first=True)
arrow(cx3, 415, cx3, 440)
box(X3 + 25, 440, W3 - 50, 78, ["检测性能提升", "上下翻转增强；多模型平均" if MID else "翻转与90°旋转增强；多模型平均"], 20, bold_first=True)
arrow(cx3, 518, cx3, 543)
box(X3 + 25, 543, W3 - 50, 78, ["评价：选择集选取设置", "测试集报告精确率、召回率、mAP"], 19)
arrow(cx3, 621, cx3, 645)
rect(X3 + 25, 645, W3 - 50, 160, "#fff7e0", "#c98a2b", sw=2.2)
text(cx3, 682, "任务书指标", 23, HEI, color="#8a5a12")
text(cx3, 722, "精确率 ≥ 90%", 21, SONG)
text(cx3, 754, "召回率 ≥ 90%", 21, SONG)
text(cx3, 786, "mAP@50-95 ≥ 0.5", 21, SONG)
text(cx3, 838, "测试集只用于最终评价", 18, SONG, color="#475569")

# 栏间箭头
for xa, xb in ((X1 + W1 + 6, X2 - 6), (X2 + W2 + 6, X3 - 6)):
    svg.append(f'<path d="M {xa} 430 L {xb - 16} 430 L {xb - 16} 414 L {xb} 445 L {xb - 16} 476 L {xb - 16} 460 L {xa} 460 z" fill="#3b6fb6"/>')
svg.append("</svg>")

src = OUT / f"{NAME}.svg"
src.write_text("\n".join(svg), encoding="utf-8")
png = OUT / "_route.png"
subprocess.run([RSVG, "-w", str(W * 2), "-o", str(png), str(src)], check=True)
from PIL import Image
im = Image.open(png).convert("RGB"); im.save(OUT / f"{NAME}.jpg", quality=95, dpi=(300, 300)); png.unlink()
sizes = json.load(open(OUT / "sizes.json"))
sizes[NAME] = {"px": list(im.size), "pt": [im.size[0] * 72 / 300, im.size[1] * 72 / 300]}
json.dump(sizes, open(OUT / "sizes.json", "w"), indent=1)
print("route", im.size)
