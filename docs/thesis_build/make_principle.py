# 协同引导原理图（2026-11-27，导师要求中期报告补原理图）：素材来自 code/experiments/capture_guidance.py 的一次真实运行（j0021，短路）
#   fig_principle：一个去噪步内的计算流程，节点处为该步的真实中间结果
#   fig_mask_steps：去噪过程中背景模型估计、缺陷模型估计与注意力掩码随时间步的变化
# 风格与 make_route.py 一致：标题黑体，其余宋体，数字与字母 Times New Roman
# 用法：FONTCONFIG_FILE=tools/libreoffice/fonts/fonts.conf python -I make_principle.py <Graduate_Design 目录>
import base64, io, json, os, re, subprocess, sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image

ROOT = Path(sys.argv[1])
OUT = ROOT / "docs" / "thesis_build" / "figures"
RSVG = os.environ.get("RSVG", "rsvg-convert")
SONG, HEI, TNR = "FandolSong", "FandolHei", "Times New Roman"
INK, LINE = "#1f2937", "#475569"
d = np.load(ROOT / "results" / "experiments" / "principle" / "j0021.npz")
lx, ly = map(int, d["loc"]); P = 64
T = d["t"]


def rgb(x):
    return Image.fromarray(((np.clip(x, -1, 1) + 1) * 127.5).round().astype(np.uint8).transpose(1, 2, 0))


def gray(a):
    a = a[0] if a.ndim == 3 else a
    return Image.fromarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8)).convert("RGB")


def uri(im, size):
    im = im.resize((size, size), Image.NEAREST if im.width <= 64 else Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def win(x):
    return x[:, ly:ly + P, lx:lx + P]


class SVG:
    def __init__(self, w, h):
        self.w, self.h, self.s = w, h, []
        self.s.append(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{w}" height="{h}" '
                      f'viewBox="0 0 {w} {h}"><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                      f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/></marker></defs>'
                      '<rect width="100%" height="100%" fill="#ffffff"/>')

    def text(self, x, y, s, size=20, font=SONG, anchor="middle", color=INK):
        """s 中 _{...} 下标、^{...} 上标、*x* 斜体；拉丁字母和数字用 Times New Roman"""
        parts = []
        for tok in re.split(r"(#[a-z]#|\*[^*]+\*|_\{[^}]*\}|\^\{[^}]*\})", s):
            if not tok:
                continue
            if tok.startswith("#"):
                parts.append(f'<tspan font-family="{TNR}" font-style="italic">{tok[1]}</tspan>'
                             f'<tspan font-family="{TNR}" dx="-0.36em" dy="-0.30em">ˆ</tspan>'
                             f'<tspan font-family="{TNR}" dx="-0.04em" dy="0.30em">\u200a</tspan>')
            elif tok.startswith("*"):
                parts.append(f'<tspan font-family="{TNR}" font-style="italic">{escape(tok[1:-1])}</tspan>')
            elif tok[:2] in ("_{", "^{"):
                sh = "sub" if tok[0] == "_" else "super"
                inner = tok[2:-1]
                st = ' font-style="italic"' if inner.startswith("*") else ""
                parts.append(f'<tspan font-family="{TNR}"{st} font-size="{size * 0.68:.0f}" baseline-shift="{sh}">{escape(inner.strip("*"))}</tspan>')
            else:
                for seg in re.split(r"([A-Za-z0-9@.,:;%×=+\-/()<>←→≥ ηλ∇^]+)", tok):
                    if seg:
                        f = TNR if re.fullmatch(r"[A-Za-z0-9@.,:;%×=+\-/()<>←→≥ ηλ∇^]+", seg) else font
                        parts.append(f'<tspan font-family="{f}">{escape(seg)}</tspan>')
        self.s.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}" xml:space="preserve">{"".join(parts)}</text>')

    def rect(self, x, y, w, h, fill="#ffffff", stroke=LINE, sw=1.6, r=8, dash=None):
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        self.s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dd}/>')

    def img(self, x, y, im, size, border="#94a3b8"):
        self.s.append(f'<image x="{x}" y="{y}" width="{size}" height="{size}" xlink:href="{uri(im, size * 2)}"/>')
        self.s.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="none" stroke="{border}" stroke-width="1.2"/>')

    def arrow(self, *pts, color=LINE, sw=2.2):
        p = " ".join(f"{a},{b}" for a, b in pts)
        self.s.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}" marker-end="url(#ah)"/>')

    def save(self, name, width_px):
        self.s.append("</svg>")
        src = OUT / f"{name}.svg"; src.write_text("\n".join(self.s), encoding="utf-8")
        png = OUT / f"_{name}.png"
        subprocess.run([RSVG, "-w", str(width_px), "-o", str(png), str(src)], check=True)
        im = Image.open(png).convert("RGB"); im.save(OUT / f"{name}.jpg", quality=95, dpi=(300, 300)); png.unlink()
        sizes = json.load(open(OUT / "sizes.json"))
        sizes[name] = {"px": list(im.size), "pt": [im.size[0] * 72 / 300, im.size[1] * 72 / 300]}
        json.dump(sizes, open(OUT / "sizes.json", "w"), indent=1)
        print(name, im.size)


# 图中文字统一为小五号（9 pt，写作指南 §2.13.1：字号一致，中文宋体，数字与字母 Times New Roman）。
# 字号 F（SVG 像素）按插入宽度换算：F = 9 pt × SVG 宽度 / 插入宽度（pt）。插入宽度：原理图 430 pt，掩码图 420 pt。
from PIL import ImageDraw
GREEN, BLUE, ORANGE, PURPLE = ("#e7f4f1", "#2a7a6b"), ("#e8f0fb", "#2f5d9c"), ("#fdf3e1", "#c98a2b"), ("#f1ebf8", "#7a5aa8")
MUTED = "#475569"

# ===================== 图 1：一个去噪步内的协同引导 =====================
k = int(np.where(T == 15)[0][0])            # 第一次经过 t=15 的去噪步（中等噪声水平，掩码已经成形）
free, x0, c0, ad, ab = d["free"], d["x0"][k], d["c0"][k], d["ad"][k], d["ab"][k]
W1 = 1100; F = round(9 * W1 / 430)           # 23
s = SVG(W1, 770)
S = 110                                       # 缩略图边长
T_ = lambda x, y, t, **kw: s.text(x, y, t, F, SONG, **kw)

# 左：输入
s.rect(10, 10, 180, 750, "#f8fafc", "#94a3b8", sw=1.4, r=10)
T_(100, 42, "输入")
im_free = rgb(free).copy()
ImageDraw.Draw(im_free).rectangle([lx, ly, lx + P - 1, ly + P - 1], outline=(255, 215, 0), width=2)
s.img(45, 60, im_free, S); T_(100, 198, "无缺陷图像"); T_(100, 226, "*x*_{free}")
s.rect(30, 248, 140, 42, "#ffffff"); T_(100, 277, "类别：短路")
s.arrow((100, 290), (100, 314))
s.rect(30, 314, 140, 42, "#ffffff"); T_(100, 343, "加噪得到*x*_{t}")
s.arrow((100, 356), (100, 380))
s.img(45, 380, rgb(d["noised"]), S); T_(100, 518, "*t* = 29")
s.arrow((190, 435), (218, 435))

# 中：一个去噪步
s.rect(220, 10, 680, 750, "#f6fbfa", GREEN[1], sw=1.8, r=12, dash="8,5")
T_(560, 42, "每个去噪步（图中为*t* = 15）", color=GREEN[1])
s.rect(240, 60, 250, 70, BLUE[0], BLUE[1]); T_(365, 89, "背景合成模型"); T_(365, 118, "估计完整图像")
s.rect(630, 60, 250, 70, BLUE[0], BLUE[1]); T_(755, 89, "缺陷先验模型"); T_(755, 118, "估计窗口内缺陷")
s.arrow((365, 130), (365, 150)); s.arrow((755, 130), (755, 150))
s.img(310, 150, rgb(x0), S); T_(365, 288, "#x#_{0}")
s.img(700, 150, rgb(c0), S); T_(755, 288, "#c#_{0}")
s.img(505, 150, rgb(win(free)), S); T_(560, 288, "*r*(*x*_{free})")
s.arrow((700, 222), (617, 222)); T_(658, 205, "作差", color=MUTED)
s.arrow((560, 296), (560, 316))
s.rect(390, 316, 340, 76, ORANGE[0], ORANGE[1]); T_(560, 345, "注意力掩码"); T_(560, 375, "锐化、加权、连通域、滑动平均")
s.arrow((560, 392), (560, 412))
s.img(443, 412, gray(ad), S); T_(498, 550, "*A*_{d}")
s.img(567, 412, gray(ab), S); T_(622, 550, "*A*_{b}")
s.arrow((365, 296), (365, 590)); s.arrow((755, 296), (755, 590)); s.arrow((560, 558), (560, 590))
s.rect(250, 590, 620, 80, PURPLE[0], PURPLE[1])
T_(560, 622, "*L* = λ_{d}*L*_{d} + λ_{b}*L*_{b} + λ_{o}*L*_{o}")
T_(560, 654, "*L*_{d}：缺陷跟随#c#_{0}；*L*_{b}、*L*_{o}：背景保持*x*_{free}")
s.arrow((560, 670), (560, 692))
s.rect(330, 692, 460, 46, "#ffffff"); T_(560, 723, "*x*_{t} ← *x*_{t} − η∇*L*，再做一步DDIM")
s.s.append(f'<path d="M 330 715 C 228 715, 228 95, 240 95" fill="none" stroke="{LINE}" stroke-width="2" stroke-dasharray="6,4" marker-end="url(#ah)"/>')
T_(262, 470, "*t* ← *t* − 1", anchor="start", color=MUTED)

# 右：输出
s.rect(920, 10, 170, 750, "#f8fafc", "#94a3b8", sw=1.4, r=10)
T_(1005, 42, "输出")
im_res = rgb(d["result"]).copy()
ImageDraw.Draw(im_res).rectangle([lx, ly, lx + P - 1, ly + P - 1], outline=(255, 215, 0), width=2)
s.img(950, 60, im_res, S); T_(1005, 198, "生成图像")
s.img(950, 222, rgb(win(d["result"])), S); T_(1005, 360, "窗口放大")
s.rect(928, 692, 154, 46, "#ffffff"); T_(1005, 723, "*t* = 0时")
s.arrow((790, 715), (928, 715))
s.arrow((1005, 692), (1005, 372))
s.save("fig_principle", 3300)

# ===================== 图 2：估计与掩码随时间步的变化 =====================
last = {}
for i, t in enumerate(T):                      # 每个时间步取最后一次经过时的结果（回跳后会再次经过）
    last[int(t)] = i
show = [29, 25, 20, 15, 10, 5, 1]
cols = [last[t] for t in show]
S2, G2, X0 = 100, 10, 150
W2 = X0 + len(show) * (S2 + G2); F2 = round(9 * W2 / 420)       # 20
s = SVG(W2, 40 + 3 * (S2 + 14) + 4)
s.text(X0 - 16, 27, "*t*", F2, SONG, anchor="end")
for c, t in enumerate(show):
    s.text(X0 + c * (S2 + G2) + S2 / 2, 27, str(t), F2, SONG)
rows = [("#x#_{0}（窗口）", lambda i: rgb(win(d["x0"][i]))), ("#c#_{0}", lambda i: rgb(d["c0"][i])),
        ("*A*_{d}", lambda i: gray(d["ad"][i]))]
for r, (name, f) in enumerate(rows):
    y = 40 + r * (S2 + 14)
    s.text(X0 - 16, y + S2 / 2 + 7, name, F2, SONG, anchor="end")
    for c, i in enumerate(cols):
        s.img(X0 + c * (S2 + G2), y, f(i), S2)
s.save("fig_mask_steps", 2400)
