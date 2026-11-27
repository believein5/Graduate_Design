# 毕业答辩 PPT（2026-11-27）：内容全部取自毕业论文（docs/thesis_build/content.py），版式与中期汇报 PPT（make_ppt.py）一致。
# 图取自 docs/thesis_build/figures；公式用 matplotlib mathtext 渲染为 PNG；趋势图为可编辑的原生折线图。
# 每页附讲稿（备注）。16:9，中文微软雅黑，数字与字母 Arial。数值范围用短横“–”（2026-11-27 用户：“～”显示不对）；封面不写日期。
# 用法：tools/pptx-venv/bin/python make_ppt_final.py <Graduate_Design 目录> <输出 .pptx>
import io, re, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from PIL import Image

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2])
FIG = ROOT / "docs" / "thesis_build" / "figures"
EA, LAT = "微软雅黑", "Arial"
NAVY, BLUE, INK, GREY, LIGHT, PALE, GREEN = (RGBColor(0x0B, 0x3C, 0x7A), RGBColor(0x2A, 0x78, 0xD6), RGBColor(0x1F, 0x29, 0x37),
                                            RGBColor(0x64, 0x74, 0x8B), RGBColor(0xE2, 0xE8, 0xF0), RGBColor(0xF1, 0xF5, 0xFB),
                                            RGBColor(0x1B, 0x7F, 0x5B))
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
SECTIONS = ["研究背景与内容", "数据与预处理", "双扩散协同引导生成", "生成数据增强的检测实验", "检测性能的提升", "结论"]


def font(run, size, bold=False, color=INK):
    run.font.size = Pt(size); run.font.bold = bold; run.font.color.rgb = color; run.font.name = LAT
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        e = rpr.find(qn(tag))
        if e is None:
            e = rpr.makeelement(qn(tag), {}); rpr.append(e)
        e.set("typeface", EA)


def text(slide, x, y, w, h, lines, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.15):
    """lines：字符串或 (字符串, 选项) 列表；以“• ”开头的行按项目符号排版；**…** 加粗"""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame
    tf.word_wrap = True; tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, Inches(0.04))
    for k, ln in enumerate([lines] if isinstance(lines, str) else lines):
        opt = {}
        if isinstance(ln, tuple):
            ln, opt = ln
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.alignment = opt.get("align", align); p.line_spacing = spacing
        p.space_after = Pt(opt.get("after", 6))
        bullet = ln.startswith("• ")
        if bullet:
            ln = ln[2:]; p.level = 0
            r = p.add_run(); r.text = "■  "; font(r, opt.get("size", size) * 0.55, False, opt.get("bcolor", BLUE))
        for j, seg in enumerate(ln.split("**")):
            for piece in re.split(r"(_\{[^}]*\})", seg):
                if not piece:
                    continue
                r = p.add_run(); r.text = piece[2:-1] if piece.startswith("_{") else piece
                font(r, opt.get("size", size), bold or j % 2 == 1 or opt.get("bold", False), opt.get("color", color))
                if piece.startswith("_{"):
                    r._r.get_or_add_rPr().set("baseline", "-25000")
    return tb


def rect(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line; s.line.width = Pt(1)
    s.shadow.inherit = False
    return s


def image(slide, path, x, y, w, h, align="center"):
    """按比例放入 (x, y, w, h) 框内"""
    im = Image.open(path); r = im.width / im.height
    ww, hh = (w, w / r) if w / r <= h else (h * r, h)
    xx = x + (w - ww) / 2 if align == "center" else x
    return slide.shapes.add_picture(str(path), Inches(xx), Inches(y + (h - hh) / 2), Inches(ww), Inches(hh))


def caption(slide, x, y, w, s):
    text(slide, x, y, w, 0.35, s, 12, color=GREY, align=PP_ALIGN.CENTER)


def formula(tex, size=22):
    fig = plt.figure(figsize=(0.01, 0.01)); fig.text(0, 0, f"${tex}$", fontsize=size, color="#1f2937")
    plt.rcParams["mathtext.fontset"] = "stix"
    buf = io.BytesIO(); fig.savefig(buf, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.04); plt.close(fig)
    buf.seek(0); p = OUT.parent / f"_f{abs(hash(tex))}.png"; p.write_bytes(buf.read()); return p


def table(slide, x, y, w, rows, widths=None, size=13, hl_rows=(), hl_col=None, row_h=0.36):
    nr, nc = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr)); t = shp.table
    tblPr = shp._element.graphic.graphicData.tbl.tblPr
    tblPr.set("firstRow", "0"); tblPr.set("bandRow", "0")
    widths = widths or [w / nc] * nc
    for j, cw in enumerate(widths):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, v in enumerate(row):
            c = t.cell(i, j); c.margin_left = c.margin_right = Inches(0.06); c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid(); c.fill.fore_color.rgb = NAVY if i == 0 else (PALE if i in hl_rows else RGBColor(0xFF, 0xFF, 0xFF))
            tf = c.text_frame; tf.paragraphs[0].text = ""
            p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
            for piece in re.split(r"(_\{[^}]*\})", str(v)):            # _{...} 为下标
                if piece:
                    r = p.add_run(); r.text = piece[2:-1] if piece.startswith("_{") else piece
                    font(r, size, i == 0 or i in hl_rows, RGBColor(0xFF, 0xFF, 0xFF) if i == 0 else INK)
                    if piece.startswith("_{"):
                        r._r.get_or_add_rPr().set("baseline", "-25000")
    return shp


def kpi(slide, x, y, w, h, value, label, sub=None, color=NAVY):
    rect(slide, x, y, w, h, PALE)
    rect(slide, x, y, 0.08, h, color)
    text(slide, x + 0.2, y + 0.12, w - 0.3, 0.7, value, 32, True, color)
    text(slide, x + 0.2, y + 0.82, w - 0.3, 0.4, label, 15, color=INK)
    if sub:
        text(slide, x + 0.2, y + 1.18, w - 0.3, 0.4, sub, 12, color=GREY)


def page(title, section=None, notes=""):
    s = prs.slides.add_slide(BLANK)
    rect(s, 0.45, 0.42, 0.1, 0.55, NAVY)
    text(s, 0.68, 0.33, 10.5, 0.75, title, 28, True, NAVY, anchor=MSO_ANCHOR.MIDDLE)
    if section:
        text(s, 9.3, 0.42, 3.6, 0.5, f"{SECTIONS.index(section) + 1:02d}  {section}", 13, color=GREY, align=PP_ALIGN.RIGHT,
             anchor=MSO_ANCHOR.MIDDLE)
    rect(s, 0.45, 1.12, 12.43, 0.015, LIGHT)
    rect(s, 0.45, 7.0, 12.43, 0.012, LIGHT)
    text(s, 0.45, 7.03, 8, 0.35, "哈尔滨工业大学（深圳）  本科毕业论文（设计）答辩", 11, color=GREY)
    text(s, 11.9, 7.03, 0.98, 0.35, str(len(prs.slides)), 11, color=GREY, align=PP_ALIGN.RIGHT)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s




WHITE, SOFT = RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0xBF, 0xD6, 0xF6)


def line_chart(s, x, y, w, h, cats, series, vmin, vmax, unit, xtitle, ytitle):
    """series：[(名称, 数值, 颜色, 虚线?)]"""
    cd = CategoryChartData(); cd.categories = cats
    for name, vals, _, _ in series:
        cd.add_series(name, vals)
    ch = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart
    ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(13)
    va, ca = ch.value_axis, ch.category_axis
    va.minimum_scale, va.maximum_scale, va.major_unit = vmin, vmax, unit
    va.has_major_gridlines = True; va.major_gridlines.format.line.color.rgb = LIGHT
    va.tick_labels.number_format = "0.00"; va.tick_labels.number_format_is_linked = False
    for ax, t in ((ca, xtitle), (va, ytitle)):
        ax.has_title = True; ax.axis_title.text_frame.text = t; ax.tick_labels.font.size = Pt(12)
        for p in ax.axis_title.text_frame.paragraphs:
            for r in p.runs:
                font(r, 13, False, GREY)
    marks = (XL_MARKER_STYLE.SQUARE, XL_MARKER_STYLE.CIRCLE, XL_MARKER_STYLE.TRIANGLE, XL_MARKER_STYLE.DIAMOND)
    for k, (ser, (_, _, col, dash)) in enumerate(zip(ch.plots[0].series, series)):
        ser.smooth = False; ser.format.line.color.rgb = col; ser.format.line.width = Pt(2.5)
        if dash:
            ser.format.line.dash_style = 4
        ser.marker.style = marks[k % 4]; ser.marker.size = 8
        ser.marker.format.fill.solid(); ser.marker.format.fill.fore_color.rgb = col; ser.marker.format.line.color.rgb = col
    return ch


def bullets(s, x, y, w, h, items, size=15):
    return text(s, x, y, w, h, ["• " + t for t in items], size)


def head(s, x, y, w, t, color=NAVY, size=18):
    text(s, x, y, w, 0.5, t, size, True, color)


# ======================= 1 封面 =======================
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, 13.333, 7.5, NAVY)
rect(s, 0, 5.05, 13.333, 2.45, RGBColor(0x08, 0x2E, 0x5E))
rect(s, 0.9, 1.55, 0.12, 1.75, RGBColor(0x7F, 0xB2, 0xF0))
text(s, 1.25, 1.35, 11.2, 0.6, "本科毕业论文（设计）答辩", 20, color=SOFT)
text(s, 1.25, 2.0, 11.9, 1.2, "基于扩散模型的印刷电路板缺陷图像生成方法研究", 34, True, WHITE)
for k, (a, b) in enumerate([("答辩人", "陈斯萍（2023313823）"), ("专    业", "机器人工程"), ("指导教师", "王睿 副教授"),
                            ("学    院", "深圳校区机器人与先进制造学院")]):
    yy = 5.35 + (k // 2) * 0.62; xx = 1.25 + (k % 2) * 5.6
    text(s, xx, yy, 1.4, 0.5, a, 16, color=SOFT); text(s, xx + 1.45, yy, 4.0, 0.5, b, 16, color=WHITE)
s.notes_slide.notes_text_frame.text = ("各位老师好，我是机器人工程专业的陈斯萍，指导教师是王睿老师。我的毕业论文题目是基于扩散模型的印刷电路板缺陷"
                                       "图像生成方法研究。下面从六个方面汇报。")

# ======================= 2 提纲 =======================
s = page("汇报提纲", notes="汇报分六部分：研究背景与内容、数据与预处理、双扩散协同引导生成、生成数据增强的检测实验、检测性能的提升，最后是结论。")
desc = ["缺陷样本少的问题；研究现状；研究内容与技术路线", "数据集划分；缺陷局部块；基于模板的无缺陷背景块",
        "扩散模型原理；双扩散模型；协同引导机制与算法", "与其他方法对比；真实数据量；扩增倍率；消融实验",
        "输入尺寸；翻转与90°旋转增强；多模型平均；任务书指标", "主要结论；不足与展望"]
for k, (a, b) in enumerate(zip(SECTIONS, desc)):
    col, row = k % 2, k // 2
    x, y = 0.9 + col * 6.1, 1.6 + row * 1.7
    rect(s, x, y, 0.85, 0.85, NAVY if (k + row) % 2 == 0 else BLUE)
    text(s, x, y, 0.85, 0.85, f"{k + 1:02d}", 26, True, WHITE, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    text(s, x + 1.1, y - 0.02, 4.8, 0.5, a, 21, True, NAVY)
    text(s, x + 1.1, y + 0.48, 4.8, 0.8, b, 14, color=GREY)

# ======================= 3 背景 =======================
s = page("研究背景与意义", SECTIONS[0], notes=(
    "PCB是电子设备的基础，制造中可能出现缺孔、鼠咬、开路、短路、毛刺和多余铜箔六类缺陷。深度学习检测需要大量带标注的缺陷图像，但实际生产中"
    "缺陷少、标注贵。旋转翻转等常规增强不能产生新的缺陷形态，扩散模型可以学习数据分布、生成新的缺陷图像。本课题研究用扩散模型生成缺陷图像扩充"
    "检测训练集，并在检测任务中评价生成数据的作用。任务书要求精确率、召回率90%以上，mAP@50-95达到0.5。"))
image(s, FIG / "op_defects.jpg", 0.55, 1.35, 5.4, 4.5)
caption(s, 0.55, 5.85, 5.4, "PCB六类典型缺陷：缺孔、鼠咬、开路、短路、毛刺、多余铜箔")
head(s, 6.4, 1.4, 6.5, "问题")
bullets(s, 6.4, 1.9, 6.5, 1.8, ["检测模型需要大量带标注的缺陷图像", "生产中缺陷出现概率低，采集、标注成本高",
                                 "常规增强（旋转、翻转、裁剪）不产生新的缺陷形态"], 16)
head(s, 6.4, 3.45, 6.5, "思路")
bullets(s, 6.4, 3.95, 6.5, 1.3, ["用扩散模型生成带标注的缺陷图像，扩充检测训练集", "以检测指标评价生成数据是否有用、在什么条件下有用"], 16)
head(s, 6.4, 5.05, 6.5, "任务书指标")
for k, (v, l) in enumerate([("≥ 90%", "精确率"), ("≥ 90%", "召回率"), ("≥ 0.5", "mAP@50-95")]):
    rect(s, 6.4 + k * 2.2, 5.55, 2.05, 1.2, PALE); rect(s, 6.4 + k * 2.2, 5.55, 0.07, 1.2, NAVY)
    text(s, 6.6 + k * 2.2, 5.6, 1.8, 0.6, v, 26, True, NAVY); text(s, 6.6 + k * 2.2, 6.2, 1.8, 0.4, l, 14)

# ======================= 4 研究现状 =======================
s = page("国内外研究现状", SECTIONS[0], notes=(
    "检测方面，从手工特征发展到卷积网络和Transformer，主要改进网络结构，但都受训练数据规模限制。扩散模型方面，DDPM建立了基本框架，DDIM减少了"
    "采样步数，后续发展到潜在空间。缺陷生成方面，RePaint、CoPaint等修复方法、ControlNet条件生成，以及AnomalyDiffusion、DefFiller等工业缺陷生成，"
    "PCB上有双扩散的生成框架。现有不足是：面向PCB的研究少，生成图像看起来真实不等于对检测有用，需要在检测任务中验证并分析在什么条件下有效。"))
cols = [("PCB缺陷检测", ["手工特征：模板匹配、形态学", "卷积网络：TDD-Net、多尺度特征融合", "Transformer与注意力机制",
                         "共同限制：训练数据规模与类别不平衡"]),
        ("扩散模型", ["DDPM：前向加噪、反向去噪", "DDIM：确定性采样，减少步数", "改进DDPM、分类器引导", "潜在扩散模型（LDM）"]),
        ("缺陷图像生成", ["图像修复：SDEdit、RePaint、CoPaint", "条件生成：ControlNet", "工业缺陷：AnomalyDiffusion、DefFiller",
                          "PCB：双扩散生成框架"])]
for k, (t, it) in enumerate(cols):
    x = 0.6 + k * 4.15
    rect(s, x, 1.45, 3.9, 0.6, NAVY if k != 2 else BLUE)
    text(s, x, 1.45, 3.9, 0.6, t, 18, True, WHITE, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    rect(s, x, 2.05, 3.9, 2.9, PALE)
    bullets(s, x + 0.2, 2.25, 3.6, 2.6, it, 14)
rect(s, 0.6, 5.25, 12.2, 1.5, RGBColor(0xFD, 0xF3, 0xE1)); rect(s, 0.6, 5.25, 0.08, 1.5, RGBColor(0xC9, 0x8A, 0x2B))
text(s, 0.9, 5.33, 11.8, 0.45, "存在的问题", 17, True, RGBColor(0x8A, 0x5A, 0x12))
text(s, 0.9, 5.8, 11.8, 0.9, "面向PCB缺陷的生成研究较少；生成图像看起来真实并不等于能提高检测性能——生成数据的作用需要在统一的检测任务中验证，"
                             "并分析它在什么条件下有效。", 15)

# ======================= 5 研究内容与技术路线 =======================
s = page("研究内容与技术路线", SECTIONS[0], notes=(
    "研究内容分四部分，对应论文第2到5章：数据整理与预处理，双扩散模型搭建与协同引导生成，生成数据增强的检测实验，以及检测性能的提升。"
    "技术路线图三栏分别对应数据、生成和检测。"))
image(s, FIG / "route.jpg", 0.5, 1.3, 8.9, 5.6, align="left")
for k, (a, b) in enumerate([("第2章", "数据整理与预处理"), ("第3章", "双扩散模型与协同引导生成"), ("第4章", "生成数据增强的检测实验"),
                            ("第5章", "检测性能的提升")]):
    y = 1.55 + k * 1.3
    rect(s, 9.7, y, 3.2, 1.1, PALE); rect(s, 9.7, y, 0.07, 1.1, NAVY)
    text(s, 9.95, y + 0.08, 2.9, 0.4, a, 14, True, BLUE); text(s, 9.95, y + 0.45, 2.9, 0.6, b, 15, True, INK)

# ======================= 6 数据集与划分 =======================
s = page("数据集与划分", SECTIONS[1], notes=(
    "数据集是PKU-Market-PCB，10块电路板693张图像，6类缺陷基本均衡。同一块板的图像背景几乎相同，所以扩散模型按电路板划分，10到12号板不参与训练；"
    "检测实验按照片划分为训练、选择、测试三组，选择集只用来选设置，测试集只用于最终评价。训练子集取每类10到60张，10和20张模拟缺陷样本很少的情况。"))
table(s, 0.6, 1.5, 6.2, [["缺陷类别", "图像数", "缺陷数", "电路板01、04–09"], ["缺孔", "115", "497", "370"], ["鼠咬", "115", "492", "366"],
                         ["开路", "116", "482", "351"], ["短路", "116", "491", "365"], ["毛刺", "115", "488", "362"],
                         ["多余铜箔", "116", "503", "371"], ["合计", "693", "2953", "2185"]], [1.6, 1.2, 1.2, 2.2], 14, hl_rows=(7,), row_h=0.5)
caption(s, 0.6, 5.6, 6.2, "PKU-Market-PCB：10块电路板，693张图像，6类缺陷")
head(s, 7.3, 1.45, 5.6, "扩散模型：按电路板划分", size=19)
bullets(s, 7.3, 1.95, 5.6, 1.2, ["训练：电路板01、04–09（541张）", "评价生成质量：电路板10–12（152张）"], 16)
head(s, 7.3, 3.1, 5.6, "检测实验：按照片划分", size=19)
bullets(s, 7.3, 3.6, 5.6, 3.3, ["训练照片：1044个缺陷；每类50张为基本设置，另加50张无缺陷图像",
                                "选择集300张：只用于选择训练轮次和设置", "测试集300张：只用于最终评价",
                                "每类10–60张的训练子集：考察真实数据量的影响"], 16)

# ======================= 7 预处理 =======================
s = page("训练数据预处理", SECTIONS[1], notes=(
    "先把整幅图缩小为0.5倍，94.7%的缺陷能放进64×64窗口，尺度也和预训练模型一致。缺陷局部块以缺陷框为中心截取，共2185个。无缺陷背景块方面，"
    "同一块板的各图像背景相同、每个缺陷只出现在一张图上，所以逐像素取中值得到模板，用模板替换缺陷区域；再按网格截取，覆盖更均匀，"
    "变异系数从0.096降到0.031，逐像素检查后得到3000个背景块。"))
head(s, 0.6, 1.4, 6, "缺陷局部块（缺陷先验模型）", size=17)
image(s, FIG / "fig_pre_overview.jpg", 0.6, 1.9, 4.1, 2.6)
image(s, FIG / "fig_pre_patches.jpg", 4.85, 1.85, 1.5, 2.75)
bullets(s, 0.6, 4.75, 5.9, 2.2, ["整幅图像缩小为0.5倍：94.7%的缺陷可放入64×64窗口", "以缺陷框为中心加随机偏移截取，共**2185**个",
                                 "115个较大缺陷放大窗口后再缩小，不舍弃"], 14)
head(s, 6.9, 1.4, 6, "无缺陷背景块（背景合成模型）", size=17)
image(s, FIG / "fig_pre_clean.jpg", 6.9, 1.85, 1.9, 2.75)
image(s, FIG / "fig_pre_coverage.jpg", 8.95, 2.0, 3.95, 2.5)
bullets(s, 6.9, 4.75, 6.0, 2.2, ["同板图像逐像素取中值得到模板，替换缺陷区域", "按网格截取：覆盖次数变异系数 **0.096 → 0.031**",
                                 "逐像素检查，共**3000**个256×256背景块"], 14)

# ======================= 8 扩散模型原理 =======================
s = page("扩散模型原理与双扩散模型", SECTIONS[2], notes=(
    "DDPM前向过程逐步加噪，任意时间步可由原图一步得到；网络学习预测噪声。DDIM训练目标相同，采样确定、可跳步，还能把已有图像加噪到中间步再去噪，"
    "只改局部内容，这是协同引导的基础。两个模型都是改进DDPM的U-Net：背景合成模型256×256、无条件；缺陷先验模型64×64、以类别为条件。"))
image(s, FIG / "op_ddpm.jpg", 0.6, 1.3, 7.3, 1.75, align="left")
F = [(r"x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon", "前向加噪"),
     (r"\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}", "估计原图"),
     (r"x_s=\sqrt{\bar\alpha_s}\,\hat x_0+\sqrt{1-\bar\alpha_s}\,\epsilon_\theta(x_t,t)", "DDIM更新")]
for k, (tex, lab) in enumerate(F):
    y = 3.3 + k * 1.2
    rect(s, 0.6, y, 7.3, 1.05, PALE)
    image(s, formula(tex, 20), 0.8, y + 0.08, 5.6, 0.9, align="left")
    text(s, 6.5, y, 1.3, 1.05, lab, 14, color=GREY, anchor=MSO_ANCHOR.MIDDLE)
table(s, 8.3, 1.45, 4.6, [["项目", "背景合成模型", "缺陷先验模型"], ["输入尺寸", "256×256", "64×64"], ["训练数据", "无缺陷背景块", "缺陷局部块"],
                         ["条件", "无", "缺陷类别"], ["基础通道数", "128", "64"], ["训练迭代", "500000", "200000"]],
      [1.3, 1.65, 1.65], 13, row_h=0.45)
bullets(s, 8.3, 4.35, 4.6, 2.5, ["改进DDPM的U-Net，1000步，线性噪声调度", "同时预测噪声和方差", "补充训练程序与独立的DDIM采样程序",
                                 "两个模型都没有复制训练样本（最近邻距离比0.97、1.28）"], 14)

# ======================= 9 单模型采样 =======================
s = page("单模型采样结果", SECTIONS[2], notes=(
    "不用协同引导，单独从两个模型采样。缺陷先验模型能生成六类可辨认的缺陷，每行一类；采样过程显示缺陷结构在中等噪声水平下形成，这也是协同引导"
    "从中等噪声开始去噪的原因。背景合成模型能生成清晰的焊盘、线路和过孔。"))
image(s, FIG / "fig_64_samples.jpg", 0.5, 1.35, 3.6, 5.2); caption(s, 0.5, 6.55, 3.6, "缺陷先验模型（每行一类缺陷）")
image(s, FIG / "fig_64_process.jpg", 4.3, 1.35, 4.2, 5.2); caption(s, 4.3, 6.55, 4.2, "短路采样过程中的原图估计（t: 999 → 0）")
image(s, FIG / "fig_256_samples.jpg", 8.7, 1.35, 4.2, 5.2); caption(s, 8.7, 6.55, 4.2, "背景合成模型")

# ======================= 10 协同引导原理 =======================
s = page("协同引导机制", SECTIONS[2], notes=(
    "生成的输入是无缺陷图像、缺陷类别和位置。先加噪到中等噪声水平，再逐步去噪。每一步背景模型估计完整图像，缺陷模型估计窗口内的缺陷；两者与原背景"
    "的差异得到注意力掩码；引导损失让缺陷区域跟随缺陷模型、其余区域保持原图；对x_t做两次梯度下降后做一步DDIM更新。图中每个缩略图都是一次实际"
    "生成的中间结果。"))
image(s, FIG / "fig_principle.jpg", 0.5, 1.3, 8.4, 5.6, align="left")
head(s, 9.2, 1.4, 3.7, "引导损失", size=19)
image(s, formula(r"L=\lambda_d L_d+\lambda_b L_b+\lambda_o L_o", 20), 9.2, 1.9, 3.6, 0.6, align="left")
text(s, 9.2, 2.65, 3.7, 1.9, ["• L_{d}：A_{d}内图像跟随缺陷估计", "• L_{b}：A_{b}内保持无缺陷背景", "• L_{o}：窗口外保持无缺陷背景"], 14)
head(s, 9.2, 4.3, 3.7, "注意力掩码 A_{d}", size=19)
text(s, 9.2, 4.8, 3.7, 2.1, ["• 由缺陷估计与原背景之差计算", "• Sigmoid锐化、中心加权、连通域筛选", "• 相邻时间步之间指数滑动平均"], 14)

# ======================= 11 算法与掩码公式 =======================
s = page("协同引导生成算法", SECTIONS[2], notes=(
    "完整的生成过程如左表：从第29步开始，按回跳时间表去噪，每10步一段、每段去噪3遍。每一步先由两个模型估计，再更新掩码，做两次梯度下降，"
    "最后DDIM更新。右边是掩码的计算公式：差异归一化、Sigmoid锐化并按中心距离加权，再与上一步做滑动平均。"))
algo = [["行", "步骤"], ["输入", "无缺陷图像 x_{free}，缺陷类别 y，窗口位置；起始时间步 t_{0} = 29"],
        ["1", "把 x_{free} 加噪到 t_{0} 得到 x_{t}；取窗口内部分作为 c_{t}"],
        ["2", "for t in 回跳时间表（每10步一段，每段去噪3遍）："], ["3", "    x̂_{0} ← 背景合成模型(x_{t}, t)；ĉ_{0} ← 缺陷先验模型(c_{t}, t, y)"],
        ["4", "    由 r(x_{free}) 与 ĉ_{0} 更新 A_{d}；A_{b} ← [A_{d} < 0.01]"], ["5", "    重复2次：计算 L，x_{t} ← x_{t} − η∇L"],
        ["6", "    DDIM更新得到 x_{t−1}、c_{t−1}"], ["7", "    段末且未满3遍：重新加噪回到段首"], ["8", "return x_{0}"]]
shp = table(s, 0.6, 1.45, 7.6, algo, [0.8, 6.8], 13, row_h=0.46)
for i in range(1, len(algo)):
    shp.table.cell(i, 1).text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT
caption(s, 0.6, 6.15, 7.6, "共73个去噪步；单张图像约17–30 s（RTX 4090）")
head(s, 8.6, 1.4, 4.3, "掩码计算", size=19)
for k, (tex, lab) in enumerate([(r"D=\mathrm{norm}\left(\frac{1}{3}\sum_k |r(x_{free})_k-\hat c_{0,k}|\right)", "像素差归一化"),
                                (r"A'=\min\left(1,\ \sigma\!\left(\frac{D-0.5}{\tau}\right)\left(1+w\,e^{-\rho^2/2\sigma_r^2}\right)\right)", "锐化与中心加权"),
                                (r"A_d^{(t)}=\gamma_t A_d^{(t+1)}+(1-\gamma_t)A^{(t)}", "时间步间滑动平均")]):
    y = 1.95 + k * 1.55
    rect(s, 8.6, y, 4.3, 1.4, PALE)
    image(s, formula(tex, 17), 8.75, y + 0.08, 4.0, 0.95, align="left")
    text(s, 8.75, y + 1.0, 4.0, 0.35, lab, 12, color=GREY)

# ======================= 12 去噪过程与结果 =======================
s = page("去噪过程与生成结果", SECTIONS[2], notes=(
    "上图是同一次生成中各时间步的中间结果：缺陷模型从第一步就给出了铜桥；背景模型对窗口的估计从第20步左右开始出现铜桥并逐步清晰；掩码很早就落在铜桥"
    "处。下图是六类缺陷的生成结果，窗口外与原图一致，标签框取该位置真实缺陷框的大小。"))
image(s, FIG / "fig_mask_steps.jpg", 0.7, 1.3, 11.9, 2.75)
caption(s, 0.7, 4.0, 11.9, "去噪过程中的估计与缺陷区域掩码（t 为250步采样序列中的序号）")
image(s, FIG / "fig_crg.jpg", 0.7, 4.4, 8.6, 2.2, align="left")
caption(s, 0.7, 6.6, 8.6, "六类缺陷的生成结果（上：无缺陷背景；下：生成结果；红框：标签框）")
bullets(s, 9.6, 4.45, 3.3, 2.4, ["缺陷先验模型先确定缺陷形态", "引导逐步把缺陷传递到整幅图像", "窗口外区域保持不变"], 14)

# ======================= 13 生成方法对比 =======================
s = page("与其他生成方法的对比：生成质量", SECTIONS[3], notes=(
    "在同一批生成任务上比较了七种方法。缺陷窗口的KID衡量生成缺陷与真实缺陷的距离：双扩散协同引导为12.4，明显低于其他方法；DDIM微调直接粘贴局部块，"
    "边缘不自然，窗口KID最大；DefFiller整幅和窗口都差别较大。"))
image(s, FIG / "fig_methods_thesis.jpg", 0.5, 1.3, 4.3, 5.6, align="left")
table(s, 5.2, 1.45, 7.7, [["方法", "KID×10³（整幅）", "KID×10³（窗口）", "LPIPS多样性"],
                         ["DDIM微调", "11.4", "30.0", "0.485"], ["RePaint", "10.2", "23.1", "0.482"], ["CoPaint", "12.2", "24.0", "0.480"],
                         ["ControlNet", "16.5", "20.3", "0.529"], ["DefFiller", "56.0", "62.0", "0.556"], ["AnomalyDiffusion", "6.4", "15.5", "0.503"],
                         ["复制粘贴", "9.4", "25.6", "0.485"], ["双扩散协同引导", "12.3", "12.4", "0.493"]],
      [2.3, 1.8, 1.8, 1.8], 13, hl_rows=(8,), row_h=0.44)
text(s, 5.2, 5.6, 7.7, 1.3, ["• 双扩散协同引导的**缺陷窗口KID最小**（12.4），生成缺陷最接近真实缺陷",
                             "• 参考集为选择集的真实图像；真实图像的LPIPS多样性为0.476"], 14)

# ======================= 14 检测对比与真实数据量 =======================
s = page("生成数据的作用与真实数据量", SECTIONS[3], notes=(
    "这是全文最主要的结论。训练集取每类10到60张真实图像，各加300张生成图像。双扩散协同引导在真实数据少时作用明显：每类10张时mAP@50-95从0.388提高到"
    "0.447，提高15.2%，3个随机种子都更高；每类20张提高5.0%；每类40张及以上与只用真实数据相当。其他六种生成方法在各数据量下都没有提高，多数还使指标下降。"
    "按选择集插值，约6到8张生成图像相当于1张真实图像。"))
line_chart(s, 0.5, 1.35, 7.6, 5.5, ["10", "20", "30", "40", "50", "60"],
           [("只用真实数据", (0.388, 0.446, 0.478, 0.494, 0.497, 0.490), RGBColor(0x47, 0x55, 0x69), True),
            ("双扩散协同引导", (0.447, 0.468, 0.486, 0.488, 0.494, 0.490), BLUE, False),
            ("DDIM微调", (0.419, 0.454, 0.475, 0.473, 0.495, 0.509), RGBColor(0xB4, 0xBC, 0xC8), False),
            ("CoPaint", (0.356, 0.438, 0.466, 0.480, 0.487, 0.496), RGBColor(0xCB, 0xD2, 0xDB), False)],
           0.34, 0.52, 0.04, "每类真实图像数", "mAP@50-95（测试集）")
kpi(s, 8.5, 1.45, 4.4, 1.55, "+15.2%", "每类10张：0.388 → 0.447", "3个随机种子都高于只用真实数据", BLUE)
kpi(s, 8.5, 3.2, 4.4, 1.55, "≈ 6–8 : 1", "生成图像与真实图像的替代关系", "按选择集结果在相邻数据量间插值")
bullets(s, 8.5, 5.0, 4.4, 1.9, ["每类20张+5.0%，30张+1.7%，40张及以上相当", "其他6种生成方法均没有提高",
                                "每类50张时与只用真实数据相当（0.494 / 0.497）"], 14)

# ======================= 15 按类别与扩增倍率 =======================
s = page("按类别分析与扩增倍率", SECTIONS[3], notes=(
    "按类别看，每类10张时六类都有提高，开路、毛刺和多余铜箔提高最多，约0.08到0.10，毛刺召回率从0.645提高到0.842；缺孔形态简单，只用真实数据时召回率已是1，"
    "提高最少。扩增倍率方面，每类50张时加300、600、900张生成图像都没有提高；质量差的方法如DefFiller随倍率增加而下降，质量好的方法加到3倍也不降。"))
head(s, 0.6, 1.4, 6.6, "每类10张真实图像时各类缺陷（mAP@50-95）", size=17)
table(s, 0.6, 1.9, 6.6, [["缺陷类别", "只用真实数据", "加入生成数据", "召回率"], ["缺孔", "0.484", "0.495", "1.000→0.965"],
                         ["鼠咬", "0.398", "0.455", "0.731→0.824"], ["开路", "0.368", "0.465", "0.823→0.856"], ["短路", "0.385", "0.403", "0.821→0.866"],
                         ["毛刺", "0.353", "0.441", "0.645→0.842"], ["多余铜箔", "0.341", "0.423", "0.705→0.854"]],
      [1.5, 1.6, 1.6, 1.9], 13, hl_rows=(3, 5, 6), row_h=0.45)
text(s, 0.6, 5.2, 6.6, 1.6, ["• 开路、毛刺、多余铜箔提高约**0.08–0.10**", "• 缺孔形态简单，少量真实样本已足够"], 14)
head(s, 7.6, 1.4, 5.3, "扩增倍率（每类50张，mAP@50-95）", size=17)
table(s, 7.6, 1.9, 5.3, [["方法", "1倍", "2倍", "3倍"], ["双扩散协同引导", "0.494", "0.485", "0.492"], ["DDIM微调", "0.495", "0.496", "0.493"],
                         ["ControlNet", "0.482", "0.475", "0.470"], ["DefFiller", "0.486", "0.476", "0.464"], ["复制粘贴", "0.501", "0.497", "0.494"]],
      [2.0, 1.1, 1.1, 1.1], 13, hl_rows=(1,), row_h=0.45)
text(s, 7.6, 4.75, 5.3, 2.0, ["• 只用真实数据为0.497；增加倍率没有提高任何方法", "• 质量差的方法随倍率增加而下降（DefFiller 3倍 −0.033）",
                              "• 提高生成质量比增加数量更重要"], 14)

# ======================= 16 消融 =======================
s = page("消融实验", SECTIONS[3], notes=(
    "消融实验每次去掉协同引导的一个部分。去掉缺陷损失后缺陷很弱甚至消失，而标签框仍在原处，相当于错误标签，每类10张时mAP@50-95降到0.348，比只用真实数据"
    "还低。两项背景损失和掩码滑动平均各贡献约0.01到0.02。去掉整个注意力掩码后窗口KID翻倍，但检测结果不变，说明掩码主要改善观感，对检测有用的信息"
    "来自缺陷先验模型给出的缺陷形态。"))
table(s, 0.6, 1.45, 6.3, [["设置", "窗口KID×10³", "每类10张", "变化"], ["完整方法", "12.4", "0.447", "—"], ["去掉缺陷损失", "19.0", "0.348", "−0.099"],
                         ["去掉窗口内背景损失", "15.8", "0.428", "−0.019"], ["去掉窗口外背景损失", "11.8", "0.432", "−0.015"],
                         ["去掉掩码滑动平均", "12.2", "0.436", "−0.011"], ["去掉注意力掩码", "24.7", "0.447", "0.000"], ["只用真实数据", "—", "0.388", "−0.059"]],
      [2.4, 1.4, 1.2, 1.3], 13, hl_rows=(2,), row_h=0.46)
text(s, 0.6, 5.3, 6.3, 1.6, ["• **缺陷损失是关键**：去掉后低于只用真实数据", "• 背景损失与滑动平均各贡献约0.01–0.02",
                             "• 注意力掩码主要改善观感，对检测影响不大"], 14)
image(s, FIG / "fig_ablation.jpg", 7.2, 1.3, 5.7, 5.6)

# ======================= 17 检测性能提升 =======================
s = page("检测性能的提升：输入尺寸、翻转与多模型平均", SECTIONS[4], notes=(
    "检测图像是256×256，放大到640后召回率从0.843提高到0.936，继续放大没有收益。每类50张时mAP@50约0.96，差距在高IoU下的定位精度。加上下翻转，"
    "并把3个随机种子模型的预测框加权平均，mAP@50-95达到0.527到0.529，但测试时要运行3个模型。"))
head(s, 0.6, 1.4, 5.6, "输入尺寸（只用真实数据）", size=17)
table(s, 0.6, 1.9, 5.6, [["输入", "精确率", "召回率", "mAP@50-95"], ["256", "0.929", "0.843", "0.451"], ["640", "0.956", "0.936", "0.497"],
                         ["800", "0.942", "0.927", "0.489"], ["960", "0.933", "0.923", "0.484"]], [1.1, 1.4, 1.4, 1.7], 13, hl_rows=(2,), row_h=0.45)
text(s, 0.6, 4.3, 5.6, 1.5, ["• 放大到640后召回率明显提高，继续放大无收益", "• 后续实验采用640输入"], 14)
head(s, 6.6, 1.4, 6.3, "翻转增强与多模型平均（每类50张）", size=17)
table(s, 6.6, 1.9, 6.3, [["方式", "生成数据", "精确率", "召回率", "mAP@50-95"], ["单模型", "无", "0.956", "0.936", "0.497"],
                         ["单模型+上下翻转", "无", "0.953", "0.942", "0.512"], ["3模型平均", "无", "0.951", "0.963", "0.521"],
                         ["3模型平均+上下翻转", "无", "0.962", "0.975", "0.529"], ["3模型平均+上下翻转", "有", "0.959", "0.959", "0.527"]],
      [2.15, 0.95, 0.95, 0.95, 1.3], 13, hl_rows=(4, 5), row_h=0.45)
text(s, 6.6, 4.75, 6.3, 2.0, ["• mAP@50约0.96：缺陷基本都能检出，差距在框的定位精度", "• 多模型平均：重叠的同类框按置信度加权融合",
                              "• 代价：测试时运行3个模型，耗时约3倍"], 14)

# ======================= 18 90°旋转 =======================
s = page("90°旋转增强", SECTIONS[4], notes=(
    "上下翻转有效，说明缺陷类别与方向无关；PCB的线路和焊盘多沿水平和竖直排列，旋转90°后仍是合理的电路板图像。所以在翻转的基础上再加转置，每张图以相同"
    "概率取8种90°旋转与镜像之一，不需要插值，标注框也仍然紧贴。旋转增强在所有设置下都提高了mAP@50-95；每类50张时单个模型达到0.516，精确率和召回率都在"
    "0.95以上，不需要多模型平均就达标。加入旋转后，每类10张时生成数据仍提高0.040。"))
line_chart(s, 0.5, 1.35, 6.6, 5.5, ["每类10张", "每类20张", "每类50张"],
           [("默认增强，真实数据", (0.388, 0.446, 0.490), RGBColor(0x94, 0xA3, 0xB8), True),
            ("旋转增强，真实数据", (0.437, 0.476, 0.516), RGBColor(0x47, 0x55, 0x69), False),
            ("旋转增强，加生成数据", (0.477, 0.493, 0.510), BLUE, False)],
           0.36, 0.54, 0.04, "真实数据量", "mAP@50-95（测试集）")
rect(s, 7.5, 1.45, 5.4, 1.75, PALE); rect(s, 7.5, 1.45, 0.08, 1.75, NAVY)
text(s, 7.75, 1.5, 5.0, 0.45, "做法", 16, True, NAVY)
text(s, 7.75, 1.95, 5.0, 1.2, "左右、上下翻转再加随机转置：8种90°旋转与镜像等概率出现；无需插值，标注框仍紧贴缺陷", 14, spacing=1.05)
kpi(s, 7.5, 3.4, 2.6, 1.55, "0.516", "单模型mAP@50-95", "3个种子均 ≥ 0.513", GREEN)
kpi(s, 10.3, 3.4, 2.6, 1.55, "+0.040", "每类10张，生成数据", "0.437 → 0.477", BLUE)
bullets(s, 7.5, 5.15, 5.4, 1.8, ["6种设置下都提高，测试集+0.007–0.049", "生成数据与旋转增强在少样本时可以叠加",
                                 "两阶段训练、标签框修正均未超过混合训练"], 14)

# ======================= 19 任务书指标 =======================
s = page("任务书指标的完成情况", SECTIONS[4], notes=(
    "按选择集选出的设置：单模型采用旋转增强、每类50张真实图像，精确率0.965、召回率0.960、mAP@50-95 0.516，全部达到任务书要求；6个模型平均后mAP@50-95为0.542。"
    "需要说明的是，这个条件下只用真实数据也能达标，达标主要靠检测模型的训练方式；生成数据的作用体现在真实数据少的时候，每类10张时加入生成数据后精确率和"
    "召回率都超过0.90，mAP@50-95为0.477。"))
for k, (v, l, t) in enumerate([("0.965", "精确率", "目标 ≥ 0.90"), ("0.960", "召回率", "目标 ≥ 0.90"), ("0.516", "mAP@50-95", "目标 ≥ 0.5")]):
    kpi(s, 0.6 + k * 4.15, 1.45, 3.9, 1.6, v, l, t, GREEN)
caption(s, 0.6, 3.1, 12.2, "单模型，旋转增强，每类50张真实图像（3个随机种子的均值）")
table(s, 0.6, 3.55, 12.2, [["设置", "精确率", "召回率", "mAP@50-95"], ["单模型，真实数据", "0.965±0.006", "0.960±0.008", "0.516±0.004"],
                          ["单模型，真实数据+生成数据", "0.960±0.007", "0.954±0.004", "0.510±0.006"], ["3模型平均，真实数据", "0.964", "0.972", "0.530"],
                          ["6模型平均，两种训练集", "0.960", "0.965", "0.542"], ["单模型，每类10张，加生成数据", "0.939±0.008", "0.919±0.008", "0.477±0.009"]],
      [5.0, 2.4, 2.4, 2.4], 14, hl_rows=(1,), row_h=0.48)
text(s, 0.6, 6.5, 12.2, 0.45, "每类50张时只用真实数据也能达标，指标达标主要依靠检测模型的训练方式；生成数据的作用体现在真实数据较少时。", 14, color=GREY)

# ======================= 20 结论 =======================
s = page("结论", SECTIONS[5], notes=(
    "总结三点结论：第一，用电路板模板去除缺陷构建了均匀、无缺陷的背景数据，协同引导生成的缺陷窗口KID为12.4，在比较的方法中最接近真实缺陷。第二，生成数据的作用"
    "取决于真实数据量，每类10张时提高15.2%，40张及以上不再提高，缺陷损失是生成数据起作用的关键。第三，90°旋转增强后单个模型达到任务书要求。"))
for k, (a, b) in enumerate([("基于模板的数据构建与协同引导生成", "模板去除缺陷、按网格截取背景块；生成缺陷的窗口KID为12.4，在所比较的方法中最接近真实缺陷"),
                            ("生成数据的作用取决于真实数据量", "每类10张时mAP@50-95提高15.2%，约6–8张生成图像相当于1张真实图像；每类40张及以上不再提高；缺陷损失是关键"),
                            ("检测指标达到任务书要求", "90°旋转增强后，单个模型精确率0.965、召回率0.960、mAP@50-95 0.516")]):
    y = 1.55 + k * 1.7
    s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.9), Inches(y + 0.1), Inches(0.75), Inches(0.75))
    o = s.shapes[-1]; o.fill.solid(); o.fill.fore_color.rgb = NAVY; o.line.fill.background()
    text(s, 0.9, y + 0.1, 0.75, 0.75, str(k + 1), 24, True, WHITE, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    text(s, 2.0, y, 10.8, 0.55, a, 21, True, NAVY)
    text(s, 2.0, y + 0.6, 10.8, 0.9, b, 16, color=INK)

# ======================= 21 不足与展望 =======================
s = page("不足与展望", SECTIONS[5], notes=(
    "本文的不足：实验只基于一个公开数据集，按照片划分时同一块电路板会同时出现在训练集和测试集中；缺陷先验模型使用公开预训练权重，自行训练时存在复制训练样本"
    "的问题；生成一幅图像需要17到30秒。今后可以在更多电路板和产线数据上验证，改进缺陷先验模型的训练方法，并提高生成效率。"))
head(s, 0.6, 1.45, 6, "不足", size=20)
for k, t in enumerate(["只基于一个公开数据集；按照片划分时同一电路板同时出现在训练集和测试集中",
                       "缺陷先验模型使用公开预训练权重；自行训练时样本少，存在复制训练样本的问题", "生成一幅图像需要17–30 s，速度较慢"]):
    y = 2.0 + k * 1.5
    rect(s, 0.6, y, 5.9, 1.3, PALE); rect(s, 0.6, y, 0.08, 1.3, NAVY)
    text(s, 0.85, y, 5.5, 1.3, t, 15, anchor=MSO_ANCHOR.MIDDLE, spacing=1.05)
head(s, 6.9, 1.45, 6, "展望", GREEN, 20)
for k, t in enumerate(["在更多电路板和实际产线数据上验证生成数据的作用", "对缺陷局部块做增强并加入正则化，减少缺陷先验模型对训练样本的复制",
                       "减少采样步数或进行模型蒸馏，提高生成效率；由生成过程的掩码直接得到标签框"]):
    y = 2.0 + k * 1.5
    rect(s, 6.9, y, 5.9, 1.3, RGBColor(0xEC, 0xF7, 0xF2)); rect(s, 6.9, y, 0.08, 1.3, GREEN)
    text(s, 7.15, y, 5.5, 1.3, t, 15, anchor=MSO_ANCHOR.MIDDLE, spacing=1.05)

# ======================= 22 致谢页 =======================
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, 13.333, 7.5, NAVY)
text(s, 0, 2.6, 13.333, 1.2, "谢谢！", 54, True, WHITE, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
text(s, 0, 3.85, 13.333, 0.7, "恳请各位老师批评指正", 22, color=SOFT, align=PP_ALIGN.CENTER)
s.notes_slide.notes_text_frame.text = "我的汇报到此结束，恳请各位老师批评指正，谢谢。"

prs.save(OUT)
for p in OUT.parent.glob("_f*.png"):
    p.unlink()
print("slides:", len(prs.slides), "->", OUT)
