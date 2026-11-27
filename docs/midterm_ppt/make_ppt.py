# 中期检查汇报 PPT（2026-11-27）：内容全部取自中期报告（docs/midterm_rebuild/content.json，第十六轮），不含中期之后的实验结果。
# 图取自 docs/midterm_rebuild/figures；公式用 matplotlib mathtext 渲染为 PNG；“不同真实数据量”为可编辑的原生折线图。
# 每页附讲稿（备注）。16:9，中文微软雅黑，数字与字母 Arial。数值范围用短横“–”（2026-11-27 用户：“～”显示不对）；封面不写日期。
# 用法：tools/pptx-venv/bin/python make_ppt.py <Graduate_Design 目录> <输出 .pptx>
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
FIG = ROOT / "docs" / "midterm_rebuild" / "figures"
EA, LAT = "微软雅黑", "Arial"
NAVY, BLUE, INK, GREY, LIGHT, PALE, GREEN = (RGBColor(0x0B, 0x3C, 0x7A), RGBColor(0x2A, 0x78, 0xD6), RGBColor(0x1F, 0x29, 0x37),
                                            RGBColor(0x64, 0x74, 0x8B), RGBColor(0xE2, 0xE8, 0xF0), RGBColor(0xF1, 0xF5, 0xFB),
                                            RGBColor(0x1B, 0x7F, 0x5B))
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
SECTIONS = ["研究背景与目标", "已完成的工作", "检测验证结果", "问题与后续计划"]


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
            r = p.add_run(); r.text = str(v)
            font(r, size, i == 0 or i in hl_rows, RGBColor(0xFF, 0xFF, 0xFF) if i == 0 else INK)
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
    text(s, 0.45, 7.03, 8, 0.35, "哈尔滨工业大学（深圳）  毕业论文（设计）中期检查", 11, color=GREY)
    text(s, 11.9, 7.03, 0.98, 0.35, str(len(prs.slides)), 11, color=GREY, align=PP_ALIGN.RIGHT)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


# ======================= 1 封面 =======================
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, 13.333, 7.5, NAVY)
rect(s, 0, 5.05, 13.333, 2.45, RGBColor(0x08, 0x2E, 0x5E))
rect(s, 0.9, 1.55, 0.12, 1.75, RGBColor(0x7F, 0xB2, 0xF0))
text(s, 1.25, 1.35, 11.2, 0.6, "毕业论文（设计）中期检查汇报", 20, color=RGBColor(0xBF, 0xD6, 0xF6))
text(s, 1.25, 2.0, 11.9, 1.2, "基于扩散模型的印刷电路板缺陷图像生成方法研究", 34, True, RGBColor(0xFF, 0xFF, 0xFF))
info = [("汇报人", "陈斯萍（2023313823）"), ("专    业", "机器人工程"), ("指导教师", "王睿 副教授"), ("学    院", "深圳校区机器人与先进制造学院")]
for k, (a, b) in enumerate(info):
    yy = 5.35 + (k // 2) * 0.62; xx = 1.25 + (k % 2) * 5.6
    text(s, xx, yy, 1.4, 0.5, a, 16, color=RGBColor(0xBF, 0xD6, 0xF6))
    text(s, xx + 1.45, yy, 4.0, 0.5, b, 16, color=RGBColor(0xFF, 0xFF, 0xFF))
s.notes_slide.notes_text_frame.text = ("各位老师好，我是机器人工程专业的陈斯萍，指导教师是王睿老师。我的毕业设计题目是基于扩散模型的印刷电路板缺陷"
                                       "图像生成方法研究。下面汇报第一阶段的工作进展。")

# ======================= 2 提纲 =======================
s = page("汇报提纲", notes="汇报分四部分：研究背景与目标、第一阶段已完成的工作、检测验证结果，以及存在的问题和后续计划。")
items = [("研究背景与目标", "缺陷样本少的问题；双扩散生成方案；任务书指标"),
         ("已完成的工作", "数据集划分与预处理；双扩散模型；协同引导生成；生成质量与参数调试"),
         ("检测验证结果", "生成数据与真实数据量的关系；检测精度的提高；任务书指标完成情况"),
         ("问题与后续计划", "存在的困难与解决方案；后期研究内容与进度安排")]
for k, (a, b) in enumerate(items):
    y = 1.6 + k * 1.28
    rect(s, 1.3, y, 0.9, 0.9, NAVY if k % 2 == 0 else BLUE)
    text(s, 1.3, y, 0.9, 0.9, f"{k + 1:02d}", 28, True, RGBColor(0xFF, 0xFF, 0xFF), PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    text(s, 2.5, y + 0.02, 9.5, 0.5, a, 22, True, NAVY)
    text(s, 2.5, y + 0.5, 9.5, 0.4, b, 15, color=GREY)

# ======================= 3 背景与目标 =======================
s = page("研究背景与目标", SECTIONS[0], notes=(
    "PCB缺陷检测模型需要大量带标注的缺陷图像，但实际生产中缺陷样本少、标注成本高。本课题用扩散模型生成缺陷图像来扩充检测训练集。"
    "方案是双扩散结构：背景合成模型学习无缺陷电路板，缺陷先验模型学习六类缺陷的局部形态，生成时两者协同，在指定位置画出缺陷而其余区域不变。"
    "任务书要求检测的精确率、召回率达到90%以上，mAP@50-95达到0.5以上。"))
text(s, 0.7, 1.45, 6.6, 0.5, "问题", 20, True, NAVY)
text(s, 0.7, 1.95, 6.6, 2.0, ["• 深度学习检测模型需要大量带标注的缺陷图像",
                              "• 实际生产中缺陷出现概率低，采集和标注成本高",
                              "• 旋转、翻转等常规增强不能产生新的缺陷形态"], 17)
text(s, 0.7, 3.6, 6.6, 0.5, "方案：双扩散模型协同引导生成", 20, True, NAVY)
text(s, 0.7, 4.1, 6.6, 2.6, ["• **背景合成模型**（256×256）：学习无缺陷电路板的线路、焊盘和铜箔",
                              "• **缺陷先验模型**（64×64）：以类别为条件学习六类缺陷的局部形态",
                              "• 生成时两者协同：在指定位置生成缺陷，其余区域保持不变",
                              "• 生成图像加入检测训练集，用检测指标评价其作用"], 17)
text(s, 8.0, 1.45, 4.8, 0.5, "任务书指标", 20, True, NAVY)
for k, (v, l) in enumerate([("≥ 90%", "精确率 P"), ("≥ 90%", "召回率 R"), ("≥ 0.5", "mAP@50-95")]):
    kpi(s, 8.0, 2.0 + k * 1.55, 4.8, 1.35, v, l)

# ======================= 4 技术路线 =======================
s = page("技术路线", SECTIONS[0], notes=(
    "技术路线分三栏：第一栏整理数据集，构建缺陷局部块和无缺陷背景块两类训练数据；第二栏是双扩散模型协同引导生成，每个去噪步计算注意力掩码和"
    "引导损失；第三栏把生成图像加入检测训练集，评价并提高检测性能，以任务书指标为目标。"))
image(s, FIG / "route_mid.jpg", 0.9, 1.3, 11.5, 5.6)

# ======================= 5 进度 =======================
s = page("进度概况", SECTIONS[0], notes=(
    "按开题报告的安排，2026年11月到2027年2月是第一阶段，目前第一阶段的工作基本完成：数据集核查与划分、两类训练数据的构建、双扩散模型和协同引导"
    "生成流程、生成质量评价和超参数调试都已完成。检测验证中，加上下翻转增强并对3个模型取平均后，三项指标已达到任务书要求。"))
phases = [("第一阶段", "2026.11—2027.02", "数据预处理、双扩散模型搭建、\n模型训练与超参数调试", "基本完成", GREEN),
          ("第二阶段", "2027.02—2027.04", "对比实验与消融实验、\n扩增倍率实验", "待开展", GREY),
          ("第三阶段", "2027.04—2027.05", "整理实验结果、撰写论文、\n准备答辩", "待开展", GREY)]
for k, (a, t, d, st, c) in enumerate(phases):
    x = 0.8 + k * 4.1
    rect(s, x, 1.55, 3.8, 0.12, c)
    text(s, x, 1.8, 3.8, 0.5, a, 20, True, NAVY); text(s, x, 2.28, 3.8, 0.4, t, 14, color=GREY)
    text(s, x, 2.75, 3.8, 1.0, d.split("\n"), 15, spacing=1.0)
    rect(s, x, 3.7, 1.3, 0.42, c, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, x, 3.7, 1.3, 0.42, st, 14, True, RGBColor(0xFF, 0xFF, 0xFF), PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
text(s, 0.8, 4.45, 11.8, 0.5, "第一阶段已完成", 18, True, NAVY)
text(s, 0.8, 4.95, 5.9, 2.0, ["• PKU-Market-PCB数据集核查与划分",
                              "• 缺陷局部块（2185个）与无缺陷背景块（3000个）",
                              "• 双扩散模型搭建、训练程序与DDIM采样程序"], 16)
text(s, 6.9, 4.95, 5.9, 2.0, ["• 协同引导生成流程，生成质量评价",
                              "• 生成参数与检测输入尺寸调试",
                              "• 检测验证：P 0.959、R 0.959、mAP@50-95 0.527"], 16)

# ======================= 6 数据集 =======================
s = page("数据集与划分", SECTIONS[1], notes=(
    "数据集是PKU-Market-PCB，693张图像，10块电路板，6类缺陷，类别基本均衡。同一块电路板的不同图像背景几乎相同，所以划分时要考虑相关性："
    "扩散模型只用电路板01、04到09训练，10到12留作评价生成质量；检测实验按照片分成训练、选择、测试三组互不重叠。"
    "训练照片里有1044个缺陷，每类50张是基本设置，每类10、20张的子集用来模拟缺陷样本很少的情况。"))
table(s, 0.7, 1.5, 5.6, [["缺陷类别", "图像数", "缺陷数"], ["缺孔", "115", "497"], ["鼠咬", "115", "492"], ["开路", "116", "482"],
                         ["短路", "116", "491"], ["毛刺", "115", "488"], ["多余铜箔", "116", "503"], ["合计", "693", "2953"]],
      [2.2, 1.7, 1.7], 15, hl_rows=(7,), row_h=0.5)
caption(s, 0.7, 5.6, 5.6, "PKU-Market-PCB：10块电路板，693张图像，6类缺陷")
text(s, 6.9, 1.45, 5.9, 0.5, "扩散模型：按电路板划分", 19, True, NAVY)
text(s, 6.9, 1.95, 5.9, 1.3, ["• 训练：电路板01、04–09，541张图像",
                              "• 评价生成质量：电路板10–12，152张图像"], 16)
text(s, 6.9, 3.15, 5.9, 0.5, "检测实验：按照片划分（互不重叠）", 19, True, NAVY)
text(s, 6.9, 3.65, 5.9, 3.2, ["• 训练照片：1044个缺陷，每类50张为基本设置，另加50张无缺陷图像",
                              "• 选择集300张：只用于选择训练轮次和设置",
                              "• 测试集300张：只用于最终评价",
                              "• 每类10–60张的子集：考察真实数据量的影响，10、20张模拟缺陷样本很少的情况"], 16)

# ======================= 7 预处理：缺陷局部块 =======================
s = page("训练数据预处理（一）：尺度统一与缺陷局部块", SECTIONS[1], notes=(
    "原图分辨率高，缺陷框长边中位数约75像素，放不进64×64的局部块，所以先把整幅图缩小为0.5倍，约95%的缺陷可以被完整容纳，"
    "焊盘尺度也与预训练模型一致。然后以缺陷框中心加少量随机偏移截取64×64局部块；缩小后仍放不下的115个较大缺陷先放大窗口再缩小。"
    "共得到2185个缺陷局部块。"))
image(s, FIG / "fig_pre_overview.jpg", 0.6, 1.4, 6.2, 3.7)
caption(s, 0.6, 5.12, 6.2, "缺陷局部块的截取窗口（红框：标注框；黄框：截取窗口）")
image(s, FIG / "fig_pre_patches.jpg", 7.3, 1.35, 2.2, 4.2)
caption(s, 7.0, 5.6, 2.8, "六类缺陷的局部块示例")
text(s, 10.0, 1.5, 2.9, 4.5, [("0.5×", {"size": 30, "bold": True, "color": NAVY, "after": 0}), ("整幅图像缩小，约95%缺陷可完整放入64×64窗口", {"size": 14, "after": 16}),
                              ("2185", {"size": 30, "bold": True, "color": NAVY, "after": 0}), ("个缺陷局部块，每类351–371个", {"size": 14, "after": 16}),
                              ("115", {"size": 30, "bold": True, "color": NAVY, "after": 0}), ("个较大缺陷：放大窗口后再缩小，不舍弃", {"size": 14})])
text(s, 0.6, 5.85, 12.2, 0.9, "截取时加入不超过8像素的随机偏移，避免缺陷总位于局部块正中；文件名带类别，供缺陷先验模型读取条件标签。", 15, color=GREY)

# ======================= 8 预处理：背景块 =======================
s = page("训练数据预处理（二）：无缺陷背景块", SECTIONS[1], notes=(
    "背景合成模型只能学无缺陷的电路板。直接避开缺陷取窗口，缺陷密集的区域就很少被取到。同一块电路板的各张图像背景相同，每个缺陷只出现在一张"
    "图像中，所以对同一块板的全部图像逐像素取中值得到模板，用模板覆盖缺陷附近区域就能去除缺陷。再按网格截取256×256背景块，"
    "覆盖次数的变异系数从0.096降到0.031，最后逐像素检查，共3000个背景块。"))
image(s, FIG / "fig_pre_clean.jpg", 0.6, 1.35, 3.6, 5.1)
caption(s, 0.6, 6.45, 3.6, "原图 / 电路板模板 / 去除缺陷后")
image(s, FIG / "fig_pre_coverage.jpg", 4.5, 1.5, 5.0, 3.6)
caption(s, 4.5, 5.1, 5.0, "电路板01上背景块的覆盖次数（左：避开缺陷随机截取；右：按网格截取）")
text(s, 9.9, 1.5, 3.0, 5.2, [("逐像素中值", {"size": 20, "bold": True, "color": NAVY, "after": 2}),
                             ("同一电路板的全部图像取中值，得到不含缺陷的模板", {"size": 14, "after": 14}),
                             ("0.096 → 0.031", {"size": 24, "bold": True, "color": NAVY, "after": 2}),
                             ("按网格截取后，中部区域覆盖次数的变异系数", {"size": 14, "after": 14}),
                             ("3000", {"size": 28, "bold": True, "color": NAVY, "after": 2}),
                             ("个256×256背景块，逐像素与模板比较，不合格的改从其他图像截取", {"size": 14})])

# ======================= 9 扩散模型原理 =======================
s = page("扩散模型与DDIM采样", SECTIONS[1], notes=(
    "DDPM包括前向加噪和反向去噪：前向过程逐步加入高斯噪声，任意时间步的加噪图像可以由原图一步得到；网络学习预测加入的噪声，训练目标是噪声的"
    "均方误差。由预测的噪声可以估计原图。DDIM训练目标相同，但采样确定、可以跳步，还能把已有图像加噪到中间时间步再去噪，只修改局部内容，"
    "这是协同引导的基础。"))
image(s, FIG / "op_ddpm.jpg", 0.7, 1.35, 11.9, 2.3)
caption(s, 0.7, 3.62, 11.9, "DDPM的前向加噪与反向去噪过程")
F = [(r"x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon", "前向加噪：任意时间步一步得到"),
     (r"L_{\mathrm{simple}}=\mathbb{E}\,\|\epsilon-\epsilon_\theta(x_t,t)\|^2", "训练目标：预测噪声"),
     (r"\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}", "由预测噪声估计原图"),
     (r"x_s=\sqrt{\bar\alpha_s}\,\hat x_0+\sqrt{1-\bar\alpha_s}\,\epsilon_\theta(x_t,t)", "DDIM确定性更新，可跳步")]
for k, (tex, lab) in enumerate(F):
    x = 0.7 + (k % 2) * 6.1; y = 4.15 + (k // 2) * 1.35
    rect(s, x, y, 5.8, 1.15, PALE)
    image(s, formula(tex, 20), x + 0.15, y + 0.1, 3.75, 0.95, align="left")
    text(s, x + 4.1, y, 1.65, 1.15, lab, 13, color=GREY, anchor=MSO_ANCHOR.MIDDLE)

# ======================= 10 双扩散模型 =======================
s = page("双扩散模型：结构与单模型采样", SECTIONS[1], notes=(
    "两个模型都是改进DDPM的U-Net，1000步，线性噪声调度，同时预测噪声和方差。我补充了训练程序，用不同训练数据训练了多组模型，和公开的预训练"
    "权重比较，以检测结果为准选最好的一组；自训练的缺陷模型样本只有几百个，生成接近复制，所以后续用预训练权重。单独采样时，缺陷模型能生成"
    "六类可辨认的缺陷，背景模型能生成清晰的焊盘和线路。"))
table(s, 0.6, 1.45, 5.4, [["项目", "背景合成模型", "缺陷先验模型"], ["输入尺寸", "256×256", "64×64"], ["训练数据", "无缺陷背景块", "缺陷局部块"],
                         ["条件信息", "无", "缺陷类别（6类）"], ["基础通道数", "128", "64"], ["通道倍率", "1,1,2,3,4,4", "1,2,3,4"]],
      [1.5, 1.95, 1.95], 13, row_h=0.42)
text(s, 0.6, 4.2, 5.4, 2.6, ["• 基于改进DDPM的U-Net，扩散步数1000，线性噪声调度，同时预测噪声和方差",
                             "• 补充训练程序，训练多组模型并与公开预训练权重比较，以检测结果为准选择最好的一组",
                             "• 补充独立的DDIM采样程序，检查各模型学到的内容"], 14)
image(s, FIG / "fig_64_samples.jpg", 6.3, 1.4, 3.2, 4.9)
caption(s, 6.3, 6.35, 3.2, "缺陷先验模型采样（每行一类）")
image(s, FIG / "fig_256_samples.jpg", 9.7, 1.4, 3.2, 4.9)
caption(s, 9.7, 6.35, 3.2, "背景合成模型采样")

# ======================= 11 协同引导原理 =======================
s = page("协同引导缺陷生成：原理", SECTIONS[1], notes=(
    "输入是无缺陷图像、缺陷类别和位置。先把图像加噪到中等噪声水平，再逐步去噪。每一步，背景模型估计完整图像，缺陷模型估计窗口内的缺陷，"
    "两者与原背景的差异得到注意力掩码，区分缺陷区域和背景区域。引导损失让缺陷区域跟随缺陷模型、其余区域保持原图，对x_t做两次梯度下降后再做一步"
    "DDIM更新。图中每个缩略图都是一次实际生成的中间结果。"))
image(s, FIG / "fig_principle.jpg", 0.5, 1.3, 8.4, 5.6, align="left")
text(s, 9.2, 1.45, 3.7, 0.5, "引导损失", 19, True, NAVY)
image(s, formula(r"L=\lambda_d L_d+\lambda_b L_b+\lambda_o L_o", 20), 9.2, 1.95, 3.6, 0.6, align="left")
text(s, 9.2, 2.7, 3.7, 2.2, ["• L_{d}：掩码A_{d}内，图像估计跟随缺陷估计",
                             "• L_{b}：掩码A_{b}内，保持无缺陷背景",
                             "• L_{o}：窗口外，保持无缺陷背景"], 14)
text(s, 9.2, 4.55, 3.7, 0.5, "注意力掩码", 19, True, NAVY)
text(s, 9.2, 5.05, 3.7, 1.9, ["• 由缺陷估计与原背景之差计算",
                              "• 锐化、中心加权、连通域筛选",
                              "• 相邻时间步之间滑动平均"], 14)

# ======================= 12 去噪过程与生成结果 =======================
s = page("协同引导缺陷生成：过程与结果", SECTIONS[1], notes=(
    "上图是同一次生成中各时间步的中间结果。缺陷模型从第一步就给出了两条线路之间的铜桥；背景模型对窗口的估计开始还是原来的线路，从第20步左右"
    "开始出现铜桥并逐步清晰；掩码很早就落在铜桥处。下图是六类缺陷的生成结果，上行是无缺陷背景，下行是生成结果，红框是标签框，"
    "标签框取该位置真实缺陷框的大小。"))
image(s, FIG / "fig_mask_steps.jpg", 0.7, 1.3, 11.9, 2.75)
caption(s, 0.7, 4.0, 11.9, "去噪过程中的估计与缺陷区域掩码（时间步为250步采样序列中的序号，从第29步开始去噪）")
image(s, FIG / "fig_crg.jpg", 0.7, 4.4, 8.6, 2.2, align="left")
caption(s, 0.7, 6.6, 8.6, "六类缺陷的生成结果（上：无缺陷背景；下：生成结果；红框：标签框）")
text(s, 9.6, 4.45, 3.3, 2.4, ["• 背景取自检测训练照片",
                              "• 位置取同板同类真实缺陷处",
                              "• 标签框取真实缺陷框大小",
                              "• 单张约17–30 s（RTX 4090）"], 14)

# ======================= 13 生成质量与参数调试 =======================
s = page("生成质量评价与参数调试", SECTIONS[1], notes=(
    "生成质量方面，两个模型各采样600张，与电路板10到12的真实图像比较。最近邻距离比接近或大于1，说明没有复制训练样本；缺陷模型覆盖了约一半的"
    "真实缺陷形态。背景模型逼真程度偏低，但生成时背景直接取自真实图像，影响不大。参数调试设置了40组参数，用检出率初筛，前3组在每类10张的条件下"
    "训练检测模型，都没有超过默认配置，后续保持默认参数。"))
text(s, 0.6, 1.4, 6.1, 0.5, "生成质量（与电路板10–12的真实图像比较）", 18, True, NAVY)
table(s, 0.6, 1.95, 6.1, [["模型", "KID×10³", "生成召回率", "最近邻距离比"], ["缺陷先验模型", "25.3", "0.51", "0.97"],
                         ["背景合成模型", "98.0", "0.69", "1.28"]], [1.9, 1.3, 1.45, 1.45], 14, row_h=0.48)
text(s, 0.6, 3.55, 6.1, 3.2, ["• 最近邻距离比接近或大于1：没有复制训练样本",
                              "• 缺陷先验模型覆盖约一半的真实缺陷形态",
                              "• 背景合成模型逼真度偏低（丝印字符），但生成时背景取自真实图像，影响小"], 15)
text(s, 7.1, 1.4, 5.8, 0.5, "生成参数筛选（40组 → 3组）", 18, True, NAVY)
table(s, 7.1, 1.95, 5.8, [["设置", "检出率", "mAP@50-95"], ["默认配置", "0.470", "0.447"], ["每步梯度下降4次", "0.497", "0.442"],
                         ["掩码阈值0.8", "0.453", "0.431"], ["掩码温度0.5，阈值0.8", "0.473", "0.428"]],
      [2.8, 1.4, 1.6], 14, hl_rows=(1,), row_h=0.48)
text(s, 7.1, 4.55, 5.8, 2.2, ["• 检出率：只用真实数据训练的检测模型在生成缺陷处的检出比例，用于初筛",
                              "• mAP@50-95：每类10张真实图像，3个随机种子；只用真实图像为0.388",
                              "• 筛选出的设置都没有超过默认配置，保持默认参数"], 14)

# ======================= 14 检测设置与输入尺寸 =======================
s = page("检测验证：实验设置与输入尺寸", SECTIONS[2], notes=(
    "检测模型用YOLO11n，单阶段检测器，约260万参数，训练200个epoch。训练中用选择集选最优轮次，测试集只用于最终评价，每个设置3个随机种子。"
    "检测图像是256×256，比较了四种输入尺寸：放大到640后召回率明显提高，因为小缺陷在特征图上占的像素更多；继续放大没有收益，所以用640。"))
text(s, 0.6, 1.4, 5.6, 0.5, "实验设置", 18, True, NAVY)
text(s, 0.6, 1.9, 5.6, 4.8, ["• 检测模型：YOLO11n（单阶段，约260万参数）",
                             "• 训练200个epoch，批次32",
                             "• 训练集：每类50张真实缺陷图像 + 50张无缺陷图像；加入生成数据时另加300张",
                             "• 选择集选取训练轮次和设置，测试集只用于最终评价",
                             "• 每个设置3个随机种子，取平均",
                             "• 指标：精确率P、召回率R、mAP@50-95"], 15)
text(s, 6.6, 1.4, 6.3, 0.5, "输入尺寸的比较（只用真实数据）", 18, True, NAVY)
table(s, 6.6, 1.95, 6.3, [["输入尺寸", "精确率", "召回率", "mAP@50-95"], ["256", "0.929", "0.843", "0.451"], ["640", "0.956", "0.936", "0.497"],
                         ["800", "0.942", "0.927", "0.489"], ["960", "0.933", "0.923", "0.484"]],
      [1.5, 1.5, 1.5, 1.8], 14, hl_rows=(2,), row_h=0.48)
text(s, 6.6, 4.55, 6.3, 2.2, ["• 256 → 640：召回率由0.843提高到0.936，小缺陷在特征图上占据更多像素",
                              "• 继续放大没有提高，显存和训练时间增加",
                              "• 后续实验采用640输入"], 15)

# ======================= 15 生成数据与真实数据量 =======================
s = page("检测验证：生成数据与真实数据量的关系", SECTIONS[2], notes=(
    "这是第一阶段最主要的结论。训练集取每类10到60张真实图像，各加入300张生成图像。真实数据越少，生成数据的作用越明显：每类10张时mAP@50-95"
    "从0.388提高到0.447，提高15.2%，3个随机种子都更高；每类20张提高5%；每类40张及以上时与只用真实数据相当。按选择集插值，"
    "大约6到8张生成图像相当于1张真实图像。"))
cd = CategoryChartData(); cd.categories = ["10", "20", "30", "40", "50", "60"]
cd.add_series("只用真实数据", (0.388, 0.446, 0.478, 0.494, 0.497, 0.490))
cd.add_series("真实数据+300张生成数据", (0.447, 0.468, 0.486, 0.488, 0.494, 0.490))
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.5), Inches(1.35), Inches(7.6), Inches(5.5), cd); ch = gf.chart
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
va = ch.value_axis; va.minimum_scale, va.maximum_scale, va.major_unit = 0.36, 0.52, 0.04
va.has_major_gridlines = True; va.major_gridlines.format.line.color.rgb = LIGHT; va.tick_labels.number_format = "0.00"
va.tick_labels.number_format_is_linked = False
ca = ch.category_axis; ca.has_title = True; ca.axis_title.text_frame.text = "每类真实图像数"
va.has_title = True; va.axis_title.text_frame.text = "mAP@50-95（测试集）"
for ax in (va, ca):
    for p in ax.axis_title.text_frame.paragraphs:
        for r in p.runs:
            font(r, 13, False, GREY)
    ax.tick_labels.font.size = Pt(12)
ch.legend.font.size = Pt(13)
for ser, col, ms in zip(ch.plots[0].series, (RGBColor(0x47, 0x55, 0x69), BLUE), (XL_MARKER_STYLE.SQUARE, XL_MARKER_STYLE.CIRCLE)):
    ser.format.line.color.rgb = col; ser.format.line.width = Pt(2.5); ser.smooth = False
    ser.marker.style = ms; ser.marker.size = 8; ser.marker.format.fill.solid(); ser.marker.format.fill.fore_color.rgb = col
    ser.marker.format.line.color.rgb = col
ch.plots[0].series[0].format.line.dash_style = 4
kpi(s, 8.5, 1.45, 4.4, 1.55, "+15.2%", "每类10张：0.388 → 0.447", "3个随机种子都高于只用真实数据", BLUE)
kpi(s, 8.5, 3.2, 4.4, 1.55, "≈ 6–8 : 1", "生成图像与真实图像的替代关系", "按选择集结果在相邻数据量间插值")
text(s, 8.5, 5.0, 4.4, 1.9, ["• 每类20张提高5.0%，30张提高1.7%",
                             "• 每类40张及以上与只用真实数据相当（在种子波动范围内）",
                             "• 生成数据主要在真实缺陷少时起作用"], 14)

# ======================= 16 检测精度提高与指标 =======================
s = page("检测验证：检测精度的提高与任务书指标", SECTIONS[2], notes=(
    "每类50张时mAP@50约0.96，缺陷基本都能检出，差距在高IoU阈值下的框定位精度。所以做了两点调整：训练时加上下翻转，因为缺陷上下翻转后类别不变；"
    "测试时把3个随机种子模型的预测框加权平均。两者结合后mAP@50-95达到0.527到0.529，精确率和召回率都在0.95以上，三项指标都达到任务书要求。"
    "代价是测试时要运行3个模型。"))
table(s, 0.6, 1.45, 7.2, [["方式", "生成数据", "精确率", "召回率", "mAP@50-95"],
                         ["单模型", "无", "0.956", "0.936", "0.497"], ["单模型", "有", "0.969", "0.921", "0.494"],
                         ["单模型+上下翻转", "无", "0.953", "0.942", "0.512"], ["3模型平均", "无", "0.951", "0.963", "0.521"],
                         ["3模型平均+上下翻转", "无", "0.962", "0.975", "0.529"], ["3模型平均+上下翻转", "有", "0.959", "0.959", "0.527"]],
      [2.6, 1.1, 1.1, 1.1, 1.3], 13, hl_rows=(5, 6), row_h=0.45)
text(s, 0.6, 4.75, 7.2, 2.1, ["• **上下翻转增强**：缺陷上下翻转后类别不变，训练时以0.5的概率翻转",
                              "• **多模型平均**：3个随机种子的模型分别预测，重叠的同类框按置信度加权平均",
                              "• 训练集为每类50张真实图像，“有”表示另加300张生成图像"], 14)
text(s, 8.3, 1.4, 4.6, 0.5, "任务书指标完成情况", 18, True, NAVY)
for k, (v, l, t) in enumerate([("0.959", "精确率", "目标 ≥ 0.90"), ("0.959", "召回率", "目标 ≥ 0.90"), ("0.527", "mAP@50-95", "目标 ≥ 0.5")]):
    kpi(s, 8.3, 1.95 + k * 1.55, 4.6, 1.42, v, l, t, GREEN)
caption(s, 8.3, 6.55, 4.6, "原始数据+生成数据，上下翻转，3模型平均")

# ======================= 17 困难与方案 =======================
s = page("存在的困难与解决方案", SECTIONS[3], notes=(
    "目前有三个困难。一是真实数据充足时生成数据没有提高检测指标，后续把少样本作为主要评价场景，并用对比和消融实验分析原因。二是单模型精度不够稳定，"
    "达标依赖多模型平均，测试集也只有300张，后续每个设置多种子重复、只按选择集选设置，并尝试两阶段训练、标签框修正和旋转增强。"
    "三是自训练的缺陷先验模型容易复制训练样本，后续对局部块做旋转翻转增强，并按最近邻距离比监视复制程度。"))
rows = [("真实数据充足时，生成数据没有提高检测指标", "每类50张时加入生成数据与只用真实数据相当，调参后也无明显变化",
         "把少样本条件作为主要评价场景；通过对比实验与消融实验找出影响增强效果的因素"),
        ("单模型精度不够稳定", "单模型mAP@50-95在0.49–0.51之间波动，达标依赖多模型平均；测试集只有300张",
         "多随机种子重复，只按选择集选设置；尝试两阶段训练、标签框修正、90°旋转增强提高单模型精度"),
        ("自训练的缺陷先验模型容易复制训练样本", "局部块只有几百个，模型基本记住了训练样本，效果不如预训练权重",
         "对局部块做旋转、翻转增强；按最近邻距离比监视复制程度，以少样本检测结果选择权重")]
text(s, 0.6, 1.35, 5.9, 0.45, "困难", 18, True, NAVY); text(s, 6.95, 1.35, 5.9, 0.45, "解决方案", 18, True, GREEN)
for k, (a, b, c) in enumerate(rows):
    y = 1.9 + k * 1.68
    rect(s, 0.6, y, 5.9, 1.5, PALE); rect(s, 0.6, y, 0.08, 1.5, NAVY)
    text(s, 0.85, y + 0.1, 5.5, 0.5, a, 16, True, INK)
    text(s, 0.85, y + 0.62, 5.5, 0.85, b, 13, color=GREY, spacing=1.05)
    arr = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(6.55), Inches(y + 0.55), Inches(0.35), Inches(0.4))
    arr.fill.solid(); arr.fill.fore_color.rgb = LIGHT; arr.line.fill.background()
    rect(s, 6.95, y, 5.9, 1.5, RGBColor(0xEC, 0xF7, 0xF2)); rect(s, 6.95, y, 0.08, 1.5, GREEN)
    text(s, 7.2, y + 0.1, 5.5, 1.35, c, 15, color=INK, anchor=MSO_ANCHOR.MIDDLE, spacing=1.05)

# ======================= 18 后续计划 =======================
s = page("后期研究工作与进度安排", SECTIONS[3], notes=(
    "后期按开题报告开展对比实验和消融实验：与DDIM微调、RePaint、CoPaint、ControlNet、DefFiller、AnomalyDiffusion和复制粘贴比较；"
    "消融分别去掉缺陷损失、背景损失和掩码；比较1到3倍扩增；同时改进缺陷先验模型、提高单模型精度。2027年4月以后整理结果、撰写论文。"))
text(s, 0.6, 1.4, 7.0, 0.5, "后期研究内容", 18, True, NAVY)
text(s, 0.6, 1.9, 7.0, 5.0, ["• **改进缺陷先验模型**：局部块旋转、翻转增强，按复制程度和检测结果选权重",
                             "• **对比实验**：DDIM微调、RePaint、CoPaint、ControlNet、DefFiller、AnomalyDiffusion、复制粘贴",
                             "• **消融实验**：分别去掉缺陷损失、背景损失和掩码引导",
                             "• **扩增倍率与少样本**：1倍、2倍、3倍扩增；每类10、20张条件",
                             "• **提高单模型精度**：两阶段训练、标签框修正、90°旋转增强",
                             "• 整理实验结果，撰写论文，准备答辩"], 15)
text(s, 8.1, 1.4, 4.8, 0.5, "进度安排", 18, True, NAVY)
for k, (t, d) in enumerate([("2027.02 — 2027.04", "开展对比实验与消融实验，测试不同扩增倍率，对比基线方法指标，分析实验结果，优化生成效果"),
                            ("2027.04 — 2027.05", "整理实验数据与图表，修改完善毕业论文，最终定稿并准备答辩")]):
    y = 2.0 + k * 2.3
    s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(8.1), Inches(y + 0.05), Inches(0.3), Inches(0.3)).fill.solid()
    dot = s.shapes[-1]; dot.fill.fore_color.rgb = BLUE; dot.line.fill.background()
    if k == 0:
        rect(s, 8.235, y + 0.4, 0.03, 2.0, LIGHT)
    text(s, 8.55, y - 0.02, 4.3, 0.45, t, 17, True, NAVY)
    text(s, 8.55, y + 0.45, 4.3, 1.6, d, 14, spacing=1.1)

# ======================= 19 小结 =======================
s = page("总结", SECTIONS[3], notes=(
    "总结一下：第一阶段的数据预处理、双扩散模型搭建和超参数调试已基本完成；检测指标已达到任务书要求；少样本条件下生成数据能明显提高检测指标。"
    "后续实验所需的数据和流程都已准备好，论文能够按期完成。"))
for k, (a, b) in enumerate([("第一阶段工作基本完成", "数据集划分与预处理、双扩散模型搭建、协同引导生成、生成质量评价与超参数调试"),
                            ("检测指标达到任务书要求", "上下翻转增强 + 3模型平均：精确率0.959、召回率0.959、mAP@50-95 0.527"),
                            ("少样本条件下生成数据有效", "每类10张真实图像时mAP@50-95提高15.2%，约6–8张生成图像相当于1张真实图像"),
                            ("论文能够按期完成", "后续实验所需的数据、生成和检测流程均已准备好，主要风险已有对应方案")]):
    y = 1.5 + k * 1.3
    s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.9), Inches(y + 0.12), Inches(0.62), Inches(0.62))
    o = s.shapes[-1]; o.fill.solid(); o.fill.fore_color.rgb = GREEN if k < 3 else NAVY; o.line.fill.background()
    text(s, 0.9, y + 0.12, 0.62, 0.62, "✓" if k < 3 else "→", 20, True, RGBColor(0xFF, 0xFF, 0xFF), PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    text(s, 1.85, y, 10.8, 0.5, a, 21, True, NAVY)
    text(s, 1.85, y + 0.52, 10.8, 0.5, b, 15, color=GREY)

# ======================= 20 致谢页 =======================
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, 13.333, 7.5, NAVY)
text(s, 0, 2.6, 13.333, 1.2, "谢谢！", 54, True, RGBColor(0xFF, 0xFF, 0xFF), PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
text(s, 0, 3.85, 13.333, 0.7, "恳请各位老师批评指正", 22, color=RGBColor(0xBF, 0xD6, 0xF6), align=PP_ALIGN.CENTER)
s.notes_slide.notes_text_frame.text = "我的汇报到此结束，恳请各位老师批评指正，谢谢。"

prs.save(OUT)
for p in OUT.parent.glob("_f*.png"):
    p.unlink()
print("slides:", len(prs.slides), "->", OUT)
