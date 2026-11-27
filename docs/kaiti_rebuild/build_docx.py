# 按模板红字要求生成开题报告 .docx（仅用标准库手写 OOXML）
# 用法：python build_docx.py <kaiti_rebuild 目录> <输出 .docx>
import sys, json, re, zipfile, datetime
from xml.sax.saxutils import escape

HERE, OUT = sys.argv[1], sys.argv[2]
content = json.load(open(f"{HERE}/content.json"))
sizes = json.load(open(f"{HERE}/figures/sizes.json"))

NS = ('xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
      'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
      'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"')
XML_HEAD = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'

TNR, SONG, HEI, XINWEI = "Times New Roman", "宋体", "黑体", "华文新魏"
HEADER_TEXT = "哈尔滨工业大学深圳校区毕业论文（设计）开题报告"
TEXT_W = 11906 - 2 * 1701          # 正文版心宽度（twips）
TEXT_W_PT = TEXT_W / 20


# ---------- 基础构件 ----------
def rpr(east=None, ascii_=None, bold=False, scale=None, size=None, sup=False, hint=False):
    """按 CT_RPr 规定顺序拼接 w:rPr"""
    parts = []
    attrs = []
    if ascii_:
        attrs += [f'w:ascii="{ascii_}"', f'w:hAnsi="{ascii_}"']
    if east:
        attrs.append(f'w:eastAsia="{east}"')
    if hint:
        attrs.append('w:hint="eastAsia"')
    if attrs:
        parts.append(f'<w:rFonts {" ".join(attrs)}/>')
    if bold:
        parts.append('<w:b/><w:bCs/>')
    if scale:
        parts.append(f'<w:w w:val="{scale}"/>')
    if size:
        parts.append(f'<w:sz w:val="{int(size * 2)}"/><w:szCs w:val="{int(size * 2)}"/>')
    if sup:
        parts.append('<w:vertAlign w:val="superscript"/>')
    return f'<w:rPr>{"".join(parts)}</w:rPr>' if parts else ""


def run(text, props=""):
    return f'<w:r>{props}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


TOKEN = re.compile(r"(\^\[[^\]]+\]\^|[“”‘’…—]+|[→≥])")
# 符号统一用 Times New Roman（模板：“数字及符号用Times New Roman字体”）。不指定时随前一字符取字体，
# 同一符号会在中文后用宋体、在西文后用 Times New Roman，字形不一致
SYMBOL = re.compile(r"[→≥]")


def text_runs(s, **base):
    """拆分上标引用 ^[n]^、符号，以及需按中文字体显示的引号、省略号、破折号"""
    out = []
    for tok in TOKEN.split(s):
        if not tok:
            continue
        if tok.startswith("^["):
            # 引用前加 U+200D（Word 的“无宽非断空格”）：上标引用不与前一字符断开，不会单独出现在行首
            if out:
                out.append(run("‍", rpr(**base)))
            out.append(run(tok[1:-1], rpr(sup=True, **base)))
        elif SYMBOL.fullmatch(tok):
            out.append(run(tok, rpr(ascii_=TNR, east=TNR, **base)))
        elif TOKEN.fullmatch(tok):
            out.append(run(tok, rpr(hint=True, **base)))
        else:
            out.append(run(tok, rpr(**base)))
    return "".join(out)


def para(runs, style=None, ppr_extra="", sect=""):
    ppr = (f'<w:pStyle w:val="{style}"/>' if style else "") + ppr_extra + sect
    return f'<w:p>{f"<w:pPr>{ppr}</w:pPr>" if ppr else ""}{runs}</w:p>'


def field(instr, cached, props=""):
    return (f'<w:r>{props}<w:fldChar w:fldCharType="begin"/></w:r>'
            f'<w:r>{props}<w:instrText xml:space="preserve"> {instr} </w:instrText></w:r>'
            f'<w:r>{props}<w:fldChar w:fldCharType="separate"/></w:r>'
            f'{run(cached, props)}'
            f'<w:r>{props}<w:fldChar w:fldCharType="end"/></w:r>')


# ---------- 图片 ----------
images = []   # (rId, 文件名)


def picture(name, width_pt=None):
    w_pt, h_pt = sizes[name]["pt"]
    if width_pt is None:
        width_pt = min(w_pt, TEXT_W_PT)
    h = h_pt * width_pt / w_pt
    cx, cy = int(width_pt * 12700), int(h * 12700)
    idx = len(images) + 1
    rid = f"rIdImg{idx}"
    images.append((rid, f"{name}.jpg"))
    return (f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
            f'<wp:docPr id="{idx}" name="Picture {idx}"/>'
            f'<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            f'<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:pic><pic:nvPicPr><pic:cNvPr id="{idx}" name="{name}.jpg"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
            f'</a:graphicData></a:graphic></wp:inline></w:drawing></w:r>')


# ---------- 表格 ----------
def border(side, sz):
    return f'<w:{side} w:val="single" w:sz="{sz}" w:space="0" w:color="000000"/>'


NO_BORDERS = ('<w:tblBorders><w:top w:val="nil"/><w:left w:val="nil"/><w:bottom w:val="nil"/>'
              '<w:right w:val="nil"/><w:insideH w:val="nil"/><w:insideV w:val="nil"/></w:tblBorders>')


def tbl(widths, rows_xml, borders, jc="center"):
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    return (f'<w:tbl><w:tblPr><w:tblW w:w="{sum(widths)}" w:type="dxa"/><w:jc w:val="{jc}"/>'
            f'{borders}<w:tblLayout w:type="fixed"/>'
            f'<w:tblLook w:val="0000" w:firstRow="0" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" '
            f'w:noHBand="0" w:noVBand="0"/></w:tblPr><w:tblGrid>{grid}</w:tblGrid>{rows_xml}</w:tbl>')


def tc(width, paras, tc_borders="", valign=None):
    tcpr = f'<w:tcW w:w="{width}" w:type="dxa"/>'
    if tc_borders:
        tcpr += f"<w:tcBorders>{tc_borders}</w:tcBorders>"
    if valign:
        tcpr += f'<w:vAlign w:val="{valign}"/>'
    return f"<w:tc><w:tcPr>{tcpr}</w:tcPr>{paras}</w:tc>"


def tr(cells, header=False):
    trpr = "<w:cantSplit/>" + ("<w:tblHeader/>" if header else "")
    return f"<w:tr><w:trPr>{trpr}</w:trPr>{cells}</w:tr>"


# ---------- 节属性 ----------
def sect_pr(kind):
    if kind == "cover":
        return ('<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
                '<w:pgMar w:top="1440" w:right="1797" w:bottom="1440" w:left="1797" '
                'w:header="1701" w:footer="1304" w:gutter="0"/>'
                '<w:cols w:space="425"/><w:docGrid w:linePitch="395"/></w:sectPr>')
    footer, fmt = ("rIdFtrRoman", "upperRoman") if kind == "toc" else ("rIdFtrArabic", "decimal")
    return (f'<w:sectPr><w:headerReference w:type="default" r:id="rIdHdr"/>'
            f'<w:footerReference w:type="default" r:id="{footer}"/>'
            '<w:pgSz w:w="11906" w:h="16838"/>'
            '<w:pgMar w:top="2155" w:right="1701" w:bottom="1701" w:left="1701" '
            'w:header="1361" w:footer="1304" w:gutter="0"/>'
            f'<w:pgNumType w:fmt="{fmt}" w:start="1"/>'
            '<w:cols w:space="425"/><w:docGrid w:linePitch="391"/></w:sectPr>')


# ---------- 封面 ----------
def cover():
    x = []
    # 四个字体槽都设为 Wingdings 2：私用区码位可能被按东亚文字处理而改用 eastAsia 字体
    box = ('<w:rPr><w:rFonts w:ascii="Wingdings 2" w:eastAsia="Wingdings 2" w:hAnsi="Wingdings 2" '
           'w:cs="Wingdings 2"/></w:rPr>')
    x.append(para(f'<w:r>{box}<w:sym w:font="Wingdings 2" w:char="F052"/></w:r>'
                  + run("毕业论文  ")             # 与模板一致：复选框后不留空格，两项之间两个空格
                  + f'<w:r>{box}<w:sym w:font="Wingdings 2" w:char="F0A3"/></w:r>'
                  + run("毕业设计"), ppr_extra='<w:jc w:val="left"/>'))
    title_ppr = '<w:spacing w:before="0" w:after="0" w:line="288" w:lineRule="auto"/><w:jc w:val="center"/>'
    # 与 2026-09-01 开题报告模板一致：华文新魏 36 磅（小初）、加粗、字符缩放 90%
    title_rpr = rpr(east=XINWEI, bold=True, scale=90, size=36)
    x.append(para(run("哈尔滨工业大学深圳校区", title_rpr),
                  ppr_extra=title_ppr.replace('w:before="0"', 'w:before="1100"')))
    x.append(para(run("毕业论文（设计）开题报告", title_rpr), ppr_extra=title_ppr))
    x.append(para("", ppr_extra='<w:spacing w:before="1600" w:after="0" w:line="240" w:lineRule="auto"/>'))

    W_LABEL, W_COLON, W_VALUE = 1500, 420, 5400
    cell_ppr = '<w:spacing w:before="240" w:after="0" w:line="360" w:lineRule="auto"/>'
    underline = border("bottom", 6)

    def row(label, value, size, value_font):
        lab_r = rpr(east=HEI, size=size)
        val_r = rpr(east=value_font, size=size)
        cells = tc(W_LABEL, para(run(label, lab_r), ppr_extra=cell_ppr + '<w:jc w:val="distribute"/>'),
                   valign="bottom")
        cells += tc(W_COLON, para(run("：" if label else "", lab_r), ppr_extra=cell_ppr), valign="bottom")
        cells += tc(W_VALUE, para(run(value, val_r), ppr_extra=cell_ppr + '<w:jc w:val="center"/>'),
                    tc_borders=underline, valign="bottom")
        return tr(cells)

    rows = row("题目", "基于扩散模型的印刷电路板", 18, HEI)
    rows += row("", "缺陷图像生成方法研究", 18, HEI)
    for label, value in [("姓名", "陈斯萍"), ("学号", "2023313823"), ("学院", "深圳校区机器人与先进制造学院"),
                         ("专业", "机器人工程"), ("指导教师", "王睿"), ("日期", "2026.10.5")]:
        rows += row(label, value, 15, SONG)
    x.append(tbl([W_LABEL, W_COLON, W_VALUE], rows, NO_BORDERS))
    x.append(para("", sect=sect_pr("cover")))
    return "".join(x)


# ---------- 目录 ----------
headings = []   # (级别, 显示文字, 书签名, 缓存页码)


def heading_label(c):
    return f'{c["num"]}  {c["text"]}' if c["num"] else c["text"]


for i, c in enumerate(content):
    if c["t"] in ("h1", "h2", "h3"):
        c["bookmark"] = f"_Toc20261019{len(headings) + 1:03d}"
        headings.append((int(c["t"][1]), heading_label(c), c["bookmark"], c["toc_page"]))


def toc():
    x = [para(run("目  录"), style="TOCTitle")]
    n = len(headings)
    for k, (level, label, bm, page) in enumerate(headings):
        begin = ('<w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r>'
                 '<w:r><w:instrText xml:space="preserve"> TOC \\o "1-3" \\h \\z \\u </w:instrText></w:r>'
                 '<w:r><w:fldChar w:fldCharType="separate"/></w:r>') if k == 0 else ""
        end = '<w:r><w:fldChar w:fldCharType="end"/></w:r>' if k == n - 1 else ""
        pageref = field("PAGEREF " + bm + " \\h", str(page))
        link = (f'<w:hyperlink w:anchor="{bm}" w:history="1">{run(label)}<w:r><w:tab/></w:r>'
                f'{pageref}</w:hyperlink>')
        x.append(para(begin + link + end, style=f"TOC{level}", sect=sect_pr("toc") if k == n - 1 else ""))
    return "".join(x)


# ---------- 正文 ----------
# 图3-2（六幅子图）原尺寸高约 400 磅，放不下时整图移到下一页，留下大片空白；缩小后高约 300 磅
# 图1-1、图1-2 由约 167 磅宽缩至 140 磅（共省约 2 行），使第 1 章末行不单独占一页
FIG_WIDTH_PT = {"fig1": 140, "fig2a": 140, "fig2b": 140, "fig3": 400, "fig4": 225}


def breakable_url(s):
    # 在 URL 的“://”和各“/”之后插入零宽空格，使两端对齐时长 URL 可以断行，不把上一行撑开
    return re.sub(r"(https?://)(\S+)", lambda m: m.group(1) + "​" + m.group(2).replace("/", "/​"), s)


def body():
    x = []
    bm_id = 0
    for c in content:
        t = c["t"]
        if t in ("h1", "h2", "h3"):
            bm_id += 1
            style = {"h1": "Heading1", "h2": "Heading2", "h3": "Heading3"}[t]
            x.append(para(f'<w:bookmarkStart w:id="{bm_id}" w:name="{c["bookmark"]}"/>'
                          f'{text_runs(heading_label(c))}<w:bookmarkEnd w:id="{bm_id}"/>', style=style))
        elif t == "p":
            x.append(para(text_runs(c["text"]), style="BodyIndent"))
        elif t == "cap":
            # 图号与图题间 2 个半角空格；子图标注换行置于图题下方
            sub = '<w:r><w:br/></w:r>' + text_runs(c["sub"]) if c.get("sub") else ""
            x.append(para(run(f'图{c["num"]}  ') + text_runs(c["text"]) + sub, style="FigCaption"))
        elif t == "fig" and c["name"] == "fig2":
            half = TEXT_W // 2
            cells = ""
            for sub, label in (("fig2a", "（a）开路缺陷"), ("fig2b", "（b）缺孔缺陷")):
                cells += tc(half, para(picture(sub, FIG_WIDTH_PT.get(sub)), style="FigurePara")
                            + para(text_runs(label), style="SubCaption", ppr_extra="<w:keepNext/>"),
                            valign="bottom")
            x.append(tbl([half, half], tr(cells), NO_BORDERS))
        elif t == "fig":
            x.append(para(picture(c["name"], FIG_WIDTH_PT.get(c["name"])), style="FigurePara"))
        elif t == "table":
            widths = [2100, 4204, 2200]
            rows = ""
            for r_i, row_cells in enumerate(c["rows"]):
                is_head = r_i == 0
                cells = ""
                for k, (w, txt) in enumerate(zip(widths, row_cells)):
                    if is_head:
                        ppr = '<w:jc w:val="center"/>'
                    elif k > 0:
                        # 表内文字说明首行缩进 2 个半角字符（§2.12）
                        ppr = '<w:ind w:firstLineChars="100" w:firstLine="210"/>'
                    else:
                        ppr = ""
                    b = border("bottom", 6) if is_head else ""
                    cells += tc(w, para(text_runs(txt, bold=is_head), style="TableText", ppr_extra=ppr),
                                tc_borders=b, valign="center")
                rows += tr(cells, header=is_head)
            three_line = (f'<w:tblBorders>{border("top", 12)}<w:left w:val="nil"/>{border("bottom", 12)}'
                          '<w:right w:val="nil"/><w:insideH w:val="nil"/><w:insideV w:val="nil"/></w:tblBorders>')
            # 表与上下文之间空一行；表题置于表上方
            x.append(para(""))
            x.append(para(run(f'表{c["num"]}  {c["title"]}'), style="TableCaption"))
            x.append(tbl(widths, rows, three_line))
            x.append(para(""))
        elif t == "ref":
            x.append(para(run(f'[{c["n"]}]') + '<w:r><w:tab/></w:r>' + text_runs(breakable_url(c["text"])),
                          style="RefEntry"))
    return "".join(x)


# ---------- 样式 ----------
def pstyle(sid, name, ppr, rpr_xml="", based="Normal", nxt=None, ui=None, extra=""):
    return (f'<w:style w:type="paragraph" w:styleId="{sid}"><w:name w:val="{name}"/>'
            + (f'<w:basedOn w:val="{based}"/>' if based else "")
            + (f'<w:next w:val="{nxt}"/>' if nxt else "")
            + (f'<w:uiPriority w:val="{ui}"/>' if ui is not None else "")
            + extra + f"<w:qFormat/><w:pPr>{ppr}</w:pPr>{rpr_xml}</w:style>")


def spacing(before_lines, after_lines, line=300):
    # 同时写“行”与“磅”（1 行按 12 磅），与模板一致；Word 以 *Lines 为准
    b, a = before_lines, after_lines
    return (f'<w:spacing w:before="{int(b * 240)}" w:beforeLines="{int(b * 100)}" '
            f'w:after="{int(a * 240)}" w:afterLines="{int(a * 100)}" w:line="{line}" w:lineRule="auto"/>')


FIVE = '<w:rPr><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'   # 五号 10.5 磅


def heading_rpr(size):
    return f'<w:rPr><w:rFonts w:eastAsia="{HEI}"/><w:sz w:val="{size * 2}"/><w:szCs w:val="{size * 2}"/></w:rPr>'


def toc_style(level, left, hanging, east):
    rp = f'<w:rPr><w:rFonts w:eastAsia="{east}"/></w:rPr>' if east != SONG else ""
    return pstyle(f"TOC{level}", f"toc {level}",
                  f'<w:tabs><w:tab w:val="right" w:leader="dot" w:pos="{TEXT_W}"/></w:tabs>'
                  # 固定 18 磅：31 条目录可排在一页内（模板红字只规定目录字体字号，未规定行距）
                  f'<w:spacing w:before="0" w:after="0" w:line="360" w:lineRule="exact"/>'
                  f'<w:ind w:left="{left}" w:hanging="{hanging}"/>', rp, nxt="Normal", ui=39,
                  extra="<w:unhideWhenUsed/>")


STYLES = (
    f'<w:styles {NS}><w:docDefaults><w:rPrDefault><w:rPr>'
    f'<w:rFonts w:ascii="{TNR}" w:eastAsia="{SONG}" w:hAnsi="{TNR}" w:cs="{TNR}"/>'
    '<w:kern w:val="2"/><w:sz w:val="24"/><w:szCs w:val="24"/>'
    '<w:lang w:val="en-US" w:eastAsia="zh-CN" w:bidi="ar-SA"/></w:rPr></w:rPrDefault>'
    '<w:pPrDefault><w:pPr><w:widowControl w:val="0"/><w:jc w:val="both"/></w:pPr></w:pPrDefault></w:docDefaults>'
    # 正文：宋体小四，1.25 倍，段前段后 0
    + pstyle("Normal", "Normal", '<w:spacing w:before="0" w:after="0" w:line="300" w:lineRule="auto"/>',
             based=None).replace('<w:style w:type="paragraph"', '<w:style w:type="paragraph" w:default="1"', 1)
    + pstyle("BodyIndent", "正文首行缩进", '<w:ind w:firstLineChars="200" w:firstLine="480"/>', nxt="BodyIndent")
    # 章：黑体小二，1.25 倍，段前 1 行，段后 0.8 行，每章另起一页
    + pstyle("Heading1", "heading 1",
             '<w:keepNext/><w:keepLines/><w:pageBreakBefore/>' + spacing(1, 0.8)
             + '<w:jc w:val="center"/><w:outlineLvl w:val="0"/>', heading_rpr(18), nxt="BodyIndent", ui=9)
    # 节：黑体小三，段前段后 0.5 行
    + pstyle("Heading2", "heading 2", '<w:keepNext/><w:keepLines/>' + spacing(0.5, 0.5)
             + '<w:outlineLvl w:val="1"/>', heading_rpr(15), nxt="BodyIndent", ui=9)
    # 条：黑体四号，段前段后 0.5 行
    + pstyle("Heading3", "heading 3", '<w:keepNext/><w:keepLines/>' + spacing(0.5, 0.5)
             + '<w:outlineLvl w:val="2"/>', heading_rpr(14), nxt="BodyIndent", ui=9)
    # 款、项：黑体小四，段前段后 0 行
    + pstyle("Heading4", "heading 4", '<w:keepNext/><w:keepLines/>' + spacing(0, 0)
             + '<w:outlineLvl w:val="3"/>', heading_rpr(12), nxt="BodyIndent", ui=9)
    + pstyle("TOCTitle", "目录标题", '<w:keepNext/>' + spacing(1, 0.8) + '<w:jc w:val="center"/>',
             heading_rpr(18), nxt="Normal")
    # 目录：章黑体小四，其余宋体小四；悬挂缩进使换行后对齐标题首字
    + toc_style(1, 240, 240, HEI)
    + toc_style(2, 660, 420, SONG)
    + toc_style(3, 1080, 600, SONG)
    + pstyle("FigurePara", "图片", '<w:keepNext/><w:spacing w:before="120" w:after="0" w:line="240" '
             'w:lineRule="auto"/><w:jc w:val="center"/>')
    # 图题、表题、表内文字：宋体五号（§2.12、§2.13.1）
    + pstyle("FigCaption", "图表名", '<w:keepLines/><w:spacing w:before="120" w:after="156" w:line="240" '
             'w:lineRule="auto"/><w:jc w:val="center"/>', FIVE)
    + pstyle("TableCaption", "表名", '<w:keepNext/><w:keepLines/><w:spacing w:before="0" w:after="60" '
             'w:line="240" w:lineRule="auto"/><w:jc w:val="center"/>', FIVE)
    + pstyle("SubCaption", "子图题", '<w:spacing w:before="60" w:after="0" w:line="240" w:lineRule="auto"/>'
             '<w:jc w:val="center"/>', '<w:rPr><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>')
    + pstyle("TableText", "表格文字", '<w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/>', FIVE)
    # 两端对齐（指导教师意见）；URL 内的断行点由 breakable_url 提供
    + pstyle("RefEntry", "参考文献条目", '<w:ind w:left="500" w:hanging="500"/><w:jc w:val="both"/>')
    # 页眉下方上细下粗双线，与开题报告模板一致（thinThickSmallGap 3 磅）
    + pstyle("Header", "header", '<w:pBdr><w:bottom w:val="thinThickSmallGap" w:sz="24" w:space="1" '
             'w:color="000000"/></w:pBdr><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/>'
             '<w:jc w:val="center"/>', '<w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>', ui=99,
             extra="<w:unhideWhenUsed/>")
    + pstyle("Footer", "footer", '<w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/>'
             '<w:jc w:val="center"/>', '<w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>', ui=99,
             extra="<w:unhideWhenUsed/>")
    + '<w:style w:type="character" w:default="1" w:styleId="DefaultParagraphFont">'
      '<w:name w:val="Default Paragraph Font"/><w:uiPriority w:val="1"/><w:semiHidden/><w:unhideWhenUsed/></w:style>'
    + '<w:style w:type="table" w:default="1" w:styleId="TableNormal"><w:name w:val="Normal Table"/>'
      '<w:uiPriority w:val="99"/><w:semiHidden/><w:unhideWhenUsed/><w:tblPr><w:tblInd w:w="0" w:type="dxa"/>'
      '<w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:left w:w="108" w:type="dxa"/>'
      '<w:bottom w:w="0" w:type="dxa"/><w:right w:w="108" w:type="dxa"/></w:tblCellMar></w:tblPr></w:style>'
    + "</w:styles>"
)

# ---------- 其他部件 ----------
HEADER = f'<w:hdr {NS}>{para(run(HEADER_TEXT), style="Header")}</w:hdr>'
FOOTER_ROMAN = f'<w:ftr {NS}>{para(field("PAGE", "I"), style="Footer")}</w:ftr>'
FOOTER_ARABIC = f'<w:ftr {NS}>{para(run("- ") + field("PAGE", "1") + run(" -"), style="Footer")}</w:ftr>'

SETTINGS = (f'<w:settings {NS}><w:zoom w:percent="100"/><w:defaultTabStop w:val="420"/>'
            '<w:characterSpacingControl w:val="compressPunctuation"/><w:updateFields w:val="true"/>'
            '<w:compat><w:compatSetting w:name="compatibilityMode" w:uri="http://schemas.microsoft.com/office/word" '
            'w:val="15"/></w:compat></w:settings>')

FONTS = (f'<w:fonts {NS}>'
         f'<w:font w:name="{TNR}"><w:panose1 w:val="02020603050405020304"/><w:charset w:val="00"/>'
         '<w:family w:val="roman"/><w:pitch w:val="variable"/></w:font>'
         f'<w:font w:name="{SONG}"><w:altName w:val="SimSun"/><w:panose1 w:val="02010600030101010101"/>'
         '<w:charset w:val="86"/><w:family w:val="auto"/><w:pitch w:val="variable"/></w:font>'
         f'<w:font w:name="{HEI}"><w:altName w:val="SimHei"/><w:panose1 w:val="02010609060101010101"/>'
         '<w:charset w:val="86"/><w:family w:val="modern"/><w:pitch w:val="fixed"/></w:font>'
         f'<w:font w:name="{XINWEI}"><w:altName w:val="STXinwei"/><w:charset w:val="86"/>'
         '<w:family w:val="auto"/><w:pitch w:val="variable"/></w:font>'
         '<w:font w:name="Wingdings 2"><w:panose1 w:val="05020102010507070707"/><w:charset w:val="02"/>'
         '<w:family w:val="roman"/><w:pitch w:val="variable"/></w:font></w:fonts>')

document = f'<w:document {NS}><w:body>{cover()}{toc()}{body()}{sect_pr("body")}</w:body></w:document>'

REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
doc_rels = (f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rIdStyles" Type="{REL}/styles" Target="styles.xml"/>'
            f'<Relationship Id="rIdSettings" Type="{REL}/settings" Target="settings.xml"/>'
            f'<Relationship Id="rIdFonts" Type="{REL}/fontTable" Target="fontTable.xml"/>'
            f'<Relationship Id="rIdHdr" Type="{REL}/header" Target="header1.xml"/>'
            f'<Relationship Id="rIdFtrRoman" Type="{REL}/footer" Target="footer1.xml"/>'
            f'<Relationship Id="rIdFtrArabic" Type="{REL}/footer" Target="footer2.xml"/>'
            + "".join(f'<Relationship Id="{rid}" Type="{REL}/image" Target="media/{fn}"/>' for rid, fn in images)
            + "</Relationships>")

CT = "application/vnd.openxmlformats-officedocument.wordprocessingml"
content_types = ('<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                 '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                 '<Default Extension="xml" ContentType="application/xml"/>'
                 '<Default Extension="jpg" ContentType="image/jpeg"/>'
                 f'<Override PartName="/word/document.xml" ContentType="{CT}.document.main+xml"/>'
                 f'<Override PartName="/word/styles.xml" ContentType="{CT}.styles+xml"/>'
                 f'<Override PartName="/word/settings.xml" ContentType="{CT}.settings+xml"/>'
                 f'<Override PartName="/word/fontTable.xml" ContentType="{CT}.fontTable+xml"/>'
                 f'<Override PartName="/word/header1.xml" ContentType="{CT}.header+xml"/>'
                 f'<Override PartName="/word/footer1.xml" ContentType="{CT}.footer+xml"/>'
                 f'<Override PartName="/word/footer2.xml" ContentType="{CT}.footer+xml"/>'
                 '<Override PartName="/docProps/core.xml" '
                 'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
                 '<Override PartName="/docProps/app.xml" '
                 'ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
                 '</Types>')

root_rels = ('<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
             f'<Relationship Id="rId1" Type="{REL}/officeDocument" Target="word/document.xml"/>'
             '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/'
             'metadata/core-properties" Target="docProps/core.xml"/>'
             f'<Relationship Id="rId3" Type="{REL}/extended-properties" Target="docProps/app.xml"/>'
             '</Relationships>')

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
core = ('<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<dc:title>基于扩散模型的印刷电路板缺陷图像生成方法研究</dc:title>'
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>')
app = ('<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
       '<Application>Microsoft Office Word</Application></Properties>')

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("[Content_Types].xml", XML_HEAD + content_types)
    z.writestr("_rels/.rels", XML_HEAD + root_rels)
    z.writestr("docProps/core.xml", XML_HEAD + core)
    z.writestr("docProps/app.xml", XML_HEAD + app)
    z.writestr("word/document.xml", XML_HEAD + document)
    z.writestr("word/styles.xml", XML_HEAD + STYLES)
    z.writestr("word/settings.xml", XML_HEAD + SETTINGS)
    z.writestr("word/fontTable.xml", XML_HEAD + FONTS)
    z.writestr("word/header1.xml", XML_HEAD + HEADER)
    z.writestr("word/footer1.xml", XML_HEAD + FOOTER_ROMAN)
    z.writestr("word/footer2.xml", XML_HEAD + FOOTER_ARABIC)
    z.writestr("word/_rels/document.xml.rels", XML_HEAD + doc_rels)
    for rid, fn in images:
        z.write(f"{HERE}/figures/{fn}", f"word/media/{fn}")
print("wrote", OUT, "headings:", len(headings), "images:", len(images))
