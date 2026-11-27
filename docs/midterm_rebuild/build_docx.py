# 按 2026-09-01 中期报告模板与《写作指南（理工类）》生成中期报告 .docx（仅用标准库手写 OOXML）
# 由 docs/kaiti_rebuild/build_docx.py 改写：封面/页眉改为中期报告，目录行距 1.25 倍，增加公式、符号注释与通用三线表
# 用法：python -I build_docx.py <midterm_rebuild 目录> <输出 .docx>
import sys, json, re, zipfile, datetime
from xml.sax.saxutils import escape

HERE, OUT = sys.argv[1], sys.argv[2]
content = json.load(open(f"{HERE}/content.json"))
sizes = json.load(open(f"{HERE}/figures/sizes.json"))
# 目录页码缓存值：由 toc_pages.py 从渲染 PDF 读出；Word/WPS 打开时更新域后以实际页码为准
try:
    TOC_PAGES = json.load(open(f"{HERE}/toc_pages.json", encoding="utf-8"))
except FileNotFoundError:
    TOC_PAGES = {}

NS = ('xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
      'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
      'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" '
      'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"')
XML_HEAD = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'

TNR, SONG, HEI, XINWEI = "Times New Roman", "宋体", "黑体", "华文新魏"
HEADER_TEXT = "哈尔滨工业大学深圳校区毕业论文（设计）中期报告"
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


TOKEN = re.compile(r"(\$[^$]+\$|\^\[[^\]]+\]\^|[“”‘’…—]+|[→≥])")
# 符号统一用 Times New Roman（模板：“数字及符号用Times New Roman字体”）。不指定时随前一字符取字体，
# 同一符号会在中文后用宋体、在西文后用 Times New Roman，字形不一致
SYMBOL = re.compile(r"[→≥]")


def text_runs(s, **base):
    """拆分上标引用 ^[n]^、符号，以及需按中文字体显示的引号、省略号、破折号"""
    out = []
    for tok in TOKEN.split(s):
        if not tok:
            continue
        if tok.startswith("$"):
            out.append(inline_math(tok[1:-1]))
        elif tok.startswith("^["):
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
                '<w:pgMar w:top="1440" w:right="1800" w:bottom="1440" w:left="1800" '   # 与中期报告模板封面节一致
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
    # 与 2026-09-01 中期报告模板一致：华文新魏 36 磅（小初）、加粗、字符缩放 90%
    title_rpr = rpr(east=XINWEI, bold=True, scale=90, size=36)
    x.append(para(run("哈尔滨工业大学深圳校区", title_rpr),
                  ppr_extra=title_ppr.replace('w:before="0"', 'w:before="1100"')))
    x.append(para(run("毕业论文（设计）中期报告", title_rpr), ppr_extra=title_ppr))
    # 2026-11-27：用户在 Word 中删去了标题与信息表之间的 80 磅空段（Word 中封面因此不再溢出到第二页），此处同步删去

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
                         ("专业", "机器人工程"), ("指导教师", "王睿"), ("日期", "2026.11.25")]:
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
        c["bookmark"] = f"_Toc20261113{len(headings) + 1:03d}"
        headings.append((int(c["t"][1]), heading_label(c), c["bookmark"], TOC_PAGES.get(heading_label(c), 1)))


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


# ---------- 公式（OMML） ----------
# 支持的 LaTeX 子集：\frac \sqrt \bar \hat \mathrm \mathbf、_ ^、希腊字母与少量运算符。
# 所有数学文字用 m:nor + Times New Roman（模板：“全文所有数字、英文：Times New Roman字体”）；
# 变量（拉丁、希腊字母）斜体，数字、运算符、\mathrm 内文字正体（指南 §2.9）
GREEK = {"alpha": "α", "epsilon": "ε", "theta": "θ", "lambda": "λ", "eta": "η", "sigma": "σ", "beta": "β", "gamma": "γ", "tau": "τ", "mu": "μ", "rho": "ρ", "Sigma": "Σ"}
OPS = {"odot": "⊙", "sim": "~", "times": "×", "cdot": "·", "quad": " ", "leftarrow": "←", "nabla": "∇",
       "lVert": "‖", "rVert": "‖", "cap": "∩", "cup": "∪", "ge": "≥", "le": "≤", "dots": "…", "mid": "|", "prime": "′"}
NARY = {"sum": "∑", "prod": "∏"}                     # 求和、连乘：上下限写在符号上下方
MATH_TOKEN = re.compile(r"\\[A-Za-z]+|\\,|[{}_^]|\S")      # 空白只用于分隔命令，不输出


def mrun(text, italic=False, bold=False):
    r = f'<w:rFonts w:ascii="{TNR}" w:hAnsi="{TNR}" w:eastAsia="{TNR}" w:cs="{TNR}"/>'
    r += ("<w:b/>" if bold else "") + ("<w:i/>" if italic else "")
    return f'<m:r><m:rPr><m:nor/></m:rPr><w:rPr>{r}</w:rPr><m:t xml:space="preserve">{escape(text)}</m:t></m:r>'


class Tex:
    def __init__(self, src):
        self.toks = MATH_TOKEN.findall(src)
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def group(self):
        """读取 {…} 或单个记号，返回 OMML 字符串"""
        if self.peek() == "{":
            self.take()
            out = self.seq("}")
            self.take()
            return out
        return self.atom()

    def raw_group(self):
        assert self.take() == "{"
        out = []
        while self.peek() != "}":
            out.append(self.take())
        self.take()
        return "".join(out)

    def atom(self):
        t = self.take()
        if t == "\\frac":
            return f"<m:f><m:num>{self.group()}</m:num><m:den>{self.group()}</m:den></m:f>"
        if t == "\\sqrt":
            return f'<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/><m:e>{self.group()}</m:e></m:rad>'
        if t == "\\bar":
            return f'<m:bar><m:barPr><m:pos m:val="top"/></m:barPr><m:e>{self.group()}</m:e></m:bar>'
        if t == "\\hat":
            return f'<m:acc><m:accPr><m:chr m:val="&#770;"/></m:accPr><m:e>{self.group()}</m:e></m:acc>'
        if t == "\\mathrm":
            return mrun(self.raw_group())
        if t == "\\mathbf":
            return mrun(self.raw_group(), bold=True)
        if t[1:] in NARY:
            sub = sup = ""
            while self.peek() in ("_", "^"):
                if self.take() == "_":
                    sub = self.group()
                else:
                    sup = self.group()
            body = self.group()
            hide = ("" if sub else '<m:subHide m:val="1"/>') + ("" if sup else '<m:supHide m:val="1"/>')
            return (f'<m:nary><m:naryPr><m:chr m:val="{NARY[t[1:]]}"/><m:limLoc m:val="undOvr"/>{hide}</m:naryPr>'
                    f'<m:sub>{sub}</m:sub><m:sup>{sup}</m:sup><m:e>{body}</m:e></m:nary>')
        if t[1:] in GREEK:
            return mrun(GREEK[t[1:]], italic=t[1] != "S")       # 大写 Σ（协方差）正体
        if t[1:] in OPS:
            return mrun(OPS[t[1:]])
        if t == "\\,":
            return mrun(" ")
        if t.startswith("\\"):
            raise ValueError("不支持的命令 " + t)
        if t == "-":
            return mrun("−")
        if t == ",":
            return mrun(", ")
        if t in "=+<>~":
            return mrun(f" {t} " if t in "=+<>" else t)
        return mrun(t, italic=t.isalpha())

    def seq(self, stop=None):
        out = []
        while self.peek() is not None and self.peek() != stop:
            base = self.atom() if self.peek() != "{" else self.group()
            sub = sup = None
            while self.peek() in ("_", "^"):
                if self.take() == "_":
                    sub = self.group()
                else:
                    sup = self.group()
            if sub and sup:
                base = f"<m:sSubSup><m:e>{base}</m:e><m:sub>{sub}</m:sub><m:sup>{sup}</m:sup></m:sSubSup>"
            elif sub:
                base = f"<m:sSub><m:e>{base}</m:e><m:sub>{sub}</m:sub></m:sSub>"
            elif sup:
                base = f"<m:sSup><m:e>{base}</m:e><m:sup>{sup}</m:sup></m:sSup>"
            out.append(base)
        return "".join(out)


def omath(tex):
    t = Tex(tex)
    out = t.seq()
    assert t.peek() is None, tex
    return f"<m:oMath>{out}</m:oMath>"


# 行内符号不用公式对象，直接写成 Times New Roman 文字块：变量斜体，下标/上标用 vertAlign。
# 公式对象在行内会改变行高，且部分软件（如 LibreOffice）转换后丢失斜体
COMBINING = {"\\hat": "\u0302", "\\bar": "\u0304"}


def inline_math(tex):
    toks = MATH_TOKEN.findall(tex)
    pos = [0]
    pieces = []       # (文字, 斜体, 上下标)

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def take():
        pos[0] += 1
        return toks[pos[0] - 1]

    def group(vert):
        if peek() == "{":
            take()
            while peek() != "}":
                item(vert)
            take()
        else:
            item(vert)

    def item(vert):
        t = take()
        if t in COMBINING:
            start = len(pieces)
            group(vert)
            txt, it, v = pieces[start]
            pieces[start] = (txt[0] + COMBINING[t] + txt[1:], it, v)
        elif t in ("\\mathrm", "\\mathbf"):
            assert take() == "{"
            word = []
            while peek() != "}":
                word.append(take())
            take()
            pieces.append(("".join(word), False, vert))
        elif t[1:] in GREEK:
            pieces.append((GREEK[t[1:]], True, vert))
        elif t[1:] in OPS:
            pieces.append((OPS[t[1:]], False, vert))
        elif t.startswith("\\"):
            raise ValueError("不支持的命令 " + t)
        elif t in "{}":
            raise ValueError(tex)
        else:
            pieces.append(({"-": "−", "<": " < ", ",": ", "}.get(t, t), t.isalpha(), vert))
        while peek() in ("_", "^"):
            group("subscript" if take() == "_" else "superscript")

    while peek() is not None:
        item(None)
    out = []
    for txt, it, v in pieces:
        r = f'<w:rFonts w:ascii="{TNR}" w:hAnsi="{TNR}" w:eastAsia="{TNR}" w:cs="{TNR}"/>'
        r += ("<w:i/><w:iCs/>" if it else "") + (f'<w:vertAlign w:val="{v}"/>' if v else "")
        out.append(f'<w:r><w:rPr>{r}</w:rPr><w:t xml:space="preserve">{escape(txt)}</w:t></w:r>')
    return "".join(out)


# ---------- 正文 ----------
# 宽度（磅）：版心宽 425 磅；按图的信息量选择，避免整图移页留下大片空白
FIG_WIDTH_PT = {"fig_pre_overview": 400, "fig_pre_patches": 150, "fig_pre_rescale": 230, "fig_pre_clean": 230,
                "fig_pre_coverage": 400, "fig_pre_free": 425, "fig_64_samples": 360,
                "fig_64_process": 360, "fig_256_samples": 400, "fig_crg_input": 170, "fig_crg_pairs": 330,
                "fig_methods": 300, "fig_kid": 420, "fig_crg": 300, "route_mid": 425, "op_ddpm": 420,
                "fig_principle": 430, "fig_mask_steps": 420}
THREE_LINE = (f'<w:tblBorders>{border("top", 12)}<w:left w:val="nil"/>{border("bottom", 12)}'
              '<w:right w:val="nil"/><w:insideH w:val="nil"/><w:insideV w:val="nil"/></w:tblBorders>')
# 公式：居中制表位放公式，右对齐制表位放编号（指南 §2.10：公式居中、编号右对齐）
EQ_TABS = (f'<w:tabs><w:tab w:val="center" w:pos="{TEXT_W // 2}"/>'
           f'<w:tab w:val="right" w:pos="{TEXT_W}"/></w:tabs>')
NOTE_IND = 720      # 符号列位置：“式中”两字（480 twips）后再留两个半角空格（240 twips）


def breakable_url(s):
    # 在 URL 的“://”和各“/”之后插入零宽空格，使两端对齐时长 URL 可以断行，不把上一行撑开
    return re.sub(r"(https?://)(\S+)", lambda m: m.group(1) + "​" + m.group(2).replace("/", "/​"), s)


def breakable_ref(s):
    # 除 URL 外，在“[C]//”“[J]//”等之后也加零宽空格：否则“Models[C]//Advances”整体不可断，前一行被拉松
    return breakable_url(re.sub(r"\]//", "]//​", s))


def table(c):
    widths = c["widths"]
    rows = ""
    for r_i, row_cells in enumerate(c["rows"]):
        is_head = r_i == 0
        cells = ""
        for k, (w, txt) in enumerate(zip(widths, row_cells)):
            if is_head or c["align"] == "center" or k == 0:
                ppr = '<w:jc w:val="center"/>'
            else:
                # 表内文字说明首行缩进 2 个半角字符（§2.12）
                ppr = '<w:ind w:firstLineChars="100" w:firstLine="210"/>'
            b = border("bottom", 6) if is_head else ""
            if r_i < len(c["rows"]) - 1:
                ppr += "<w:keepNext/>"            # 整表不跨页（表较短）
            # 单元格内 “\n” 为手动换行（表头较长时按词分两行，避免单字落到第二行）
            runs = '<w:r><w:br/></w:r>'.join(text_runs(seg) for seg in txt.split("\n"))
            cells += tc(w, para(runs, style="TableText", ppr_extra=ppr), tc_borders=b, valign="center")
        rows += tr(cells, header=is_head)
    x = [para(""),                                   # 表与上下文之间空一行；表题置于表上方
         para(run(f'表{c["num"]}  {c["title"]}'), style="TableCaption"),
         tbl(widths, rows, THREE_LINE)]
    if c.get("note"):
        x.append(para(text_runs(c["note"]), style="TableText", ppr_extra='<w:spacing w:before="60"/>'))
    x.append(para(""))
    return "".join(x)


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
            # 图号与图题间 2 个半角空格
            x.append(para(run(f'图{c["num"]}  ') + text_runs(c["text"]), style="FigCaption"))
        elif t == "subcap":
            # 分图图题列在主图题之下（指南 §2.13.1）
            x.append(para(text_runs(c["text"]), style="SubCaption"))
        elif t == "fig":
            x.append(para(picture(c["name"], FIG_WIDTH_PT.get(c["name"])), style="FigurePara"))
        elif t == "eq":
            x.append(para('<w:r><w:tab/></w:r>' + omath(c["tex"]) + '<w:r><w:tab/></w:r>'
                          + run(f'({c["num"]})'), style="Equation"))
        elif t == "note":
            # 符号注释（模板示例与指南 §2.10）：首行“式中  ”，各符号对齐，破折号“——”后接说明
            for k, (sym, desc) in enumerate(c["items"]):
                lead = run("式中") + "<w:r><w:tab/></w:r>" if k == 0 else ""
                ind = (f'<w:ind w:left="{NOTE_IND}" w:hanging="{NOTE_IND}"/>' if k == 0
                       else f'<w:ind w:left="{NOTE_IND}"/>')
                x.append(para(lead + inline_math(sym) + text_runs("——" + desc), style="EqNote", ppr_extra=ind))
        elif t == "table":
            x.append(table(c))
        elif t == "ref":
            x.append(para(run(f'[{c["n"]}]') + '<w:r><w:tab/></w:r>' + text_runs(breakable_ref(c["text"])),
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
                  # 与中期报告模板“目录 1～3”样式一致：1.25 倍行距
                  f'<w:spacing w:before="0" w:after="0" w:line="300" w:lineRule="auto"/>'
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
    + pstyle("Equation", "公式", '<w:keepLines/>' + EQ_TABS
             + '<w:spacing w:before="60" w:after="60" w:line="300" w:lineRule="auto"/><w:jc w:val="left"/>')
    + pstyle("EqNote", "符号注释", '<w:jc w:val="left"/>')
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
    # 页眉下方双线，与中期报告模板一致（thinThickSmallGap 3 磅）
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
