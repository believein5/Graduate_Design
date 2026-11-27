# 用学校的毕业论文 WORD 模板（.dotx）生成毕业论文 .docx（2026-11-26）
# 用法：python -I build.py <输出.docx> [--pages <渲染出的PDF>]
#   --pages：从 LibreOffice 导出的 PDF 读出各标题所在页码，写入 toc_pages.json，供下一次生成作为目录页码的缓存值
# 做法：模板 document.xml 中保留封面、扉页、摘要标题、目录标题、原创性声明及各节的分节符、页眉页脚；
#      删去“书写范例”说明页和示例正文，按 content.py 生成正文。所有段落使用模板的样式 ID；
#      章、节、条标题和图题、表题由模板的多级列表自动编号，公式编号用模板的 STYLEREF/SEQ 域，参考文献用模板的自动编号。
import ast, copy, json, re, subprocess, sys, zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from lxml import etree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TPL = next((ROOT / "report" / "template").glob("*WORD模板.dotx"))
sys.path.insert(0, str(HERE))
import content as C

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
w = lambda t: f"{{{W}}}{t}"

# ---------- 复用中期报告生成程序中的文字块与公式转换（只取需要的定义）----------
_src = (ROOT / "docs" / "midterm_rebuild" / "build_docx.py").read_text(encoding="utf-8")
_want = {"rpr", "run", "text_runs", "mrun", "Tex", "omath", "inline_math"}
_nodes = []
for n in ast.parse(_src).body:
    if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in _want:
        _nodes.append(n)
    elif isinstance(n, ast.Assign):
        names = {e.id for t in n.targets for e in (t.elts if isinstance(t, ast.Tuple) else [t]) if isinstance(e, ast.Name)}
        if names & {"TNR", "TOKEN", "SYMBOL", "GREEK", "OPS", "NARY", "MATH_TOKEN", "COMBINING"}:
            _nodes.append(n)
H = {"re": re, "escape": escape}
exec(compile(ast.Module(body=_nodes, type_ignores=[]), "build_docx.py", "exec"), H)
text_runs, omath, inline_math, run = H["text_runs"], H["omath"], H["inline_math"], H["run"]

# ---------- 引用与交叉引用编号 ----------
CITE = re.compile(r"\^\[([^\]]+)\]\^")
order = []


def _scan(s):
    for m in CITE.finditer(s):
        for k in m.group(1).split(","):
            if k not in order:
                order.append(k)


def _texts(item):
    if item[0] in ("p",):
        yield item[1]
    elif item[0] == "fig":
        yield item[4]
    elif item[0] == "tab":
        yield item[2]; yield item[5]


for it in C.BODY:
    for s in _texts(it):
        _scan(s)
missing = [k for k in order if k not in C.REFS]
assert not missing, missing
NUM = {k: i for i, k in enumerate(order, 1)}


def compress(ns):
    ns, out, k = sorted(ns), [], 0
    while k < len(ns):
        j = k
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        out.append(str(ns[k]) if j == k else (f"{ns[k]},{ns[j]}" if j == k + 1 else f"{ns[k]}-{ns[j]}")); k = j + 1
    return ",".join(out)


XREF, ch = {}, 0
cnt = {"fig": 0, "tab": 0, "eq": 0}
for it in C.BODY:
    if it[0] == "h1":
        ch += 1; cnt = {k: 0 for k in cnt}
    elif it[0] in cnt:
        cnt[it[0]] += 1
        XREF[(it[0], it[1])] = f"{ch}-{cnt[it[0]]}"
KIND = {"图": "fig", "表": "tab", "式": "eq"}


def resolve(s):
    s = CITE.sub(lambda m: "^[" + compress([NUM[k] for k in m.group(1).split(",")]) + "]^", s)
    def xr(m):
        n = XREF[(KIND[m.group(1)], m.group(2))]
        return f"式({n})" if m.group(1) == "式" else f"{m.group(1)}{n}"
    return re.sub(r"\{(图|表|式):([^}]+)\}", xr, s)


HL = '<w:rPr><w:highlight w:val="yellow"/></w:rPr>'


def runs(s):
    """正文文字：【...】为待补充（黄色高亮），其余按中期报告的规则拆分上标引用、行内公式和中文标点"""
    out = []
    for part in re.split(r"(【[^】]*】)", resolve(s)):
        if part.startswith("【"):
            out.append(run(part, HL))
        elif part:
            out.append(text_runs(part))
    return "".join(out)


# ---------- 段落、图、表、公式 ----------
def para(body, style=None, ppr=""):
    st = f'<w:pStyle w:val="{style}"/>' if style else ""
    return f"<w:p><w:pPr>{st}{ppr}</w:pPr>{body}</w:p>"


TOC = []          # (级别, 显示文字, 书签名)
_bm = [9000]


def bookmark(body):
    _bm[0] += 1
    name = f"_Toc9{_bm[0]:06d}"
    return name, f'<w:bookmarkStart w:id="{_bm[0]}" w:name="{name}"/>{body}<w:bookmarkEnd w:id="{_bm[0]}"/>'


def spaced(t):
    """两个字的章标题字间加两个空格（与模板“绪  论”“结  论”一致）"""
    return f"{t[0]}  {t[1]}" if len(t) == 2 else t


SIZES = json.load(open(HERE / "figures" / "sizes.json"))
IMAGES = []       # (rId, 文件名)


def picture(name, width_pt):
    wpx, hpx = SIZES[name]["px"]
    cx, cy = int(width_pt * 12700), int(width_pt * hpx / wpx * 12700)
    idx = len(IMAGES) + 1
    rid = f"rIdThesis{idx}"
    IMAGES.append((rid, f"{name}.jpg"))
    did = 5000 + idx
    return (f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
            f'<wp:effectExtent l="0" t="0" r="0" b="0"/><wp:docPr id="{did}" name="图片 {did}"/>'
            f'<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            f'<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="{did}" name="{name}.jpg"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
            f'</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>')


def cell(width, body, borders=""):
    return (f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{borders}<w:vAlign w:val="center"/></w:tcPr>{body}</w:tc>')


def b(side, sz):
    return f'<w:{side} w:val="single" w:color="auto" w:sz="{sz}" w:space="0"/>' if sz else f'<w:{side} w:val="nil"/>'


def three_line(widths, rows):
    """三线表：顶线、底线 1.5 磅，栏目线 1 磅（模板说明），单元格样式“表正文”"""
    out = []
    for i, row in enumerate(rows):
        top = 12 if i == 0 else 0
        bottom = 8 if i == 0 else (12 if i == len(rows) - 1 else 0)
        brd = f'<w:tcBorders>{b("top", top)}{b("left", 0)}{b("bottom", bottom)}{b("right", 0)}</w:tcBorders>'
        keep = "<w:keepNext/>" if i < len(rows) - 1 else ""          # 整张表不跨页（指南 §2.12）
        cells = "".join(cell(wd, "".join(para(runs(line), "22", keep) for line in str(v).split("\n")), brd) for wd, v in zip(widths, row))
        hdr = "<w:tblHeader/>" if i == 0 else ""
        out.append(f'<w:tr><w:trPr><w:cantSplit/>{hdr}<w:jc w:val="center"/></w:trPr>{cells}</w:tr>')
    grid = "".join(f'<w:gridCol w:w="{x}"/>' for x in widths)
    return (f'<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/><w:jc w:val="center"/><w:tblBorders>{b("top", 0)}{b("left", 0)}'
            f'{b("bottom", 0)}{b("right", 0)}<w:insideH w:val="nil"/><w:insideV w:val="nil"/></w:tblBorders><w:tblLayout w:type="fixed"/>'
            f'<w:tblCellMar><w:left w:w="108" w:type="dxa"/><w:right w:w="108" w:type="dxa"/></w:tblCellMar></w:tblPr>'
            f'<w:tblGrid>{grid}</w:tblGrid>{"".join(out)}</w:tbl>')


def fldrun(instr, cached):
    return (f'<w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText xml:space="preserve"> {instr} </w:instrText></w:r>'
            f'<w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>{cached}</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r>')


def equation(tex, chap, n):
    """模板的公式写法：无框表格，左右两格各 1090，中间公式居中，右格“公式题注”样式，编号为 STYLEREF 1 \\s - SEQ 公式 \\s 1"""
    nb = (run("（") + fldrun("STYLEREF 1 \\s", chap) + run("-") + fldrun("SEQ 公式 \\* ARABIC \\s 1", n) + run("）"))
    none = f'<w:tcBorders>{b("top", 0)}{b("left", 0)}{b("bottom", 0)}{b("right", 0)}</w:tcBorders>'
    row = (cell(1090, para("", None, '<w:jc w:val="center"/>'), none)
           + cell(6324, para(omath(tex), "3", '<w:ind w:firstLine="0" w:firstLineChars="0"/><w:jc w:val="center"/>'), none)
           + cell(1090, para(nb, "20", '<w:jc w:val="right"/>'), none))
    return (f'<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/><w:tblBorders>{b("top", 0)}{b("left", 0)}{b("bottom", 0)}{b("right", 0)}'
            f'<w:insideH w:val="nil"/><w:insideV w:val="nil"/></w:tblBorders><w:tblLayout w:type="fixed"/></w:tblPr>'
            f'<w:tblGrid><w:gridCol w:w="1090"/><w:gridCol w:w="6324"/><w:gridCol w:w="1090"/></w:tblGrid>'
            f'<w:tr><w:trPr><w:cantSplit/><w:trHeight w:val="822" w:hRule="atLeast"/></w:trPr>{row}</w:tr></w:tbl>')


NOTE_IND = 840


def note(items):
    out = []
    for k, (sym, desc) in enumerate(items):
        lead = run("式中") + "<w:r><w:tab/></w:r>" if k == 0 else ""
        ind = (f'<w:ind w:left="{NOTE_IND}" w:hanging="{NOTE_IND}" w:firstLineChars="0"/>' if k == 0
               else f'<w:ind w:left="{NOTE_IND}" w:firstLine="0" w:firstLineChars="0"/>')
        out.append(para(lead + inline_math(sym) + runs("——" + desc), "3", ind))
    return "".join(out)


def body_xml():
    x, chap = [], 0
    for it in C.BODY:
        t = it[0]
        if t in ("h1", "h2", "h3"):
            text = spaced(it[1]) if t == "h1" else it[1]
            if t == "h1":
                chap += 1
            bm, inner = bookmark(run(text))
            x.append(para(inner, {"h1": "2", "h2": "35", "h3": "18"}[t]))
            TOC.append((t, text, bm))
        elif t == "p":
            x.append(para(runs(it[1]), "3"))
        elif t == "fig":
            _, key, file, width, cap, sub = it
            x.append(para(picture(file, width), None, '<w:keepNext/><w:jc w:val="center"/>'))
            if sub:
                x.append(para(runs(sub), "36", '<w:keepNext/>'))
            x.append(para(runs(cap), "37"))
        elif t == "tab":
            _, key, cap, widths, rows, nt = it
            x.append(para(runs(cap), "21", "<w:keepNext/>"))
            x.append(three_line(widths, rows))
            if nt:
                x.append(para(runs(nt), "22", '<w:jc w:val="left"/>'))
            x.append(para("", "3"))
        elif t == "eq":
            c, n = XREF[("eq", it[1])].split("-")
            x.append(equation(it[2], c, n))
        elif t == "note":
            x.append(note(it[1]))
    return "".join(x)


# ---------- 读模板 ----------
zin = zipfile.ZipFile(TPL)
doc = etree.fromstring(zin.read("word/document.xml"))
NSDECL = " ".join(f'xmlns:{k}="{v}"' for k, v in doc.nsmap.items() if k)


def frag(xml):
    return list(etree.fromstring(f"<w:body {NSDECL}>{xml}</w:body>"))


body = doc.find(w("body"))
items = list(body)
txt = lambda e: "".join(e.itertext())
sty = lambda e: (e.find(f"{w('pPr')}/{w('pStyle')}").get(w("val")) if e.find(f"{w('pPr')}/{w('pStyle')}") is not None else None)
first = lambda pred, start=0: next(i for i in range(start, len(items)) if pred(items[i]))

i_sect1 = first(lambda e: e.find(f".//{w('sectPr')}") is not None)
i_abs = first(lambda e: sty(e) == "28" and txt(e).strip() == "摘  要")
i_kw = first(lambda e: txt(e).startswith("关键词"), i_abs)
i_eabs = first(lambda e: sty(e) == "28" and txt(e).strip() == "Abstract")
i_ekw = first(lambda e: txt(e).startswith("Keywords"), i_eabs)
i_toc = first(lambda e: sty(e) == "24")
i_tocsect = first(lambda e: e.find(f".//{w('sectPr')}") is not None, i_toc)
i_concl = first(lambda e: sty(e) == "28" and txt(e).strip() == "结  论")
i_refs = first(lambda e: sty(e) == "28" and txt(e).strip() == "参考文献")
i_ach = first(lambda e: sty(e) == "28" and txt(e).startswith("攻读学士学位期间"))
i_decl = first(lambda e: sty(e) == "28" and "原创性声明" in txt(e))
i_ack = first(lambda e: sty(e) == "28" and txt(e).strip() == "致  谢")
final_sect = items[-1]
assert final_sect.tag == w("sectPr")


def set_text(p, new):
    ts = p.findall(f".//{w('t')}")
    ts[0].text = new
    for t in ts[1:]:
        t.text = ""


def strip_bookmarks(e):
    for bmk in e.findall(f".//{w('bookmarkStart')}") + e.findall(f".//{w('bookmarkEnd')}"):
        bmk.getparent().remove(bmk)


# 封面与扉页
cover = items[i_sect1 + 1:i_abs]
for p in cover:
    if p.tag != w("p") and p.tag != w("tbl"):
        continue
    for q in ([p] if p.tag == w("p") else p.findall(f".//{w('p')}")):
        t = txt(q).strip()
        rep = {"Xxx的研究中文题目": C.TITLE_CN, "学生姓名": C.COVER["name"], "2025年6月": C.COVER["date"],
               "2300000000": C.COVER["id"], "某某（副）教授": C.COVER["advisor"], "计算机科学与技术": C.COVER["major"],
               "深圳校区计算机科学与技术学院": C.COVER["school"], "2026年5月": C.COVER["defense"]}
        if t in rep:
            set_text(q, rep[t])
        elif t.startswith("RESEARCH ON"):
            set_text(q, C.TITLE_EN)
        elif t.startswith("毕业论文") and "密级" in t:
            syms = q.findall(f".//{w('sym')}")
            syms[0].set(w("char"), "F052")                 # 勾选“毕业论文”（Wingdings 2 的打勾方框）
            syms[1].set(w("char"), "F0A3")


def heading_keep(e, label):
    strip_bookmarks(e)
    name, inner = bookmark("")
    pPr = e.find(w("pPr"))
    for k, node in enumerate(frag(inner)):
        pPr.addnext(node) if k == 0 else e.find(w("bookmarkStart")).addnext(node)
    TOC.append(("h0", label, name))
    return e


def keywords(e, value, char_style):
    rs = e.findall(w("r"))
    for r_ in rs[2:]:
        e.remove(r_)
    e.extend(frag(f'<w:r><w:rPr><w:rStyle w:val="{char_style}"/></w:rPr><w:t xml:space="preserve">{" " if char_style == "33" else ""}{escape(value)}</w:t></w:r>'))
    return e


front = []
front.append(heading_keep(items[i_abs], "摘  要"))
front += frag(para(runs(C.ABSTRACT_CN), "3"))
front.append(keywords(items[i_kw], C.KEYWORDS_CN, "25"))
front.append(heading_keep(items[i_eabs], "Abstract"))
front += frag(para(runs(C.ABSTRACT_EN), "32"))
front.append(keywords(items[i_ekw], C.KEYWORDS_EN, "33"))

chapters = frag(body_xml())

back = [heading_keep(items[i_concl], "结  论")]
back += frag("".join(para(runs(s), "3") for s in C.CONCLUSION))
back.append(heading_keep(items[i_refs], "参考文献"))
back += frag("".join(para(run(C.REFS[k]), "23") for k in order))
back.append(heading_keep(items[i_ach], "攻读学士学位期间取得创新性成果"))
back += frag("".join(para(runs(s), "3") for s in C.ACHIEVEMENTS))
decl = items[i_decl:i_ack]
for p in decl:
    for t in p.findall(f".//{w('t')}"):
        if t.text and "《" in t.text and "》" in t.text:
            t.text = re.sub(r"《\s*》", f"《{C.TITLE_CN}》", t.text)
back.append(heading_keep(decl[0], "哈尔滨工业大学本科毕业论文（设计）原创性声明和使用权限"))
back += decl[1:]
back.append(heading_keep(items[i_ack], "致  谢"))
back += frag("".join(para(runs(s), "3") for s in C.ACK))

# 目录：摘要、Abstract 在前，正文各级标题，再是结论等（与模板目录的顺序一致）
front_toc = [e for e in TOC if e[1] in ("摘  要", "Abstract")]
body_toc = [e for e in TOC if e[0] != "h0"]
back_toc = [e for e in TOC if e[0] == "h0" and e[1] not in ("摘  要", "Abstract")]
try:
    PAGES = json.load(open(HERE / "toc_pages.json", encoding="utf-8"))
except FileNotFoundError:
    PAGES = {}
chap_no, sec = 0, [0, 0]
entries = []
for lvl, text, bm in front_toc + body_toc + back_toc:
    if lvl == "h1":
        chap_no += 1; sec = [0, 0]; label = f"第{chap_no}章 {text}"
    elif lvl == "h2":
        sec = [sec[0] + 1, 0]; label = f"{chap_no}.{sec[0]} {text}"
    elif lvl == "h3":
        sec[1] += 1; label = f"{chap_no}.{sec[0]}.{sec[1]} {text}"
    else:
        label = text
    entries.append((lvl, label, bm))
toc_x = []
for k, (lvl, label, bm) in enumerate(entries):
    style = {"h0": "10", "h1": "10", "h2": "11", "h3": "7"}[lvl]
    pre = ('<w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText xml:space="preserve">TOC \\o "1-1" \\t "一级节标题2.3,2,二级节标题2.3.1,3" \\h \\u </w:instrText></w:r>'
           '<w:r><w:fldChar w:fldCharType="separate"/></w:r>') if k == 0 else ""
    post = '<w:r><w:fldChar w:fldCharType="end"/></w:r>' if k == len(entries) - 1 else ""
    pg = PAGES.get(label, "")
    pageref = fldrun("PAGEREF " + bm + " \\h", pg)
    body_ = (f'{pre}<w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText xml:space="preserve"> HYPERLINK \\l {bm} </w:instrText></w:r>'
             f'<w:r><w:fldChar w:fldCharType="separate"/></w:r>{run(label)}<w:r><w:tab/></w:r>{pageref}'
             f'<w:r><w:fldChar w:fldCharType="end"/></w:r>{post}')
    toc_x.append(para(body_, style, '<w:tabs><w:tab w:val="right" w:leader="dot" w:pos="8504"/></w:tabs>'))
tocsect = items[i_tocsect]
for r_ in tocsect.findall(w("r")):
    tocsect.remove(r_)

new = cover + front + [items[i_toc]] + frag("".join(toc_x)) + [tocsect] + chapters + back + [final_sect]
for e in list(body):
    body.remove(e)
body.extend(new)

# 清理配对不上的书签（删去示例内容后留下的）
starts = {e.get(w("id")) for e in body.iter(w("bookmarkStart"))}
ends = {e.get(w("id")) for e in body.iter(w("bookmarkEnd"))}
for e in list(body.iter(w("bookmarkStart"))) + list(body.iter(w("bookmarkEnd"))):
    if (e.tag == w("bookmarkStart") and e.get(w("id")) not in ends) or (e.tag == w("bookmarkEnd") and e.get(w("id")) not in starts):
        e.getparent().remove(e)

# ---------- 写出 .docx ----------
OUT = Path(sys.argv[1])
ct = zin.read("[Content_Types].xml").decode()
ct = ct.replace("application/vnd.openxmlformats-officedocument.wordprocessingml.template.main+xml",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml")
if 'Extension="jpg"' not in ct:
    ct = ct.replace("<Default ", '<Default Extension="jpg" ContentType="image/jpeg"/><Default ', 1)
rels = zin.read("word/_rels/document.xml.rels").decode()
rels = rels.replace("</Relationships>", "".join(
    f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/thesis_{fn}"/>'
    for rid, fn in IMAGES) + "</Relationships>")
settings = zin.read("word/settings.xml").decode()
if "updateFields" not in settings:                     # 打开时提示更新域（目录页码、公式编号）
    settings = re.sub(r"(<w:settings[^>]*>)", r'\1<w:updateFields w:val="true"/>', settings, count=1)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for info in zin.infolist():
        name = info.filename
        if name == "[Content_Types].xml":
            z.writestr(name, ct)
        elif name == "word/document.xml":
            z.writestr(name, etree.tostring(doc, xml_declaration=True, encoding="UTF-8", standalone=True))
        elif name == "word/_rels/document.xml.rels":
            z.writestr(name, rels)
        elif name == "word/settings.xml":
            z.writestr(name, settings)
        else:
            z.writestr(info, zin.read(name))
    for rid, fn in IMAGES:
        z.write(HERE / "figures" / fn, f"word/media/thesis_{fn}")
print(f"wrote {OUT}: {sum(1 for e in entries if e[0] in ('h1', 'h2', 'h3'))} headings, {len(IMAGES)} figures, "
      f"{sum(1 for i in C.BODY if i[0] == 'tab')} tables, {sum(1 for i in C.BODY if i[0] == 'eq')} equations, {len(order)} references")

# ---------- 可选：从 PDF 读目录页码 ----------
if "--pages" in sys.argv:
    pdf = sys.argv[sys.argv.index("--pages") + 1]
    n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout).group(1))
    lines = [[re.sub(r"\s+", "", l) for l in subprocess.run(["pdftotext", "-layout", "-f", str(i), "-l", str(i), pdf, "-"],
                                                           capture_output=True, text=True).stdout.splitlines()] for i in range(1, n + 1)]
    key = lambda s: re.sub(r"\s+", "", s)
    roman = lambda k: ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"][k - 1]
    out, p = {}, 0
    found = []
    for lvl, label, bm in entries:
        # 长标题会折行：也接受与标题结尾相同、至少 8 个字的行（原创性声明标题第二行）
        while not any(l == key(label) or (len(l) >= 8 and key(label).endswith(l) and l != key(label)[:len(l)]) for l in lines[p]):
            p += 1
        found.append((label, p))
    abs_page = found[0][1]
    body_start = next(pg for (lab, pg), e in zip(found, entries) if e[0] == "h1")
    for (label, pg), e in zip(found, entries):
        out[label] = roman(pg - abs_page + 1) if pg < body_start else str(pg - body_start + 1)
    json.dump(out, open(HERE / "toc_pages.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("toc pages:", out)
