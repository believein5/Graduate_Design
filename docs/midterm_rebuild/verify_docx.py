# 逐条检查中期报告 .docx 是否满足中期报告模板与《写作指南》的可机检要求（M1–M9、G1–G4）
# 用法：python -I verify_docx.py <.docx> <content.json>
import sys, re, json, zipfile
import docx
from lxml import etree

PATH, CONTENT = sys.argv[1], sys.argv[2]
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
z = zipfile.ZipFile(PATH)
parts = {n: etree.fromstring(z.read(n)) for n in z.namelist() if n.endswith(".xml") or n.endswith(".rels")}
doc, sty = parts["word/document.xml"], parts["word/styles.xml"]
docx.Document(PATH)                                     # 能被 python-docx 打开
content = json.load(open(CONTENT, encoding="utf-8"))
results = []


def check(rid, desc, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {rid:<4} {desc}" + (f"  — {detail}" if detail else ""))


def style(sid):
    return sty.find(f"{W}style[@{W}styleId='{sid}']")


def val(el, path, name="val"):
    e = el.find(path) if el is not None else None
    return None if e is None else e.get(W + name)


def text(p):
    return "".join(t.text or "" for t in p.iter(W + "t"))


paras = list(doc.iter(W + "p"))

# M1 封面
cover_title = [p for p in paras if text(p) in ("哈尔滨工业大学深圳校区", "毕业论文（设计）中期报告")]
ok = len(cover_title) == 2
for p in cover_title:
    r = p.find(f"{W}r/{W}rPr")
    ok &= (val(r, f"{W}rFonts", "eastAsia") == "华文新魏" and val(r, f"{W}sz") == "72"
           and r.find(W + "b") is not None and val(r, f"{W}w") == "90")
labels = {"题目": "36"} | {k: "30" for k in ("姓名", "学号", "学院", "专业", "指导教师", "日期")}
for p in paras:
    if text(p) in labels:
        r = p.find(f"{W}r/{W}rPr")
        ok &= val(r, f"{W}rFonts", "eastAsia") == "黑体" and val(r, f"{W}sz") == labels[text(p)]
syms = [s.get(W + "char") for s in doc.iter(W + "sym")]
check("M1", "封面：华文新魏小初加粗缩放 90%，题目黑体小二，其余标签黑体小三，勾选框", ok and syms == ["F052", "F0A3"],
      f"复选框 {syms}")

# M2 章节结构
h1 = [c["num"] + " " + c["text"] for c in content if c["t"] == "h1" and c["num"]]
want = ["1 课题主要研究内容及进度", "2 已完成的研究工作及结果", "3 后期拟完成的研究工作及进度安排",
        "4 存在的困难及解决方案", "5 论文按时完成的可能性"]
h2 = {c["num"]: c["text"] for c in content if c["t"] == "h2"}
want2 = {"1.1": "课题主要研究内容", "1.2": "进度介绍", "3.1": "后期拟完成的研究工作", "3.2": "进度安排",
         "4.1": "存在的困难", "4.2": "解决方案"}
check("M2", "五章结构与模板固定节标题一致", h1 == want and all(h2.get(k) == v for k, v in want2.items()))

# M3 目录
instr = "".join(t.text for t in doc.iter(W + "instrText"))
n_heads = sum(c["t"] in ("h1", "h2", "h3") for c in content)
toc_entries = [p for p in paras if val(p, f"{W}pPr/{W}pStyle") in ("TOC1", "TOC2", "TOC3")]
ok = 'TOC \\o "1-3"' in instr and len(toc_entries) == n_heads
for lvl, east in (("TOC1", "黑体"), ("TOC2", None), ("TOC3", None)):
    s = style(lvl)
    ok &= val(s, f"{W}rPr/{W}rFonts", "eastAsia") == east
    ok &= val(s, f"{W}pPr/{W}spacing", "line") == "300" and val(s, f"{W}pPr/{W}ind", "hanging") is not None
check("M3", "目录三级、章黑体其余宋体、悬挂缩进、1.25 倍", ok, f"{len(toc_entries)}/{n_heads} 条")

# M4 页码
sects = list(doc.iter(W + "sectPr"))
fmts = [val(s, f"{W}pgNumType", "fmt") for s in sects]
starts = [val(s, f"{W}pgNumType", "start") for s in sects]
check("M4", "封面无页码；目录罗马页码；正文阿拉伯页码重新从 1 开始",
      fmts == [None, "upperRoman", "decimal"] and starts[1:] == ["1", "1"], f"{fmts}")

# M5 各级标题与正文
def spc(s, a):
    return val(s, f"{W}pPr/{W}spacing", a)


spec = {"Heading1": ("36", "100", "80"), "Heading2": ("30", "50", "50"), "Heading3": ("28", "50", "50"),
        "Heading4": ("24", "0", "0")}
ok = True
bad = []
for sid, (sz, b, a) in spec.items():
    s = style(sid)
    good = (val(s, f"{W}rPr/{W}sz") == sz and val(s, f"{W}rPr/{W}rFonts", "eastAsia") == "黑体"
            and spc(s, "line") == "300" and spc(s, "beforeLines") == b and spc(s, "afterLines") == a)
    ok &= good
    if not good:
        bad.append(sid)
normal = style("Normal")
dflt = sty.find(f"{W}docDefaults/{W}rPrDefault/{W}rPr")
ok &= spc(normal, "line") == "300" and spc(normal, "before") == "0" and spc(normal, "after") == "0"
ok &= val(dflt, f"{W}sz") == "24" and val(dflt, f"{W}rFonts", "eastAsia") == "宋体"
check("M5", "章黑体小二 1/0.8 行、节小三 0.5/0.5、条四号 0.5/0.5、款小四 0/0、正文宋体小四，均 1.25 倍", ok, str(bad))

# M6 Times New Roman
ok = val(dflt, f"{W}rFonts", "ascii") == "Times New Roman" and val(dflt, f"{W}rFonts", "hAnsi") == "Times New Roman"
other = set()
for part in ("word/document.xml", "word/styles.xml", "word/header1.xml", "word/footer1.xml", "word/footer2.xml"):
    for rf in parts[part].iter(W + "rFonts"):
        for a in ("ascii", "hAnsi"):
            v = rf.get(W + a)
            if v and v not in ("Times New Roman", "Wingdings 2"):
                other.add(v)
check("M6", "数字、英文 Times New Roman（无其他西文字体）", ok and not other, str(other or ""))

# M7 每章另起一页
check("M7", "章标题段前分页", style("Heading1").find(f"{W}pPr/{W}pageBreakBefore") is not None)

# M8 页眉
hdr = parts["word/header1.xml"]
hs = style("Header")
check("M8", "页眉文字、小五、双线、居中",
      text(hdr.find(f".//{W}p")) == "哈尔滨工业大学深圳校区毕业论文（设计）中期报告"
      and val(hs, f"{W}rPr/{W}sz") == "18" and val(hs, f"{W}pPr/{W}pBdr/{W}bottom") == "thinThickSmallGap"
      and val(hs, f"{W}pPr/{W}jc") == "center"
      and all(s.find(f"{W}headerReference") is not None for s in sects[1:]))

# M9 版面：与模板各节的 twips 值逐一比较（模板节 1：1800/1800/1440/1440；节 2、3：1701/1701/2155/1701）
tw = lambda s, k: int(val(s, f"{W}pgMar", k))
geo = [(int(val(s, f"{W}pgSz", "w")), int(val(s, f"{W}pgSz", "h")),
        *(tw(s, k) for k in ("left", "right", "top", "bottom"))) for s in sects]
check("M9", "A4；封面 31.75/31.75/25.4/25.4，其余 30/30/38/30 mm（与模板各节一致）",
      geo == [(11906, 16838, 1800, 1800, 1440, 1440)] + [(11906, 16838, 1701, 1701, 2155, 1701)] * 2, str(geo))

# G1 图
caps = [p for p in paras if val(p, f"{W}pPr/{W}pStyle") == "FigCaption"]
ok, why = True, []
for i, p in enumerate(caps):
    t = text(p)
    m = re.fullmatch(r"图(\d+)-(\d+)  ([^，。；：、！？,.;:]+?)(\[\d+(?:,\d+)*\])?", t)
    prev = p.getprevious()
    if not m:
        ok, _ = False, why.append(t)
        continue
    if prev is None or prev.find(f".//{W}drawing") is None:
        ok, _ = False, why.append("图题不在图下 " + t)
fc = style("FigCaption")
ok &= val(fc, f"{W}rPr/{W}sz") == "21" and val(fc, f"{W}rPr/{W}rFonts", "eastAsia") is None   # 继承宋体
# 按章编号且连续
nums = [tuple(map(int, re.match(r"图(\d+)-(\d+)", text(p)).groups())) for p in caps]
seq_ok = all(b == (a[0], a[1] + 1) or (b[0] > a[0] and b[1] == 1) for a, b in zip(nums, nums[1:]))
# 提示须出现在图之前
order_ok = True
seen = ""
for c in content:
    if c["t"] == "p":
        seen += c["text"]
    if c["t"] == "cap" and f"图{c['num']}" not in seen:
        order_ok = False
        why.append("提示在图后 " + c["num"])
check("G1", "图题图下、按章连续编号、2 空格、无标点、宋体五号、图前有提示", ok and seq_ok and order_ok,
      f"{len(caps)} 幅图 " + "; ".join(why))

# G2 表
tcaps = [p for p in paras if val(p, f"{W}pPr/{W}pStyle") == "TableCaption"]
ok, why = True, []
for p in tcaps:
    t = text(p)
    m = re.fullmatch(r"表(\d+)-(\d+)  ([^，。；：、！？,.;:]+)", t)
    nxt = p.getnext()
    if not m or nxt is None or nxt.tag != W + "tbl":
        ok, _ = False, why.append(t)
        continue
    b = nxt.find(f"{W}tblPr/{W}tblBorders")
    ok &= (val(b, f"{W}top") == "single" and val(b, f"{W}bottom") == "single" and val(b, f"{W}left") == "nil"
           and val(b, f"{W}right") == "nil" and val(b, f"{W}insideV") == "nil")
    # 表前后空行
    after = text(nxt.getnext())
    ok &= text(p.getprevious()) == "" and (after == "" or after.startswith("注："))
seen = ""
for c in content:
    if c["t"] == "p":
        seen += c["text"]
    if c["t"] == "table" and f"表{c['num']}" not in seen:
        ok, _ = False, why.append("无表前提示 " + c["num"])
ok &= all(val(style(s), f"{W}rPr/{W}sz") == "21" for s in ("TableCaption", "TableText"))
check("G2", "三线表、表题在上、宋体五号、表内五号、表前提示、前后空行", ok, f"{len(tcaps)} 张表 " + "; ".join(why))

# G3 公式
eqs = [p for p in paras if val(p, f"{W}pPr/{W}pStyle") == "Equation"]
ok, why = True, []
for p in eqs:
    tabs = style("Equation").findall(f"{W}pPr/{W}tabs/{W}tab")
    t = text(p)
    if not (re.search(r"\(\d+-\d+\)$", t) and p.find(f".//{M}oMath") is not None
            and [x.get(W + "val") for x in tabs] == ["center", "right"]):
        ok, _ = False, why.append(t[-8:])
# 公式内字母斜体、数字正体，字体为 TNR
for mr in doc.iter(M + "r"):
    s = mr.find(f"{M}t").text
    rp = mr.find(f"{W}rPr")
    it = rp.find(W + "i") is not None
    if val(rp, f"{W}rFonts", "ascii") != "Times New Roman":
        ok, _ = False, why.append("非 TNR " + s)
    if s.isdigit() and it:
        ok, _ = False, why.append("数字斜体 " + s)
eq_nums = [re.search(r"\((\d+-\d+)\)$", text(p))[1] for p in eqs]
seen = "".join(c.get("text", "") for c in content if c["t"] == "p")
missing = [n for n in eq_nums if f"式({n})" not in seen and f"式({n})～" not in seen and
           not any(f"～式({m})" in seen and int(n.split('-')[1]) <= int(m.split('-')[1]) for m in eq_nums)]
check("G3", "公式居中、按章编号、编号右对齐、TNR、数字正体、正文引用", ok and not missing,
      f"{len(eqs)} 个公式 " + "; ".join(why + missing))

# G4 引用
refs = [c for c in content if c["t"] == "ref"]
cited = []
for c in content:
    for m in re.finditer(r"\^\[([\d,\-]+)\]\^", c.get("text", "")):
        for part in m[1].split(","):
            a, *b = map(int, part.split("-"))
            cited += list(range(a, (b[0] if b else a) + 1))
first = list(dict.fromkeys(cited))
heads_cite = [c["text"] for c in content if c["t"].startswith("h") and "^[" in c["text"]]
types_ok = all(re.search(r"\[(J|C|M|EB/OL|DB/OL|J/OL)\]", r["text"]) for r in refs)
sup = [r for r in doc.iter(W + "r") if val(r, f"{W}rPr/{W}vertAlign") == "superscript"
       and re.fullmatch(r"\[[\d,\-]+\]", text(r))]
check("G4", "上标顺序编码、按首次出现排序、全部被引用、标题无引用、GB/T 7714 文献类型标识",
      first == list(range(1, len(refs) + 1)) and not heads_cite and types_ok and len(sup) == sum(
          len(re.findall(r"\^\[", c.get("text", ""))) for c in content),
      f"{len(refs)} 篇，首次引用顺序 {first}")

print(f"\n{sum(results)}/{len(results)} 通过")
sys.exit(0 if all(results) else 1)
