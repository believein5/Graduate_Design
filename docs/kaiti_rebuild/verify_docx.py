# 逐条检查输出 .docx 是否满足模板红字要求（可机检部分）
import sys, re, zipfile
import docx
from lxml import etree

PATH = sys.argv[1]
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
z = zipfile.ZipFile(PATH)
doc_xml = etree.fromstring(z.read("word/document.xml"))
sty_xml = etree.fromstring(z.read("word/styles.xml"))
d = docx.Document(PATH)   # 能被 python-docx 打开
results = []


def check(rid, desc, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {rid:<4} {desc}" + (f"  — {detail}" if detail else ""))


def style(sid):
    return sty_xml.find(f"{W}style[@{W}styleId='{sid}']")


def attr(el, path, name):
    e = el.find(path) if el is not None else None
    return None if e is None else e.get(W + name)


defaults = sty_xml.find(f"{W}docDefaults/{W}rPrDefault/{W}rPr")
east_default = attr(defaults, f"{W}rFonts", "eastAsia")
ascii_default = attr(defaults, f"{W}rFonts", "ascii")


def spec(sid, east, size, before_lines, after_lines):
    s = style(sid)
    sp = s.find(f"{W}pPr/{W}spacing")
    e = attr(s, f"{W}rPr/{W}rFonts", "eastAsia") or east_default
    sz = attr(s, f"{W}rPr/{W}sz", "val") or attr(defaults, f"{W}sz", "val")
    got = (e, int(sz) / 2, sp.get(W + "line"), sp.get(W + "lineRule"),
           int(sp.get(W + "beforeLines", 0)) / 100, int(sp.get(W + "afterLines", 0)) / 100)
    want = (east, size, "300", "auto", before_lines, after_lines)
    return got == want, f"{got}"


# R5 A4
secs = d.sections
check("R5", "A4 页面", all(abs(s.page_width.cm - 21.0) < 0.05 and abs(s.page_height.cm - 29.7) < 0.05 for s in secs))

# R6–R10 字体字号行距
check("R6", "章标题 黑体小二 1.25 倍 段前1行 段后0.8行", *spec("Heading1", "黑体", 18, 1, 0.8))
check("R7", "节标题 黑体小三 1.25 倍 段前/后0.5行", *spec("Heading2", "黑体", 15, 0.5, 0.5))
check("R8", "条标题 黑体四号 1.25 倍 段前/后0.5行", *spec("Heading3", "黑体", 14, 0.5, 0.5))
check("R9", "款项标题 黑体小四 1.25 倍 段前/后0行", *spec("Heading4", "黑体", 12, 0, 0))
n_sp = style("Normal").find(f"{W}pPr/{W}spacing")
ok = (east_default == "宋体" and attr(defaults, f"{W}sz", "val") == "24" and n_sp.get(W + "line") == "300"
      and n_sp.get(W + "before") == "0" and n_sp.get(W + "after") == "0")
check("R10", "正文 宋体小四 1.25 倍 段前/后0", ok)

# 正文段落不得有覆盖字体/行距的直接格式
body_paras = [p for p in doc_xml.iter(W + "p")
              if (p.find(f"{W}pPr/{W}pStyle") is not None and p.find(f"{W}pPr/{W}pStyle").get(W + "val") == "BodyIndent")]
direct = [p for p in body_paras if p.find(f"{W}pPr/{W}spacing") is not None
          or any(r.find(f"{W}rPr/{W}sz") is not None for r in p.iter(W + "r"))]
check("R10", f"{len(body_paras)} 个正文段落均无直接覆盖字号/行距", not direct, f"{len(direct)} 个违规")

# R11 西文/数字字体全部为 Times New Roman
ascii_fonts = {e.get(W + "ascii") for e in doc_xml.iter(W + "rFonts") if e.get(W + "ascii")}
ascii_fonts |= {e.get(W + "ascii") for e in sty_xml.iter(W + "rFonts") if e.get(W + "ascii")}
ascii_fonts.discard("Wingdings 2")   # 封面复选框符号
check("R11", "西文/数字字体仅 Times New Roman", ascii_default == "Times New Roman" and ascii_fonts <= {"Times New Roman"},
      f"{ascii_fonts | {ascii_default}}")

# R13 每章另起一页
h1 = [p for p in d.paragraphs if p.style.style_id == "Heading1"]
check("R13", "章标题样式含“段前分页”", style("Heading1").find(f"{W}pPr/{W}pageBreakBefore") is not None,
      f"{len(h1)} 个章标题（含参考文献）")

# R1 目录含三级标题
instr = "".join(t.text for t in doc_xml.iter(W + "instrText"))
heads = [p for p in d.paragraphs if p.style.style_id in ("Heading1", "Heading2", "Heading3")]
toc_entries = [p for p in d.paragraphs if p.style.style_id in ("TOC1", "TOC2", "TOC3")]
check("R1", "目录域 \\o \"1-3\" 且条目数 = 章节条标题数", '\\o "1-3"' in instr and len(toc_entries) == len(heads),
      f"{len(toc_entries)} 条 / {len(heads)} 个标题")

# R2 目录罗马页码，正文阿拉伯页码重新编码
pg = [s._sectPr.find(W + "pgNumType") for s in secs]
fmt = [(None if p is None else (p.get(W + "fmt"), p.get(W + "start"))) for p in pg]
check("R2", "节：封面 / 目录(罗马, 从1) / 正文(阿拉伯, 从1)",
      len(secs) == 3 and fmt[1] == ("upperRoman", "1") and fmt[2] == ("decimal", "1"), f"{fmt}")

# R3 目录字体
t1 = attr(style("TOC1"), f"{W}rPr/{W}rFonts", "eastAsia")
t2 = attr(style("TOC2"), f"{W}rPr/{W}rFonts", "eastAsia") or east_default
t3 = attr(style("TOC3"), f"{W}rPr/{W}rFonts", "eastAsia") or east_default
check("R3", "目录：章黑体小四，其余宋体小四", (t1, t2, t3) == ("黑体", "宋体", "宋体"), f"{(t1, t2, t3)}")

# R4 目录悬挂缩进
hang = [attr(style(f"TOC{i}"), f"{W}pPr/{W}ind", "hanging") for i in (1, 2, 3)]
check("R4", "目录换行悬挂缩进至标题首字", all(h and int(h) > 0 for h in hang), f"hanging={hang}")

# R5 字数
text = "".join(p.text for p in d.paragraphs)
cjk = len(re.findall(r"[一-鿿]", text))
check("R5", "字数 ≥ 3000", cjk >= 3000, f"{cjk} 个汉字")

# R15 引用按出现顺序
cites = []
for r in doc_xml.iter(W + "r"):
    va = r.find(f"{W}rPr/{W}vertAlign")
    if va is not None and va.get(W + "val") == "superscript":
        cites.append("".join(t.text for t in r.iter(W + "t")))
seen = []
for c in cites:
    for part in c.strip("[]").split(","):
        a, _, b = part.partition("-")
        for n in range(int(a), int(b or a) + 1):
            if n not in seen:
                seen.append(n)
N_REFS = sum(1 for p in d.paragraphs if p.style.style_id == "RefEntry")
check("R15", f"正文引用首次出现顺序为 1…{N_REFS}", seen == list(range(1, N_REFS + 1)), f"{len(cites)} 处上标引用")
refs = [p.text for p in d.paragraphs if p.style.style_id == "RefEntry"]
nums = [int(re.match(r"\[(\d+)\]", r).group(1)) for r in refs]
check("R15", f"参考文献列表编号连续 1…{N_REFS}", nums == list(range(1, N_REFS + 1)))

# R14 数量 / 外文 / 近两年
foreign = sum(1 for r in refs if not re.search(r"[一-鿿]", r))
# 出版年：去掉电子文献的 [引用日期] 与 URL，避免把访问日期当作出版年
years = [max(int(y) for y in re.findall(r"\b(19\d\d|20\d\d)\b",
                                        re.sub(r"\[\d{4}-\d\d-\d\d\]|https?://\S+", "", r))) for r in refs]
check("R14", "参考文献 ≥15 篇，外文 ≥3 篇", len(refs) >= 15 and foreign >= 3, f"{len(refs)} 篇，外文 {foreign}")
check("R14", "含近两年（2025–2026）文献", max(years) >= 2025, f"最新 {max(years)} 年")

# R16 按文献类型逐条匹配 GB/T 7714-2015 著录格式
YEAR = r"(19|20)\d\d"
PATTERNS = {
    "J": rf"\[J\]\. [^\[\]]+, {YEAR}, \d+(\(\d+\))?: [\w-]+\.$",          # 刊名, 年, 卷(期): 页.
    "C": rf"\[C\]//[^\[\]]+\. {YEAR}(: [\w-]+)?\.$",                       # //论文集名. 年: 页.
    "M": rf"\[M\]\. [^:]+: [^,]+, {YEAR}\.$",                               # 出版地: 出版者, 年.
    "EB/OL": r"\[EB/OL\]\. \(\d{4}-\d\d-\d\d\)\[\d{4}-\d\d-\d\d\]\. https?://\S+\.$",
}
bad = []
for r in refs:
    m = re.match(r"^\[(\d+)\]\t[^.\[\]]+(\.[^.\[\]]+)*\. [^\[\]]+\[(J|C|M|EB/OL)\]", r)
    if not m or not re.search(PATTERNS[m.group(3)], r):
        bad.append(r.split("\t")[0])
check("R16", "参考文献逐条符合 GB/T 7714 著录格式（按 J/C/M/EB/OL 分类）", not bad, f"不符：{bad}")

# 引用标注位置：上标须在句末标点之前
after_punct = []
for p in doc_xml.iter(W + "p"):
    prev = ""
    for r in p.iter(W + "r"):
        t = "".join(x.text or "" for x in r.iter(W + "t"))
        va = r.find(f"{W}rPr/{W}vertAlign")
        last = prev.rstrip("‍")[-1:]            # 跳过引用前的不换行字符 U+200D
        if va is not None and va.get(W + "val") == "superscript" and last and last in "。；，：":
            after_punct.append(prev[-6:] + t)
        prev += t
check("引用", "上标引用均位于标点之前", not after_punct, f"{len(after_punct)} 处在标点后：{after_punct[:3]}")

# ---------- 《写作指南（理工类）》 ----------
SZ = lambda sid: attr(style(sid), f"{W}rPr/{W}sz", "val")
bdr = style("Header").find(f"{W}pPr/{W}pBdr/{W}bottom")
check("G1", "页眉下方上细下粗双线（同模板 thinThickSmallGap 3 磅）",
      bdr is not None and bdr.get(W + "val") == "thinThickSmallGap" and bdr.get(W + "sz") == "24")

# 按文档顺序遍历顶层段落/表格
order, chapter = [], None
for el in doc_xml.find(W + "body"):
    if el.tag == W + "p":
        ps = el.find(f"{W}pPr/{W}pStyle")
        sid = ps.get(W + "val") if ps is not None else "Normal"
        txt = "".join(t.text or "" for t in el.iter(W + "t"))
        # 按 <w:br/> 切分为行（图题首行为“图号  标题”，次行为子图标注）
        lines = []
        cur = ""
        for r in el.iter(W + "r"):
            if r.find(W + "br") is not None:
                lines.append(cur); cur = ""
            cur += "".join(t.text or "" for t in r.iter(W + "t"))
        lines.append(cur)
        order.append(("p", sid, txt, lines, el))
    elif el.tag == W + "tbl":
        order.append(("tbl", None, "", [], el))

caps = [(i, o) for i, o in enumerate(order) if o[1] == "FigCaption"]
bad_fmt, bad_punct, bad_chapter, no_prompt = [], [], [], []
chap = None
seen_text = ""
for i, o in enumerate(order):
    if o[1] == "Heading1":
        chap = re.match(r"^(\d*)", o[2]).group(1)
    if o[1] == "FigCaption":
        title_line = o[3][0]
        m = re.match(r"^图(\d+)-(\d+)  (\S.*)$", title_line)
        if not m:
            bad_fmt.append(title_line[:12]); continue
        if m.group(1) != chap:
            bad_chapter.append(title_line[:8])
        title = re.sub(r"\[[\d,-]+\]$", "", m.group(3))
        if re.search(r"[，。；：、！？（）()]", title):
            bad_punct.append(title_line[:14])
        if f"图{m.group(1)}-{m.group(2)}" not in seen_text:
            no_prompt.append(f"图{m.group(1)}-{m.group(2)}")
    if o[1] == "BodyIndent":
        seen_text += o[2]
check("G2", f"{len(caps)} 个图题均为“图章-序  标题”且章号正确", not bad_fmt and not bad_chapter,
      f"格式不符 {bad_fmt}，章号不符 {bad_chapter}")
# 图题格式不符时其余图题检查无从进行，一并判为不通过，避免空集通过
check("G3", "图题宋体五号、图题无标点", SZ("FigCaption") == "21" and not bad_punct and not bad_fmt,
      f"含标点 {bad_punct}")
check("G5", "每幅图之前正文有“图x-y”提示", bool(caps) and not no_prompt and not bad_fmt, f"缺提示 {no_prompt}")
cap_cites = [o[3][0] for _, o in caps if re.search(r"\[[\d,-]+\]$", o[3][0])]
cap_sup = all(any(r.find(f"{W}rPr/{W}vertAlign") is not None and
                  "".join(t.text for t in r.iter(W + "t")).startswith("[") for r in o[4].iter(W + "r"))
              for _, o in caps if re.search(r"\[[\d,-]+\]$", o[3][0]))
check("G4", "引用图的文献号以上标置于图题右上角", cap_sup and len(cap_cites) >= 4, f"{len(cap_cites)} 幅引用图")

# 表
tcap = [(i, o) for i, o in enumerate(order) if o[1] == "TableCaption"]
ok_t = True
detail = []
for i, o in tcap:
    m = re.match(r"^表(\d+)-(\d+)  \S", o[2])
    nxt = order[i + 1][0] if i + 1 < len(order) else None
    before_blank = order[i - 1][1] == "Normal" and not order[i - 1][2]
    after_blank = i + 2 < len(order) and order[i + 2][1] == "Normal" and not order[i + 2][2]
    prompt = any(o2[1] == "BodyIndent" and f"表{m.group(1)}-{m.group(2)}" in o2[2] for o2 in order[:i]) if m else False
    cells = [p for p in order[i + 1][4].iter(W + "p")] if nxt == "tbl" else []
    end_punct = [("".join(t.text for t in p.iter(W + "t")))[-6:] for p in cells
                 if "".join(t.text for t in p.iter(W + "t")).endswith(("。", "；", "，"))]
    detail = [bool(m), nxt == "tbl", before_blank, after_blank, prompt, end_punct]
    ok_t &= bool(m) and nxt == "tbl" and before_blank and after_blank and prompt and not end_punct
check("G6", "表题“表章-序  标题”在表上方、表前有提示、表前后空行、表内句末无标点", bool(tcap) and ok_t,
      f"{detail}" if tcap else "无表题")
check("G6", "表题与表内文字宋体五号", SZ("TableCaption") == "21" and SZ("TableText") == "21")

# 著者姓全大写
lower = []
for r in refs:
    authors = r.split("\t", 1)[1].split(". ", 1)[0]
    for n in authors.split(", "):
        if n != "et al" and n.split(" ")[0] != n.split(" ")[0].upper():
            lower.append(n)
check("G8", "参考文献著者姓全大写", not lower, f"{lower[:4]}")

# ---------- 指导教师意见（2026-10-26） ----------
jc = style("RefEntry").find(f"{W}pPr/{W}jc")
check("T1", "参考文献两端对齐", jc is not None and jc.get(W + "val") == "both")

# 正文段落文字，上标引用处记为 \x01
body_texts = []
for p in doc_xml.iter(W + "p"):
    ps = p.find(f"{W}pPr/{W}pStyle")
    if ps is None or ps.get(W + "val") != "BodyIndent":
        continue
    s = ""
    for r in p.iter(W + "r"):
        va = r.find(f"{W}rPr/{W}vertAlign")
        t = "".join(x.text or "" for x in r.iter(W + "t"))
        s += ("\x01" + t + "\x02") if va is not None and va.get(W + "val") == "superscript" else t
    body_texts.append(s)


def clause_end(s, i, marks="；。"):
    ends = [s.find(m, i) for m in marks]
    ends = [e for e in ends if e >= 0]
    return min(ends) if ends else len(s)


AUTHOR = re.compile(r"(?<![A-Za-z])[A-Z][a-z]+(?:和[A-Z][a-z]+|等)")
METHODS = ["RePaint", "CoPaint", "Score-Based Generative Modeling through Stochastic Differential Equations",
           "Latent Diffusion Models", "相关专著"]
CLAIM = re.compile(r"(?<!该)(研究表明|研究指出|研究均说明|研究结果表明|实验表明|研究也表明|研究同样指出)")
uncited, warn = [], []
for s in body_texts:
    for m in AUTHOR.finditer(s):                    # 作者：其所在分句内须有引用
        if m.group(0) in ("Precision和Recall",):    # 指标名，非作者
            continue
        if "\x01" not in s[m.start():clause_end(s, m.start())]:
            uncited.append(m.group(0))
    for name in METHODS:                            # 方法/文献名：段内首次出现所在句须有引用
        i = s.find(name)
        if i >= 0 and "\x01" not in s[i:clause_end(s, i, "。")]:
            uncited.append(name[:16])
    for m in CLAIM.finditer(s):                     # “研究表明/指出”：同句或段内此前须有引用
        if "\x01" in s[:clause_end(s, m.start(), "。")]:
            continue
        if "已有" in s[max(0, m.start() - 12):m.start()]:
            if "\x01" not in s:                     # “已有研究…”泛指：仅当整段无引用时提示
                warn.append(s[max(0, m.start() - 8):m.end()])
            continue
        uncited.append(s[max(0, m.start() - 8):m.end()])
check("T2", "被点名的研究均在所在分句内标注引用", not uncited, f"未标注：{uncited}")
for w in warn:
    print(f"WARN  T2   来源不明、无法自动补引：…{w}")

# T3：点名的作者/方法所引编号，须指向第一作者相符的文献（防止重编号后张冠李戴）
first_author = {int(re.match(r"\[(\d+)\]", r).group(1)): r.split("\t", 1)[1].split(" ", 1)[0] for r in refs}
METHOD_AUTHOR = {"RePaint": "LUGMAYR", "CoPaint": "ZHANG", "Latent Diffusion Models": "ROMBACH",
                 "相关专著": "SANSEVIERO", "DEFectDiffuser": "LUO", "Song Y等": "SONG", "LDM": "ROMBACH",
                 "基于SDE的": "SONG"}
mismatch, checked = [], 0
for s in body_texts:
    # 仅取“X等”“X和Y”中紧邻的作者姓；论文题名中的英文单词（如 …Equations和Latent）不算
    targets = [(m.start(), m.group(0).split("和")[0].rstrip("等").upper()) for m in AUTHOR.finditer(s)
               if m.group(0) != "Precision和Recall" and not re.search(r"[A-Za-z] $", s[max(0, m.start() - 2):m.start()])]
    targets += [(m.start(), a) for name, a in METHOD_AUTHOR.items() for m in re.finditer(re.escape(name), s)]
    for i, surname in targets:
        seg = s[i:clause_end(s, i)]
        cm = re.search(r"\x01\[([^\]]+)\]\x02", seg)
        if not cm:
            continue                                # 段内后续提及（T2 已保证首次提及有引用）
        nums = [n for part in cm.group(1).split(",") for n in
                range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1)]
        checked += 1
        if surname not in {first_author.get(n) for n in nums}:
            mismatch.append(f"{surname}→[{cm.group(1)}]")
check("T3", f"点名研究的引用编号指向第一作者相符的文献（{checked} 处）", checked > 0 and not mismatch, f"不符：{mismatch}")

# T4：正文与图题不以英文论文全称作主语（指导意见：用“作者等”或方法简称）
TITLES = [re.sub(r"\[[A-Z/]+\].*$", "", r.split("\t", 1)[1].split(". ", 1)[1]) for r in refs]
long_titles = [t for t in TITLES if len(t.split()) >= 5]
# 大小写不敏感：正文常写 “through”，参考文献题名为 “Through”
hits = [t[:40] for t in long_titles for s in body_texts if t.lower() in s.lower()]
hits += [t[:40] for t in long_titles for p in d.paragraphs if p.style.style_id == "FigCaption" and t.lower() in p.text.lower()]
check("T4", "正文不出现 5 词以上的英文论文全称", not hits, f"出现：{hits}")

print(f"\n{sum(results)}/{len(results)} 项通过")
