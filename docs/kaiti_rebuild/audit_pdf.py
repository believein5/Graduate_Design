# 审查导出的 PDF：逐个文字片段判定其版面角色（封面、页眉、目录、各级标题、正文、引用、图表题、参考文献），
# 核对实际字体风格与字号是否符合模板/《写作指南》，并列出所有落到替代字体（回退字体）上的字符。
# 用法：python audit_pdf.py <pdf>
import sys, re, collections
import pymupdf

PDF = sys.argv[1]
doc = pymupdf.open(PDF)

# PDF 内字体名 → 风格（导出时的映射见 fonts.conf）
STYLE = {"FandolSong": "宋体", "FandolHei": "黑体", "STXinwei": "华文新魏", "TimesNewRomanPSMT": "TNR",
         "TimesNewRomanPS-BoldMT": "TNR", "OpenSymbol": "符号(OpenSymbol)"}
CJK = re.compile(r"[　-〿一-鿿＀-￯—‘’“”…]")
LATIN = re.compile(r"[A-Za-z0-9]")

# 角色 → (中文应为, 西文/数字应为, 字号)；字号 None 表示不检查
EXPECT = {
    "封面大标题": ("华文新魏", None, 34.2),          # 36 磅 × 字符缩放 90% → PDF 读数 34.2
    "封面题目行": ("黑体", "TNR", 18.0),
    "封面栏目名": ("黑体", "TNR", 15.0),
    "封面填写内容": ("宋体", "TNR", 15.0),
    "封面勾选行": ("宋体", "TNR", 12.0),
    "页眉": ("宋体", "TNR", 9.0),
    "页码": (None, "TNR", 9.0),
    "目录标题": ("黑体", "TNR", 18.0),
    "目录-章": ("黑体", "TNR", 12.0),
    "目录-节/条": ("宋体", "TNR", 12.0),
    "章标题": ("黑体", "TNR", 18.0),
    "节标题": ("黑体", "TNR", 15.0),
    "条标题": ("黑体", "TNR", 14.0),
    "正文": ("宋体", "TNR", 12.0),
    "上标引用": (None, "TNR", None),
    "图表题/表内文字": ("宋体", "TNR", 10.5),
    "参考文献": ("宋体", "TNR", 12.0),
}

ref_page = next(i for i, p in enumerate(doc) if i > 2 and re.search(r"^\s*参考文献\s*$", p.get_text(), re.M))
stats = collections.defaultdict(lambda: {"n": 0, "cjk": collections.Counter(), "latin": collections.Counter(),
                                         "size": collections.Counter()})
problems, fallback = [], collections.Counter()


def role(pno, span, line_text, y):
    size = round(span["size"], 1)
    if pno == 0:
        if size > 30: return "封面大标题"
        if size == 18.0: return "封面题目行"
        if size == 15.0: return "封面栏目名" if span["bbox"][0] < 190 else "封面填写内容"
        return "封面勾选行"
    if y < 95: return "页眉"
    if y > 755: return "页码"
    if size < 9.5: return "上标引用"
    if pno == 1:
        if line_text.replace(" ", "") == "目录": return "目录标题"
        return "目录-章" if re.match(r"^(\d+\s|参考文献)", line_text) else "目录-节/条"
    if size >= 17.5: return "章标题"
    if size == 15.0: return "节标题"
    if size == 14.0: return "条标题"
    if 10 <= size <= 11: return "图表题/表内文字"
    return "参考文献" if pno >= ref_page else "正文"


for pno, page in enumerate(doc):
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            line_text = "".join(s["text"] for s in l["spans"]).strip()
            for s in l["spans"]:
                t = s["text"]
                if not t.strip():
                    continue
                font = s["font"].split("+")[-1]
                style = STYLE.get(font) or STYLE.get(font.removesuffix("-Regular"), f"未预期字体:{font}")
                r = role(pno, s, line_text, s["origin"][1])
                st = stats[r]
                st["n"] += 1
                st["size"][round(s["size"], 1)] += 1
                want_cjk, want_latin, want_size = EXPECT[r]
                for ch in t:
                    if CJK.match(ch):
                        st["cjk"][style] += 1
                        if want_cjk and style != want_cjk:
                            problems.append((pno + 1, r, f"“{ch}” 用 {style}，应为 {want_cjk}", line_text[:24]))
                    elif LATIN.match(ch):
                        st["latin"][style] += 1
                        if want_latin and style != want_latin:
                            problems.append((pno + 1, r, f"“{ch}” 用 {style}，应为 {want_latin}", line_text[:24]))
                    if style not in ("宋体", "黑体", "华文新魏", "TNR"):
                        fallback[(ch, f"U+{ord(ch):04X}", style, pno + 1)] += 1
                if want_size and r != "上标引用" and abs(s["size"] - want_size) > 0.15:
                    problems.append((pno + 1, r, f"字号 {s['size']:.1f}，应为 {want_size}", line_text[:24]))

print(f"{'角色':<14}{'片段':>5}  中文字体            西文/数字字体          字号")
for r in EXPECT:
    st = stats.get(r)
    if not st:
        print(f"{r:<14}{0:>5}  （PDF 中未出现）")
        continue
    print(f"{r:<14}{st['n']:>5}  {dict(st['cjk'])!s:<20}{dict(st['latin'])!s:<22}{dict(st['size'])}")

print(f"\n不符合要求的片段：{len(problems)}")
seen = set()
for p in problems:
    key = (p[1], p[2].split('，')[-1], p[2][:1] == '字')
    if key in seen:
        continue
    seen.add(key)
    print(f"  第{p[0]}页 [{p[1]}] {p[2]} | {p[3]}")
print(f"\n回退/替代字体绘制的字符：{sum(fallback.values())}")
for (ch, cp, style, pg), n in sorted(fallback.items(), key=lambda x: x[0][3]):
    print(f"  第{pg}页 {ch!r} {cp} → {style} ×{n}")
