# 将 draft.json 规范化为 content.json：合并行、清理空格、补标点、引用重编号、参考文献重排、重建进度表
import sys, json, re
import pymupdf

HERE = sys.argv[1]
PDF = sys.argv[2]
blocks = json.load(open(f"{HERE}/draft.json"))

CJK = r"⺀-鿿　-〿＀-￯“”‘’…—"
ASCII_TAIL = re.compile(r"[A-Za-z0-9,.;:)\]]$")
ASCII_HEAD = re.compile(r"^[A-Za-z0-9(\[]")


def join_lines(text):
    out = ""
    for ln in text.split("\n"):
        ln = ln.strip()
        if out and ASCII_TAIL.search(out) and ASCII_HEAD.match(ln):
            out += " "
        out += ln
    return out


def clean_spaces(s):
    # 去掉与中文字符/全角标点相邻的空格（Word 中西文自动间距会补足），保留西文词间空格
    s = re.sub(rf"\s+(?=[{CJK}])", "", s)
    s = re.sub(rf"(?<=[{CJK}])\s+", "", s)
    s = re.sub(r"\s*→\s*", " → ", s)
    return re.sub(r" {2,}", " ", s).strip()


# 旧编号 → 新编号（按正文首次出现顺序）
FIRST_SEEN = [1, 2, 3, 4, 5, 6, 7, 8, 19, 9, 10, 11, 20, 12, 13, 14, 15, 18, 16, 17]
NEW = {old: i + 1 for i, old in enumerate(FIRST_SEEN)}


def parse_nums(s):
    nums = []
    for part in re.split(r"[,，、]", s):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            nums += range(int(a), int(b) + 1)
        elif part:
            nums.append(int(part))
    return nums


def compress(nums):
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        if j - i >= 2:
            out.append(f"{nums[i]}-{nums[j]}")
        else:
            out += [str(n) for n in nums[i:j + 1]]
        i = j + 1
    return ",".join(out)


cite_log = []


def renumber(s):
    def rep(m):
        new = compress(NEW[n] for n in parse_nums(m.group(1)))
        cite_log.append(f"{m.group(0)} -> [{new}]")
        return f"^[{new}]^"
    return re.sub(r"〔([^〕]+)〕", rep, s)


def replace_once(s, old, new):
    assert s.count(old) == 1, (old, s.count(old))
    return s.replace(old, new)


# 6 处 PDF 行尾丢失的标点
PUNCT_FIXES = [
    ("实现钻孔质量检测Liu", "实现钻孔质量检测；Liu"),
    ("背景结构依赖较强当PCB", "背景结构依赖较强，当PCB"),
    ("连续时间理论框架Latent", "连续时间理论框架；Latent"),
    ("缺陷面积较小的问题可采用", "缺陷面积较小的问题，可采用"),
    ("统一图像尺寸与归一化检查", "统一图像尺寸与归一化、检查"),
    ("验证集和测试集在数据允许", "验证集和测试集。在数据允许"),
]

CAPTION = re.compile(r"^图\s*(\d+)\s+(.*)$")

content, refs = [], []
pending_fig2_caption = False
for b in blocks:
    t = b["type"]
    if t in ("h1", "h2", "h3"):
        m = re.match(r"^([\d.]+)\s*(.*)$", b["text"])
        num, title = (m.group(1), m.group(2)) if m else ("", b["text"])
        content.append({"t": t, "num": num, "text": clean_spaces(title)})
    elif t == "fig":
        content.append({"t": "fig", "name": b["name"]})
    elif t == "table":
        content.append({"t": "table", "name": b["name"]})
    elif t == "ref":
        refs.append(join_lines(b["text"]))
    else:
        # 题注：紧跟在图之后的段落
        m = CAPTION.match(join_lines(b["text"])) if content and content[-1]["t"] == "fig" else None
        s = clean_spaces(join_lines(b["text"]))
        if m:
            content.append({"t": "cap", "text": f"图{m.group(1)} {clean_spaces(m.group(2))}"})
        elif s.startswith("（e）毛刺") and content[-1]["t"] == "cap":
            content[-1]["text"] += s          # 图 4 题注第二行
        else:
            content.append({"t": "p", "text": s})

body_text = "\n".join(json.dumps(c, ensure_ascii=False) for c in content)
for old, new in PUNCT_FIXES:
    body_text = replace_once(body_text, old, new)
body_text = replace_once(body_text, "引自文献[20]", f"引自文献[{NEW[20]}]")
content = [json.loads(l) for l in body_text.split("\n")]
for c in content:
    if "text" in c:
        c["text"] = renumber(c["text"])
        # 上标引用置于句末标点之前（GB/T 7714）：“。^[1]^” → “^[1]^。”
        c["text"] = re.sub(r"([。；，：])(\^\[[^\]]+\]\^)", r"\2\1", c["text"])


def apply_once(old, new):
    hits = [c for c in content if c["t"] == "p" and old in c["text"]]
    assert len(hits) == 1 and hits[0]["text"].count(old) == 1, (old, len(hits))
    hits[0]["text"] = hits[0]["text"].replace(old, new)


# ---------- 图：《写作指南》§2.13 ----------
# 图题不含标点；引用图在图题右上角标文献号（已为新编号）；子图标注置于图题下方，分号分隔
FIG_CAPTIONS = {
    "fig1": ("PCB电路板图像示例", None),
    "fig2": ("PKU-Market-PCB数据集缺陷样本^[9]^", None),          # 子图题已在各子图下方
    "fig3": ("基于双扩散模型的PCB缺陷图像生成总体技术路线图", None),
    "fig4": ("PCB六类典型缺陷样本^[13]^", "（a）缺孔；（b）鼠咬；（c）开路；（d）短路；（e）毛刺；（f）多余铜箔"),
    "fig5": ("DDPM基本扩散过程示意图^[14]^", None),               # 图出自文献[14]
    "fig6": ("DDIM采样加速示意图^[15]^", None),                   # 图出自文献[15]
}
fig_no, chapter, k = {}, None, 0
for i, c in enumerate(content):
    if c["t"] == "h1":
        chapter, k = c["num"], 0
    elif c["t"] == "fig":
        k += 1
        fig_no[c["name"]] = f"{chapter}-{k}"
        cap = content[i + 1]
        title, sub = FIG_CAPTIONS[c["name"]]
        assert cap["t"] == "cap" and re.sub(r"\^\[[^\]]+\]\^", "", title)[:6] in cap["text"], (title, cap)
        cap.update(num=fig_no[c["name"]], text=title, sub=sub)
# 正文中的图号按章重编：原“图n”为第 n 幅图
for c in content:
    if c["t"] == "p":
        c["text"] = re.sub(r"图(\d+)(?![-\d])", lambda m: "图" + fig_no[f"fig{m.group(1)}"], c["text"])
# 图前须有文字提示（§2.13.2）：补齐原文缺少的三处
apply_once("消融分析与模型优化。", f"消融分析与模型优化，如图{fig_no['fig3']}所示。")
apply_once("相应训练方法。", f"相应训练方法，其基本扩散过程如图{fig_no['fig5']}所示。")
apply_once("相近的生成质量。", f"相近的生成质量，其加速采样过程如图{fig_no['fig6']}所示。")

# 参考文献按新编号重排
ref_by_old = {}
for r in refs:
    m = re.match(r"^\[(\d+)\]\s*(.*)$", r)
    ref_by_old[int(m.group(1))] = m.group(2)
assert sorted(ref_by_old) == list(range(1, 21))
# 按 GB/T 7714-2015 统一著录格式（键为原编号）
REF_FIXES = {
    # 电子文献需（发表日期）[引用日期]. 获取路径；发表日期见 arXiv 页面
    19: ("[EB/OL]. arXiv:1901.08204, 2019.",
         "[EB/OL]. (2019-01-24)[2026-10-19]. https://arxiv.org/abs/1901.08204."),
    # 论文集析出文献：卷号属于论文集名，出版年后接页码
    12: ("Advances in Neural Information Processing Systems. 2020, 33: 6840-6851.",
         "Advances in Neural Information Processing Systems 33. 2020: 6840-6851."),
    17: ("International Conference on Machine Learning. 2023, 202: 41164-41193.",
         "International Conference on Machine Learning. PMLR 202. 2023: 41164-41193."),
}
for old, (a, b) in REF_FIXES.items():
    ref_by_old[old] = replace_once(ref_by_old[old], a, b)
# 著者姓全大写（《写作指南》附录 4；GB/T 28039—2011），名缩写保持不变
AUTHORS = re.compile(r"^((?:[A-Z][A-Za-z-]+(?: [A-Z])+, )*(?:[A-Z][A-Za-z-]+(?: [A-Z])+|et al))\. ")
for old, txt in ref_by_old.items():
    m = AUTHORS.match(txt)
    assert m, txt
    names = [n if n == "et al" else n.split(" ", 1)[0].upper() + " " + n.split(" ", 1)[1]
             for n in m.group(1).split(", ")]
    ref_by_old[old] = ", ".join(names) + txt[m.end(1):]
new_refs = sorted(((NEW[o], txt) for o, txt in ref_by_old.items()))
for n, txt in new_refs:
    content.append({"t": "ref", "n": n, "text": txt})

# 进度安排表：按列坐标从 PDF 第 14 页提取
page = pymupdf.open(PDF)[13]
COLS = [(80, 200), (200, 390), (390, 520)]
cell_lines = {}   # (y, 列号) -> 该列该行的文字
for blk in page.get_text("dict")["blocks"]:
    for l in blk.get("lines", []):
        for s in l["spans"]:
            # 用基线而非 bbox 顶部分组：中西文字体上沿高度不同
            y, x = round(s["origin"][1]), s["bbox"][0]
            col = next((k for k, (a, c) in enumerate(COLS) if a <= x < c), None)
            if col is None or not (165 <= y < 500) or not s["text"].strip():
                continue
            key = next((kk for kk in cell_lines if kk[1] == col and abs(kk[0] - y) < 4), (y, col))
            cell_lines.setdefault(key, []).append((x, s["text"]))
lines = sorted((y, col, "".join(t for _, t in sorted(parts))) for (y, col), parts in cell_lines.items())
# 行起点：第一列为“时间”，或以“20xx 年 x 月—”开头
starts = [y for y, col, txt in lines if col == 0 and re.match(r"^\s*(时间|20\d\d\s*年)", txt)]
rows = []
for i, y0 in enumerate(starts):
    y1 = starts[i + 1] if i + 1 < len(starts) else 10_000
    rows.append([clean_spaces(join_lines("\n".join(txt.strip() for y, col, txt in lines
                                                   if y0 <= y < y1 and col == k)))
                 for k in range(len(COLS))])
# 表内文字说明句末不加标点（《写作指南》§2.12）
rows = [rows[0]] + [[cell.rstrip("。；，") for cell in r] for r in rows[1:]]
ti = next(i for i, c in enumerate(content) if c.get("t") == "table")
table_chapter = next(c["num"] for c in reversed(content[:ti]) if c["t"] == "h1")
content[ti].update(rows=rows, num=f"{table_chapter}-1", title="进度安排")
# 表前须有文字提示（§2.12）
content.insert(ti, {"t": "p", "text": f"本课题的进度安排见表{table_chapter}-1。"})

# 目录缓存页码：取原 PDF 目录页（第 2 页）的页码，打开后需在 Word/WPS 中更新域
toc_pages = {}
for ln in pymupdf.open(PDF)[1].get_text().split("\n"):
    m = re.match(r"^(.*?)\.{4,}\s*(\d+)$", ln.strip())
    if m:
        toc_pages[clean_spaces(re.sub(r"^[\d.]+\s*", "", m.group(1)))] = int(m.group(2))
for c in content:
    if c["t"] in ("h1", "h2", "h3"):
        c["toc_page"] = toc_pages[c["text"]]

json.dump(content, open(f"{HERE}/content.json", "w"), ensure_ascii=False, indent=1)
with open(f"{HERE}/content_review.txt", "w") as f:
    for c in content:
        if c["t"] == "table":
            f.write("[table]\n" + "\n".join(" | ".join(r) for r in c["rows"]) + "\n")
        elif c["t"] == "ref":
            f.write(f"[ref] [{c['n']}] {c['text']}\n")
        elif c["t"] == "fig":
            f.write(f"[fig] {c['name']}\n")
        else:
            f.write(f"[{c['t']}] {(c.get('num', '') + '  ') if c.get('num') else ''}{c['text']}\n")
print("\n".join(cite_log))
print("blocks:", len(content), "refs:", len(new_refs), "table rows:", len(rows))
