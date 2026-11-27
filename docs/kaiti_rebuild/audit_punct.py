# 标点审查：中文段落中的半角标点、英文参考文献中的全角标点、重复标点、括号/引号不成对、标点旁多余空格
# 用法：python audit_punct.py <docx>
import sys, re
import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

d = docx.Document(sys.argv[1])
items = []                                   # (样式, 文字)
for el in d.element.body:
    tag = el.tag.split("}")[1]
    if tag == "p":
        p = Paragraph(el, d)
        items.append((p.style.style_id, p.text))
    elif tag == "tbl":
        for row in Table(el, d).rows:
            for c in row.cells:
                for cp in c.paragraphs:
                    items.append(("表格", cp.text))

CJK = r"一-鿿"
RULES = [
    # 中文语境中的半角标点（数字内的小数点、范围连字符、mAP@50-95 等除外）
    ("中文旁的半角逗号/分号/冒号/问号/叹号", rf"[{CJK}][,;:?!]|[,;:?!][{CJK}]"),
    ("中文旁的半角句点", rf"[{CJK}]\.(?!\d)|(?<!\d)\.[{CJK}]"),
    ("中文旁的半角括号", rf"[{CJK}]\(|\)[{CJK}]|\([{CJK}]|[{CJK}]\)"),
    ("重复标点", r"([，。；：、！？])\1|[，；：、][。！？]|[。！？][，；：、]"),
    ("标点前后多余空格", r"\s[，。；：、！？）”]|[，。；：、！？（“]\s"),
    ("英文省略号", r"\.\.\.|。。。"),
]
found = 0
for sid, t in items:
    if not t.strip():
        continue
    is_ref = sid == "RefEntry"
    rules = [("参考文献中的全角标点", r"[，。；：（）【】、“”]")] if is_ref else RULES
    for name, pat in rules:
        for m in re.finditer(pat, t):
            if sid.startswith("TOC") and name == "标点前后多余空格":
                continue                      # 目录条目中的制表符
            found += 1
            print(f"[{name}] ({sid}) …{t[max(0, m.start() - 10):m.end() + 10]}…")
    if not is_ref:
        for a, b in ("（）", "“”", "《》", "【】", "()"):
            if t.count(a) != t.count(b):
                found += 1
                print(f"[括号/引号不成对 {a}{b}] ({sid}) {t[:40]}…")
print(f"\n共 {found} 处")
