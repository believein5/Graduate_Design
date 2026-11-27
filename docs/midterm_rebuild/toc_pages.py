# 从渲染出的 PDF 读出各级标题所在页码（正文阿拉伯页码），写入 toc_pages.json，供 build_docx.py 作为目录域的缓存值
# 用法：python -I toc_pages.py <content.json> <PDF> <输出 toc_pages.json>
import sys, json, re, subprocess

content = json.load(open(sys.argv[1], encoding="utf-8"))
pdf = sys.argv[2]
n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout).group(1))
pages = [subprocess.run(["pdftotext", "-layout", "-f", str(i), "-l", str(i), pdf, "-"],
                        capture_output=True, text=True).stdout for i in range(1, n + 1)]
# pdftotext 会在中西文之间插入空格，比较时去掉全部空白；按行整行匹配
lines = [[re.sub(r"\s+", "", l) for l in pg.splitlines()] for pg in pages]
heads = [c for c in content if c["t"] in ("h1", "h2", "h3")]
key = lambda c: re.sub(r"\s+", "", c["num"] + c["text"])
# 正文第 1 页：第一个章标题出现的页（跳过封面和目录页）
body_start = next(i for i in range(2, n) if key(heads[0]) in lines[i])
out, p = {}, body_start
for c in heads:
    while key(c) not in lines[p]:
        p += 1
    out[f"{c['num']}  {c['text']}" if c["num"] else c["text"]] = p - body_start + 1
json.dump(out, open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False))
