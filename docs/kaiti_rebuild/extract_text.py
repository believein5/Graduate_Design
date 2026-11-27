# 从 PDF 提取正文为结构化块（标题/段落/图/表/参考文献），输出 draft.json 与行尾衔接清单 junctions.txt
import sys, json, re
import pymupdf

PDF, OUT = sys.argv[1], sys.argv[2]
doc = pymupdf.open(PDF)
FIRST_BODY_PAGE = 3          # PDF 第 3 页起为正文
TABLE_PAGE, TABLE_Y = 14, (165, 500)   # 4.1 进度安排表所在区域，单独重建
FIG_XREFS = {175: "fig1", 184: "fig2", 188: "fig2", 480: "fig3", 504: "fig4", 545: "fig5", 551: "fig6"}

def visual_lines(page):
    rows = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            spans = [s for s in l["spans"] if s["text"]]
            if not any(s["text"].strip() for s in spans):
                continue
            rows.append({"y": l["bbox"][1], "x0": l["bbox"][0], "x1": l["bbox"][2], "spans": spans})
    rows.sort(key=lambda r: (round(r["y"]), r["x0"]))
    merged = []
    for r in rows:
        if merged and abs(merged[-1]["y"] - r["y"]) < 4:
            m = merged[-1]
            m["spans"] += r["spans"]; m["x1"] = max(m["x1"], r["x1"]); m["x0"] = min(m["x0"], r["x0"])
        else:
            merged.append(dict(r))
    for m in merged:
        m["spans"].sort(key=lambda s: s["bbox"][0])
        m["size"] = max(round(s["size"], 1) for s in m["spans"] if s["text"].strip())
        m["bold"] = any("Bold" in s["font"] for s in m["spans"] if s["text"].strip())
        # 小字号（7pt）为上标引用
        m["text"] = "".join(s["text"] for s in m["spans"])
    return merged

items = []
for pno in range(FIRST_BODY_PAGE - 1, len(doc)):
    page = doc[pno]
    for info in page.get_image_info(xrefs=True):
        name = FIG_XREFS.get(info["xref"])
        if name and not any(i.get("fig") == name for i in items):
            items.append({"page": pno + 1, "y": info["bbox"][1], "fig": name})
    if pno + 1 == TABLE_PAGE:
        items.append({"page": pno + 1, "y": TABLE_Y[0], "table": "schedule"})
    for ln in visual_lines(page):
        if ln["y"] < 95 or ln["y"] > 750:
            continue  # 页眉页脚
        if pno + 1 == TABLE_PAGE and TABLE_Y[0] <= ln["y"] < TABLE_Y[1]:
            continue
        if ln["size"] == 10.5:
            continue  # 图 2 子图标题，单独重建
        ln["page"] = pno + 1
        items.append(ln)
items.sort(key=lambda i: (i["page"], i["y"]))

def kind(ln):
    t = ln["text"].strip()
    if ln["size"] >= 17: return "h1"
    if ln["size"] >= 14.5: return "h2"
    if ln["size"] >= 13.5: return "h3"
    if ln["bold"] and re.match(r"^\d+\.\d+\s", t): return "h2"
    if re.match(r"^\[\d+\]", t): return "ref"
    return "line"

blocks, junctions = [], []
prev = None
for it in items:
    if "fig" in it:
        blocks.append({"type": "fig", "name": it["fig"], "page": it["page"]}); prev = None; continue
    if "table" in it:
        blocks.append({"type": "table", "name": it["table"], "page": it["page"]}); prev = None; continue
    k = kind(it)
    t = it["text"]
    if k in ("h1", "h2", "h3"):
        blocks.append({"type": k, "text": t.strip(), "page": it["page"]}); prev = None; continue
    if k == "ref":
        blocks.append({"type": "ref", "text": t.strip(), "page": it["page"], "lines": 1}); prev = it; continue
    cont = False
    if prev is not None and blocks and blocks[-1]["type"] in ("p", "ref"):
        same_page = prev["page"] == it["page"]
        gap = it["y"] - prev["y"] if same_page else 0
        if blocks[-1]["type"] == "ref":
            cont = True
        elif it["x0"] < 100 and (not same_page or gap < 22.5):
            cont = True
    if cont:
        junctions.append(f"p{it['page']:>2}: …{blocks[-1]['text'][-12:]} ‖ {t.strip()[:12]}…")
        blocks[-1]["text"] += "\n" + t.strip()
        blocks[-1]["lines"] += 1
    else:
        blocks.append({"type": "p", "text": t.strip(), "page": it["page"], "x0": round(it["x0"], 1), "lines": 1})
    prev = it

json.dump(blocks, open(f"{OUT}/draft.json", "w"), ensure_ascii=False, indent=1)
open(f"{OUT}/junctions.txt", "w").write("\n".join(junctions))
for b in blocks:
    if b["type"] == "p":
        print(f"[p p{b['page']} x={b['x0']} n={b['lines']}] {b['text'][:50].replace(chr(10), '⏎')} … {b['text'][-25:].replace(chr(10), '⏎')}")
    elif b["type"] in ("fig", "table"):
        print(f"[{b['type']} {b['name']} p{b['page']}]")
    else:
        print(f"[{b['type']} p{b['page']}] {b['text'][:70].replace(chr(10), '⏎')}")
