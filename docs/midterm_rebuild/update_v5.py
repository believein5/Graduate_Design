# 中期报告第五轮更新（2026-11-25，用户要求）：不再比较自训练模型与作者权重；双扩散模型使用作者公开的预训练权重，
# 报告该权重的生成质量与检测结果。删去：自训练模型的训练与 KID 选择（原 2.5、2.6.1、图 2-12）、自训练权重的各行与相关困难。
# 用法：python -I update_v5.py content.json   （原文件先备份为 content_v4.json，本脚本总是从 content_v4.json 开始）
# 数字来源：results/comparison/benchmark.csv、benchmark_per_class_kid.csv（作者权重各行）
import json, re, shutil, sys
from pathlib import Path

F = Path(sys.argv[1])
if not (F.parent / "content_v4.json").exists():
    shutil.copy(F, F.parent / "content_v4.json")
c = json.load(open(F.parent / "content_v4.json", encoding="utf-8"))


def P(t):
    return {"t": "p", "text": t}


def H(level, num, text):
    return {"t": f"h{level}", "num": num, "text": text}


def idx(start):
    hits = [i for i, x in enumerate(c) if x.get("text", "").startswith(start)]
    assert len(hits) == 1, (start, hits)
    return hits[0]


def head(num):
    hits = [i for i, x in enumerate(c) if x["t"] in ("h1", "h2", "h3") and x["num"] == num]
    assert len(hits) == 1, num
    return hits[0]


def fix(start, old, new):
    i = idx(start)
    assert old in c[i]["text"], (start, old)
    c[i]["text"] = c[i]["text"].replace(old, new)


def drop(start):
    del c[idx(start)]


# ---------- 1.2 ----------
c[idx("（3）完成双扩散模型架构的搭建")] = P(
    "（3）完成双扩散模型架构的搭建：在CRG-DefectDiffuser^[2]^开源代码的基础上补充训练程序和独立的DDIM采样程序，"
    "梳理代码中的问题，跑通协同引导生成流程。双扩散模型采用作者公开的预训练权重。")
c[idx("（4）完成双扩散模型的训练")] = P(
    "（4）完成双扩散模型的生成质量评价：计算两个模型采样结果的KID、FID、生成精确率、生成召回率和最近邻距离比。")
c[idx("（5）完成超参数调试")] = P(
    "（5）完成超参数调试：对协同引导生成的加噪起始步、掩码参数和损失权重等进行了40组设置的筛选，确定了检测模型的输入尺寸。")
fix("（6）进行了初步检测验证", "用作者权重和自训练权重生成扩增数据训练检测模型", "用双扩散模型生成扩增数据训练检测模型")
fix("第一阶段计划的工作已基本完成", "剩余的缺陷先验模型改进将在2027年2月前完成", "剩余的缺陷先验模型改进训练将在2027年2月前完成")

# ---------- 2.1、2.3 ----------
fix("（2）按电路板划分（补充实验）", "测试电路板在训练中从未出现。自训练扩散模型的训练数据只取自这541张图像。",
    "测试电路板不参与检测模型的训练。")
fix("CRG-DefectDiffuser^[2]^是DEFectDiffuser^[1]^的期刊扩展版本", "本课题据此编写了训练程序，训练结果见2.5节。",
    "本课题据此编写了训练程序，用于后续改进缺陷先验模型。本阶段的生成与检测实验均使用作者公开的预训练权重。"
    "本课题的预处理按论文的描述和作者权重的像素尺度设计（0.5倍缩放、64×64缺陷局部块、256×256无缺陷背景块），"
    "与作者权重所用训练数据的格式一致。")

# ---------- 2.5 替换为预训练权重的生成质量 ----------
a, b = head("2.5"), head("2.6")
c[a:b] = [
    H(2, "2.5", "双扩散模型的生成质量"),
    P("为评价两个模型单独学到的分布，用50步DDIM从纯噪声采样（不使用协同引导），缺陷先验模型按类别均衡生成600个局部块，"
      "背景合成模型生成600个背景块，并与真实图像比较。真实图像取自电路板10～12的76张图像，按2.2节相同的方法截取"
      "缺陷局部块和无缺陷背景块。评价指标包括核起始距离KID^[12]^、FID、生成精确率（生成样本落在真实样本分布内的比例，"
      "反映逼真程度）、生成召回率（真实样本被生成样本覆盖的比例，反映多样性）和最近邻距离比（生成样本到训练数据的最近邻"
      "像素距离与真实图像到训练数据的最近邻距离之比，远小于1表示生成样本接近训练样本的复制）。结果见表2-4。"),
    {"t": "table", "num": "2-4", "title": "双扩散模型的生成质量",
     "widths": [2104, 1280, 1280, 1280, 1280, 1280], "align": "center",
     "rows": [["模型", "KID×10³", "FID", "生成精确率", "生成召回率", "最近邻距离比"],
              ["缺陷先验模型", "25.3", "84.3", "0.26", "0.51", "0.97"],
              ["背景合成模型", "98.0", "150.0", "0.07", "0.69", "1.28"]],
     "note": "注：KID与FID越小越好；FID在数百个样本上有偏，仅供参考。"},
    P("两个模型的最近邻距离比均接近或大于1，说明生成样本不是训练样本的复制。缺陷先验模型的生成召回率为0.51，"
      "生成精确率为0.26，即生成的缺陷覆盖了约一半的真实缺陷形态，但多数样本在特征空间中与真实样本仍有差别。"
      "按缺陷类别计算的KID（×10³）为：缺孔26.0、鼠咬36.5、开路34.0、短路30.7、毛刺33.7、多余铜箔37.0，"
      "缺孔最接近真实分布，多余铜箔和鼠咬差别最大。背景合成模型的生成召回率为0.69，覆盖了大部分背景类型，"
      "但生成精确率只有0.07，与图2-9中部分丝印字符模糊的现象一致。协同引导生成时，背景块取自真实图像，"
      "背景合成模型只修改缺陷窗口附近的区域，因此其逼真程度不足对生成图像的影响有限。"),
]

# ---------- 2.6：删去训练步数选择，剩两部分 ----------
c[idx("超参数调试包括三部分")] = P("超参数调试包括两部分：协同引导生成参数的选择和检测模型输入尺寸的选择。")
a, b = head("2.6.1"), head("2.6.2")
del c[a:b]
c[head("2.6.2")]["num"] = "2.6.1"
c[head("2.6.3")]["num"] = "2.6.2"

# ---------- 2.7 ----------
fix("本阶段只比较原始数据与加入CRG-DefectDiffuser生成数据两种情况",
    "生成数据分别用作者权重和2.5节的自训练权重得到。", "")
fix("本阶段只比较原始数据与加入CRG-DefectDiffuser生成数据两种情况",
    "从上到下依次为无缺陷背景、作者权重的生成结果和自训练权重的生成结果。", "上行为无缺陷背景，下行为生成结果。")
t = [x for x in c if x["t"] == "table" and x["title"] == "不同真实数据量下的mAP@50-95"][0]
t["rows"] = [r for r in t["rows"] if r[0] != "CRG（自训练权重）"]
t["rows"] = [["CRG-DefectDiffuser" if r[0] == "CRG（作者权重）" else r[0], *r[1:]] for r in t["rows"]]
t["widths"] = [2004, 1100, 1100, 1100, 1100, 1100, 1000]
t["rows"][0] = ["方法", "每类\n10张", "相对原始\n数据", "每类\n20张", "相对原始\n数据", "每类\n50张", "相对原始\n数据"]
fix("每类10张时，作者权重的生成数据使mAP@50-95提高15.2%", "每类10张时，作者权重的生成数据", "每类10张时，生成数据")
fix("每类10张时，生成数据使mAP@50-95提高15.2%", "自训练权重的结果低于作者权重，与2.5.2节中缺陷先验模型复制训练样本的现象一致。", "")
t = [x for x in c if x["t"] == "table" and x["title"] == "检测模型输入尺寸的比较"][0]
t["note"] = "注：CRG为CRG-DefectDiffuser生成的300张图像；3个随机种子的均值。"

# ---------- 第 3 章 ----------
c[idx("（1）缺陷先验模型的改进（第一阶段剩余工作）")] = P(
    "（1）缺陷先验模型的改进训练（第一阶段剩余工作）。超大缺陷不再二次缩放，在0.5倍尺度下截取覆盖缺陷最多的64×64窗口，"
    "已按此方法重新构建777个局部块；用这些局部块训练缺陷先验模型，按KID选择权重，并检验其对检测结果的影响。")
t = [x for x in c if x["t"] == "table" and x["num"] == "3-1"][0]
t["rows"][1][1] = "完成缺陷先验模型的改进训练与权重选择，完成超参数调试"

# ---------- 第 4 章 ----------
fix("（1）论文报告的提高在本课题的条件下未出现", "（2.6.2节）", "（2.6.1节）")
drop("（2）作者权重的训练数据未知")
drop("（3）自训练缺陷先验模型复制训练样本")
fix("（4）检测指标波动大", "（4）", "（2）")
fix("（5）二次缩放的局部块尺度不一致", "（5）", "（3）")
drop("（2）主实验采用按照片划分")
drop("（3）缺陷先验模型改用只来自按照片划分训练照片的局部块重新训练")
fix("（4）每个设置至少用3个随机种子", "（4）", "（2）")
fix("（5）超大缺陷不再二次缩放", "（5）超大缺陷不再二次缩放：", "（3）改进缺陷先验模型的训练数据：超大缺陷不再二次缩放，")
fix("（3）改进缺陷先验模型的训练数据", "按此方法已重新构建局部块，训练照片中的26个超大缺陷重新截取，10个被舍弃。",
    "按此方法已重新构建局部块，训练照片中的26个超大缺陷重新截取，10个被舍弃，将用于缺陷先验模型的改进训练。")

# ---------- 第 5 章 ----------
fix("本课题已完成开题报告第一阶段的主要工作", "两个扩散模型的训练和超参数调试", "训练程序的编写、双扩散模型生成质量的评价和超参数调试")

# ---------- 图表按出现顺序重新编号（第 2 章），并改正文中的引用 ----------
for kind, word in (("table", "表"), ("cap", "图")):
    items = [x for x in c if x["t"] == kind and x["num"].startswith("2-")]
    remap = {x["num"]: f"2-{k}" for k, x in enumerate(items, 1)}
    for x in items:
        x["num"] = remap[x["num"]]
    pat = re.compile(word + r"(2-\d+)")
    for x in c:
        for key in ("text", "note"):
            if isinstance(x.get(key), str):
                x[key] = pat.sub(lambda m: word + remap.get(m.group(1), m.group(1)), x[key])
    print(word, {k: v for k, v in remap.items() if k != v})
c = [x for x in c if not (x["t"] == "fig" and x["name"] == "fig_kid")]
c = [x for x in c if not (x["t"] == "subcap" and x["text"].startswith("（a）缺陷先验模型；（b）背景合成模型"))]

# ---------- 参考文献：只保留被引用的条目，按首次引用顺序编号 ----------
CITE = re.compile(r"\^\[([\d,\-–]+)\]\^")


def expand(s):
    out = []
    for part in s.split(","):
        if re.search(r"[-–]", part):
            x, y = map(int, re.split(r"[-–]", part)); out += list(range(x, y + 1))
        else:
            out.append(int(part))
    return out


def compress(ns):
    ns, out, k = sorted(ns), [], 0
    while k < len(ns):
        j = k
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        out.append(str(ns[k]) if j == k else (f"{ns[k]},{ns[j]}" if j == k + 1 else f"{ns[k]}-{ns[j]}")); k = j + 1
    return ",".join(out)


def texts():
    for x in c:
        if x["t"] == "ref":
            continue
        for key in ("text", "note"):
            if isinstance(x.get(key), str):
                yield x, key
        if x["t"] == "table":
            for r in x["rows"]:
                for k in range(len(r)):
                    yield r, k


order = []
for obj, key in texts():
    for m in CITE.finditer(obj[key]):
        for n in expand(m.group(1)):
            if n not in order:
                order.append(n)
remap = {old: new for new, old in enumerate(order, 1)}
for obj, key in texts():
    obj[key] = CITE.sub(lambda m: f"^[{compress([remap[n] for n in expand(m.group(1))])}]^", obj[key])
refs = [x for x in c if x["t"] == "ref"]
start = c.index(refs[0])
byn = {r["n"]: r for r in refs}
print("references:", len(order), "· dropped:", [r["n"] for r in refs if r["n"] not in remap])
c[start:start + len(refs)] = [{"t": "ref", "n": remap[o], "text": byn[o]["text"]} for o in order]

# 残留检查：不应再出现自训练、与作者权重的比较
left = [x["text"][:60] for x in c if x["t"] == "p" and (re.search(r"(?<!取)自训练", x["text"]) or "作者权重的生成" in x["text"])]
assert not left, [x["text"] for x in c if x["t"] == "p" and (re.search(r"(?<!取)自训练", x["text"]) or "作者权重的生成" in x["text"])]
json.dump(c, open(F, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(c), "items")
