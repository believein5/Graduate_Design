# 中期报告第十三轮（2026-11-26，用户要求）：2.3.2 改为“模型结构与训练”，写明按该结构自行训练并按 KID 选择训练步数；
# 新增 2.3.3“模型的比较与选择”，比较自训练模型（300 个局部块、777 个不二次缩放局部块）与公开预训练权重，按复制程度和检测结果选模型；
# 原 2.3.3 改为 2.3.4；第 3、4 章与之对应的缺陷先验模型改进内容同步修改。
# 用法：python -I update_v13.py content.json   （原文件先备份为 content_v12.json，本脚本总是从 content_v12.json 开始）
# 数字来源：results/comparison/benchmark.csv、checkpoint_selection.json、checkpoint_selection_defect_64_maxcov_trainA.json；
#          results/experiments/detection_{A,A10}_img640、detection_A_vbase 的 crgours_1x / crgmaxcov_1x / crg_1x / original
import json, re, shutil, sys
from pathlib import Path

F = Path(sys.argv[1])
if not (F.parent / "content_v12.json").exists():
    shutil.copy(F, F.parent / "content_v12.json")
c = json.load(open(F.parent / "content_v12.json", encoding="utf-8"))


def P(t): return {"t": "p", "text": t}
def H(level, num, text): return {"t": f"h{level}", "num": num, "text": text}


def at(start):
    hits = [i for i, x in enumerate(c) if x.get("text", "").startswith(start)]
    assert len(hits) == 1, (start, hits)
    return hits[0]


def head(num):
    hits = [i for i, x in enumerate(c) if x["t"] in ("h1", "h2", "h3") and x["num"] == num]
    assert len(hits) == 1, num
    return hits[0]


# ---------- 第2章的图、表改为按键引用（同 update_v12.py）----------
for k, x in enumerate(c):
    if x["t"] == "cap":
        x["key"] = c[k - 1]["name"] if c[k - 1]["t"] == "fig" else c[k - 2]["name"]
    elif x["t"] in ("table", "eq"):
        x["key"] = x["t"][0] + x["num"]
figkey = {x["num"]: x["key"] for x in c if x["t"] == "cap"}
REFPAT = re.compile(r"图(\d+-\d+)|表(\d+-\d+)|式\((\d+-\d+)\)")


def texts(items):
    for x in items:
        if x["t"] == "ref":
            continue
        if isinstance(x.get("text"), str) and x["t"] != "cap":
            yield x, "text"
        if isinstance(x.get("note"), str):
            yield x, "note"
        if x["t"] == "note":
            for it in x["items"]:
                yield it, 1
        if x["t"] == "table":
            for r in x["rows"]:
                for j in range(len(r)):
                    yield r, j


def to_keys(s):
    return REFPAT.sub(lambda m: "{图:" + figkey[m.group(1)] + "}" if m.group(1) else
                      ("{表:t" + m.group(2) + "}" if m.group(2) else "{式:e" + m.group(3) + "}"), s)


for obj, key in texts(c):
    obj[key] = to_keys(obj[key])

# ---------- 2.3.2 模型结构与训练 ----------
i = head("2.3.2")
c[i]["text"] = "模型结构与训练"
c[at("双扩散模型的结构和预训练权重采用已公开的实现")] = P(
    "双扩散模型采用已公开实现的网络结构^[7]^。两个模型都是基于改进DDPM的U-Net^[4,8]^，扩散步数为1000，采用线性噪声调度，"
    "网络同时预测噪声和方差，主要配置见{表:t2-2}。")
j = after = next(k for k, x in enumerate(c) if x.get("key") == "t2-2") + 1
c[j:j] = [
    P("该实现只提供了推理代码，本课题补充了训练程序，按上述结构在本课题的训练数据上训练两个模型：缺陷先验模型的批次为64，学习率为"
      "$10^{-4}$，参数滑动平均系数为0.9999，训练200000次迭代；背景合成模型的批次为8，训练500000次迭代。在单张RTX 4090上两个模型"
      "分别用时约8 h和48 h。训练损失在几万次迭代后基本不再下降，不能据此判断何时停止，因此定期保存权重，对每个权重采样后计算与"
      "留出图像（电路板10～12的一半）之间的KID^[11]^，选取KID最小的权重，见{图:fig_kid}。"),
    {"t": "fig", "name": "fig_kid"}, {"t": "cap", "num": "", "key": "fig_kid", "text": "训练过程中各权重的KID"},
    {"t": "subcap", "text": "（a）缺陷先验模型；（b）背景合成模型"},
    P("缺陷先验模型的KID在约20000次迭代后降到20～40（×10³），第150000次迭代时最低；背景合成模型原始权重的KID在相邻权重之间"
      "波动较大，第375000次迭代时最低，滑动平均权重则平稳下降。"),
    H(3, "2.3.3", "模型的比较与选择"),
    P("为了选出生成效果最好的模型，比较了三组缺陷先验模型：用300个局部块（每类50个）训练的模型；用777个局部块训练的模型，这些局部块"
      "只取自检测训练照片，较大缺陷不再二次缩放，而是在0.5倍尺度下以缺陷框为中心截取，窗口覆盖缺陷框不足80%的舍弃；以及公开的"
      "预训练权重。比较分两方面：一是最近邻距离比，即生成样本到训练局部块的最近邻像素距离与真实留出样本到训练局部块的距离之比，"
      "远小于1说明生成样本接近训练样本的复制；二是检测结果，把各模型协同引导生成的300张图像加入训练集，在每类10张和50张真实图像"
      "的条件下比较mAP@50-95。结果见{表:t_cmp}。"),
    {"t": "table", "num": "", "key": "t_cmp", "title": "不同缺陷先验模型的比较",
     "widths": [2804, 1500, 1400, 1400, 1400], "align": "center",
     "rows": [["缺陷先验模型", "最近邻距离比", "生成召回率", "每类10张", "每类50张"],
              ["只用真实数据", "—", "—", "0.388", "0.497"],
              ["自训练，300个局部块", "0.06", "0.38", "0.392", "0.484"],
              ["自训练，777个局部块", "0.06", "—", "0.392", "0.486"],
              ["预训练权重", "0.97", "0.51", "0.447", "0.494"]],
     "note": "注：每类10张、50张两列为mAP@50-95，3个随机种子的均值；300个局部块的模型与自训练的背景合成模型配合生成，其余两组与"
             "预训练的背景合成模型配合；生成召回率以电路板10～12的真实局部块为参照。"},
    P("两个自训练模型的KID都低于预训练权重，但最近邻距离比只有0.06左右：训练局部块只有几百个，训练到上万次迭代后模型基本记住了"
      "训练样本，生成的缺陷缺少新的形态，加入训练集后检测指标与只用真实数据相当。不二次缩放、统一局部块尺度后，这一现象没有改善。"
      "预训练权重的生成样本没有复制现象，每类10张时使mAP@50-95提高15.2%，效果最好。背景合成模型也做了同样的比较：自训练模型的"
      "KID较低，但生成召回率只有0.20，覆盖的背景类型偏少；协同引导时背景直接取自真实图像，背景合成模型只修改缺陷窗口附近的区域，"
      "对结果影响较小。综合比较，后续的生成与检测实验采用预训练权重。可见对缺陷先验模型而言，KID低并不代表生成数据有用，"
      "选择模型时还需要检查是否复制了训练样本，并以检测结果为准。"),
]
c[head("2.3.3") if False else [k for k, x in enumerate(c) if x["t"] == "h3" and x["text"] == "单模型采样结果"][0]]["num"] = "2.3.4"
c[at("不使用协同引导，从纯噪声分别对两个模型采样")]["text"] = c[at("不使用协同引导，从纯噪声分别对两个模型采样")]["text"].replace(
    "从纯噪声分别对两个模型采样", "从纯噪声分别对选定的两个模型采样")
i = at("两个模型各采样600张图像")
c[i]["text"] = c[i]["text"].replace("两个模型各采样600张图像", "选定的两个模型各采样600张图像")

# ---------- 1.2 ----------
i = at("目前第一阶段的工作已基本完成")
a = "补充了训练程序和DDIM采样程序，评价了两个模型的生成质量，"
assert a in c[i]["text"]
c[i]["text"] = c[i]["text"].replace(a, "补充了训练程序和DDIM采样程序，用不同训练数据训练并比较了多组模型，选出生成效果最好的一组，评价了其生成质量，")

# ---------- 第 3、4 章 ----------
c[at("（1）改进缺陷先验模型。")] = P(
    "（1）改进缺陷先验模型。自训练的缺陷先验模型容易复制训练样本（2.3.3节）。后续拟对缺陷局部块做旋转和翻转增强，增加训练样本的"
    "多样性，并以最近邻距离比和少样本条件下的检测结果选择权重。")
c[at("（3）缺陷局部块的尺度不一致。")] = P(
    "（3）自训练的缺陷先验模型容易复制训练样本。缺陷局部块只有几百个，模型训练后基本记住了训练样本，生成数据对检测的作用不如预训练"
    "权重；较大缺陷不再二次缩放、统一局部块尺度后，这一现象没有改善。")
c[at("（3）较大缺陷改为在原尺度下截取")] = P(
    "（3）对缺陷局部块做旋转、翻转等增强，增加训练样本的多样性；训练中按最近邻距离比监视复制程度，并以少样本条件下的检测结果"
    "选择权重。")

# ---------- 第2章图、表重新编号，替换引用 ----------
num, ch, cnt = {}, 0, {}
for x in c:
    if x["t"] == "h1" and x["num"]:
        ch = int(x["num"]); cnt = {"cap": 0, "table": 0, "eq": 0}
    if x["t"] in ("cap", "table", "eq"):
        if ch in (1, 2):
            cnt[x["t"]] += 1
            x["num"] = f"{ch}-{cnt[x['t']]}"
        num[x["key"]] = x["num"]
KEYPAT = re.compile(r"\{(图|表|式):([^}]+)\}")
for obj, key in texts(c):
    obj[key] = KEYPAT.sub(lambda m: ("式(" + num[m.group(2)] + ")") if m.group(1) == "式" else m.group(1) + num[m.group(2)], obj[key])
for x in c:
    x.pop("key", None)
assert not [obj[key] for obj, key in texts(c) if re.search(r"\{(图|表|式):", obj[key])]
# ---------- 参考文献按首次引用重新编号（同 update_v6.py）----------
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


allt = [(x, "text") for x in c if x["t"] != "ref" and isinstance(x.get("text"), str)] + list(
    (o, k) for o, k in texts(c) if not (isinstance(o, dict) and k == "text"))
order = []
for obj, key in allt:
    for m in CITE.finditer(obj[key]):
        for n in expand(m.group(1)):
            if n not in order:
                order.append(n)
remap = {o: n for n, o in enumerate(order, 1)}
for obj, key in allt:
    obj[key] = CITE.sub(lambda m: f"^[{compress([remap[n] for n in expand(m.group(1))])}]^", obj[key])
refs = [x for x in c if x["t"] == "ref"]
start = c.index(refs[0])
byn = {r["n"]: r for r in refs}
assert set(order) == set(byn), (set(byn) - set(order))
c[start:start + len(refs)] = [{"t": "ref", "n": remap[o], "text": byn[o]["text"]} for o in order]
print("refs remap:", {k: v for k, v in remap.items() if k != v})
body = " ".join(x.get("text", "") for x in c if x["t"] in ("p", "h3"))
for w in ("CRG", "作者", "原论文", "复现"):
    assert w not in body, w
print("figures:", [x["num"] for x in c if x["t"] == "cap"])
print("tables:", [x["num"] for x in c if x["t"] == "table"])
json.dump(c, open(F, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(c), "items")
