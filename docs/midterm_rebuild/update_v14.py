# 中期报告第十四轮（2026-11-26，用户要求）：2.3.2/2.3.3 精简——只说明训练了多组模型并选出效果最好的一组，
# 去掉 KID 曲线图、模型比较表和过程描述；原 2.3.4 改回 2.3.3；第3、4章中“2.3.3节”的引用改为“2.3.2节”。
# 用法：python -I update_v14.py content.json   （原文件先备份为 content_v13.json，本脚本总是从 content_v13.json 开始）
import json, re, shutil, sys
from pathlib import Path

F = Path(sys.argv[1])
if not (F.parent / "content_v13.json").exists():
    shutil.copy(F, F.parent / "content_v13.json")
c = json.load(open(F.parent / "content_v13.json", encoding="utf-8"))


def at(start):
    hits = [i for i, x in enumerate(c) if x.get("text", "").startswith(start)]
    assert len(hits) == 1, (start, hits)
    return hits[0]


def head(num):
    hits = [i for i, x in enumerate(c) if x["t"] in ("h1", "h2", "h3") and x["num"] == num]
    assert len(hits) == 1, num
    return hits[0]


# 图、表按键引用（同 update_v13.py）
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


for obj, key in texts(c):
    obj[key] = REFPAT.sub(lambda m: "{图:" + figkey[m.group(1)] + "}" if m.group(1) else
                          ("{表:t" + m.group(2) + "}" if m.group(2) else "{式:e" + m.group(3) + "}"), obj[key])

# 2.3.2 表2-2 之后到“单模型采样结果”之前全部换成一段
a = at("该实现只提供了推理代码，本课题补充了训练程序")
b = [k for k, x in enumerate(c) if x["t"] == "h3" and x["text"] == "单模型采样结果"][0]
c[a:b] = [{"t": "p", "text":
           "该实现只提供了推理代码，本课题补充了训练程序，按上述结构用不同的训练数据训练了多组模型，按留出图像上的KID^[9]^选择训练步数，"
           "再与公开的预训练权重^[7]^一起比较，以加入训练集后的检测结果为准，选择效果最好的一组用于后续实验。自训练的缺陷先验模型"
           "训练样本只有几百个，生成样本接近训练样本的复制，检测效果不如预训练权重，因此后续实验采用预训练权重。"}]
c[b - (b - a) + 1]["num"] = "2.3.3"                     # “单模型采样结果”
assert c[a + 1]["text"] == "单模型采样结果"

for start in ("（1）改进缺陷先验模型。",):
    i = at(start)
    c[i]["text"] = c[i]["text"].replace("（2.3.3节）", "（2.3.2节）")

# 第2章图、表重新编号，替换引用
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
assert "2.3.3节" not in " ".join(x.get("text", "") for x in c)
print("figures:", [x["num"] for x in c if x["t"] == "cap"])
print("tables:", [x["num"] for x in c if x["t"] == "table"])
json.dump(c, open(F, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(c), "items")
