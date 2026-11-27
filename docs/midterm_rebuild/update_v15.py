# 中期报告第十五轮（2026-11-27，导师要求补原理图）：2.4.1 加协同引导的原理图（一个去噪步的计算流程）和去噪过程中估计与掩码的变化图。
# 两幅图由 docs/thesis_build/make_principle.py 绘制，素材为 code/experiments/capture_guidance.py 对任务 j0021（短路）的一次实际运行。
# 另同步用户在 Word 中的两处修改（见下）。
# 用法：python -I update_v15.py content.json   （原文件先备份为 content_v14.json，本脚本总是从 content_v14.json 开始）
import json, re, shutil, sys
from pathlib import Path

F = Path(sys.argv[1])
if not (F.parent / "content_v14.json").exists():
    shutil.copy(F, F.parent / "content_v14.json")
c = json.load(open(F.parent / "content_v14.json", encoding="utf-8"))


def P(t): return {"t": "p", "text": t}
def FIG(name, cap): return [{"t": "fig", "name": name}, {"t": "cap", "num": "", "key": name, "text": cap}]


def at(start):
    hits = [i for i, x in enumerate(c) if x.get("text", "").startswith(start)]
    assert len(hits) == 1, (start, hits)
    return hits[0]


# 图、表按键引用（同 update_v14.py）
for k, x in enumerate(c):
    if x["t"] == "cap" and "key" not in x:
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

# 同步用户 2026-11-26 18:32 在 Word 中的修改：2.3.2 “只提供了” → “提供了”；进度表删去 2026年11月—2027年2月 一行
i = at("该实现只提供了推理代码")
c[i]["text"] = c[i]["text"].replace("该实现只提供了推理代码", "该实现提供了推理代码", 1)
tab = next(x for x in c if x["t"] == "table" and x.get("title") == "后期进度安排")
n0 = len(tab["rows"]); tab["rows"] = [r for r in tab["rows"] if not r[0].startswith("2026年11月")]; assert len(tab["rows"]) == n0 - 1

# 2.4.1 第一段之后：原理图
i = at("生成时以一张256×256的无缺陷图像")
c[i]["text"] += "一个去噪步内的计算过程见{图:fig_principle}，图中各处为一次实际生成的中间结果。"
c[i + 1:i + 1] = FIG("fig_principle", "协同引导的计算过程")

# 2.4.1 末尾（引导步长的符号说明之后）：去噪过程中估计与掩码的变化
i = at("每个去噪步中，先按")
assert c[i + 1]["t"] == "eq" and c[i + 2]["t"] == "note"
c[i + 3:i + 3] = [
    P("{图:fig_mask_steps}给出了同一次生成中各时间步的中间结果，从上到下依次为完整图像估计在窗口内的部分、缺陷估计和缺陷区域"
      "掩码，时间步为250步采样序列中的序号，从第29步开始去噪。缺陷先验模型"
      "从第一步起就在两条线路之间给出了铜桥；背景合成模型对窗口的估计在开始几步仍是原来的线路，从第20步左右开始出现铜桥，并逐步"
      "清晰；缺陷区域的掩码在开始几步就落在铜桥处，此后基本不变。可见协同引导把缺陷先验模型给出的缺陷形态逐步传递到整幅图像中，"
      "而窗口外的区域保持不变。"),
    *FIG("fig_mask_steps", "去噪过程中的估计与缺陷区域掩码"),
]

# 第1、2章图、表、公式重新编号，替换引用（同 update_v14.py）
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
print("figures:", [(x["num"], x["text"]) for x in c if x["t"] == "cap"][9:])
json.dump(c, open(F, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(c), "items")
