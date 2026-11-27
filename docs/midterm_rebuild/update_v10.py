# 中期报告第十轮（2026-11-26，定稿前检查）：
# - 表2-4 补回第六轮改写时漏掉的“设置3”（正文写的是“3组设置”）
# - 表2-5 输入800补到3个随机种子（detection_A_vimg800，s0–2）
# - 2.1.2 节子集的描述与表2-6一致（每类10～40张为嵌套子集，60张在50张基础上补充）
# 用法：python -I update_v10.py content.json   （原文件先备份为 content_v9.json，本脚本总是从 content_v9.json 开始）
import json, shutil, sys
from pathlib import Path

F = Path(sys.argv[1])
if not (F.parent / "content_v9.json").exists():
    shutil.copy(F, F.parent / "content_v9.json")
c = json.load(open(F.parent / "content_v9.json", encoding="utf-8"))


def idx(start):
    hits = [i for i, x in enumerate(c) if x.get("text", "").startswith(start)]
    assert len(hits) == 1, (start, hits)
    return hits[0]


def table(num):
    return next(x for x in c if x["t"] == "table" and x["num"] == num)


t = table("2-4")
assert len(t["rows"]) == 4 and t["rows"][-1][0] == "设置2"
t["rows"].append(["设置3", "掩码温度0.5，阈值0.8", "0.473", "0.377", "0.428"])
t["widths"] = [1900, 2904, 1000, 1100, 1600]

t = table("2-5")
for r in t["rows"]:
    if r[0] == "原始数据，输入800":
        r[1:] = ["0.942", "0.927", "0.951", "0.489"]
    elif r[0] == "原始数据+生成数据，输入800":
        r[1:] = ["0.948", "0.917", "0.953", "0.498"]
t["note"] = "注：生成数据为协同引导生成的300张图像；3个随机种子的均值。"

i = idx("检测实验将693张图像按照片分为互不重叠的三组")
old = "为考察真实数据较少时的情况，又从训练集中按类别取出每类10张和20张两个子集，选择集和测试集不变。"
assert old in c[i]["text"]
c[i]["text"] = c[i]["text"].replace(
    old, "为考察真实数据量的影响，又从训练集中按类别依次取出每类10、20、30、40张的子集，较小的子集包含于较大的子集；"
         "每类60张的训练集在50张的基础上，从训练照片中补充每类10张。各子集的选择集和测试集都相同。")

json.dump(c, open(F, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(c), "items")
