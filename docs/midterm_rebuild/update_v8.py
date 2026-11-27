# 中期报告第八轮（2026-11-26，用户要求）：表 2-6 补每类 30、40、60 张；保留用户在 Word 中删去的 1.2 节最后一段。
# 用法：python -I update_v8.py content.json   （原文件先备份为 content_v7.json，本脚本总是从 content_v7.json 开始）
# 数字来源：results/experiments/detection_A{10,20,30,40,60}_img640/、detection_A_img640/ 的 original 与 crg_1x，各 3 个种子
import json, shutil, sys
from pathlib import Path

F = Path(sys.argv[1])
if not (F.parent / "content_v7.json").exists():
    shutil.copy(F, F.parent / "content_v7.json")
c = json.load(open(F.parent / "content_v7.json", encoding="utf-8"))


def idx(start):
    hits = [i for i, x in enumerate(c) if x.get("text", "").startswith(start)]
    assert len(hits) == 1, (start, hits)
    return hits[0]


# 用户 2026-11-25 17:32 在 Word 中删去的段落
del c[idx("第一阶段剩余的工作是改进缺陷先验模型的训练数据并重新训练")]

t = next(x for x in c if x["t"] == "table" and x["num"] == "2-6")
t["widths"] = [2204, 2000, 2400, 1900]
t["rows"] = [["每类真实图像数", "原始数据", "原始数据+生成数据", "变化"],
             ["10", "0.388", "0.447", "+15.2%"],
             ["20", "0.446", "0.468", "+5.0%"],
             ["30", "0.478", "0.486", "+1.7%"],
             ["40", "0.494", "0.488", "−1.3%"],
             ["50", "0.497", "0.494", "−0.5%"],
             ["60", "0.490", "0.490", "−0.1%"]]
t["note"] = ("注：生成数据固定为300张；变化为相对只用原始数据的变化；3个随机种子的均值，"
             "每类30张及以上时mAP@50-95在种子间的标准差为0.002～0.013。")

c[idx("不同真实数据量下的结果见表2-6")]["text"] = (
    "为考察生成数据的作用与真实数据量的关系，训练集取每类10～60张真实图像，加入的生成数据固定为300张，结果见表2-6。")
c[idx("每类50张时，加入生成数据后")]["text"] = (
    "真实数据越少，生成数据的作用越明显：每类10张时mAP@50-95提高15.2%，3个随机种子都高于只用真实数据的结果；"
    "每类20张时提高5.0%；每类30张时只提高1.7%；每类40张及以上时与只用真实数据相当，差别在随机种子的波动范围内。"
    "可见真实数据充足时，检测模型已能较好地学到缺陷特征，生成数据主要在少样本条件下发挥作用，分界大约在每类20～30张之间。"
    "如果完全不用真实图像，只用300张生成图像训练，mAP@50-95只有0.335，召回率为0.644，说明生成的缺陷与真实缺陷仍有差距，"
    "生成数据只适合作为真实数据的补充。")

json.dump(c, open(F, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(c), "items")
