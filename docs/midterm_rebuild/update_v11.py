# 中期报告第十一轮（2026-11-26，定稿，用户要求）：说明检测训练集的含义——
# 训练照片中缺陷并不稀缺（1044 个，每类 166～182 个），每类 50 张为基本设置；每类 10、20 张模拟缺陷样本少的情况；
# 生成数据固定 300 张时与真实缺陷图像的数量比例。
# 用法：python -I update_v11.py content.json   （原文件先备份为 content_v10.json，本脚本总是从 content_v10.json 开始）
# 数字来源：data/detection_A/photo_split.csv 与 PKU 标注（训练照片的缺陷数）
import json, shutil, sys
from pathlib import Path

F = Path(sys.argv[1])
if not (F.parent / "content_v10.json").exists():
    shutil.copy(F, F.parent / "content_v10.json")
c = json.load(open(F.parent / "content_v10.json", encoding="utf-8"))


def idx(start):
    hits = [i for i, x in enumerate(c) if x.get("text", "").startswith(start)]
    assert len(hits) == 1, (start, hits)
    return hits[0]


i = idx("检测实验将693张图像按照片分为互不重叠的三组")
t = c[i]["text"]
a = "训练照片用于截取训练图像，每类50张，另加50张无缺陷图像；"
b = "各子集的选择集和测试集都相同。"
assert a in t and t.endswith(b)
t = t.replace(a, "训练照片共有1044个缺陷（每类166～182个），从中截取训练图像，每类50张作为基本设置，另加50张无缺陷图像；")
t = t[:-len(b)] + "其中每类10张和20张的子集用于模拟实际生产中缺陷样本很少的情况。" + b
c[i]["text"] = t

i = idx("为考察生成数据的作用与真实数据量的关系")
a = "加入的生成数据固定为300张，结果见表2-6。"
assert c[i]["text"].endswith(a)
c[i]["text"] = c[i]["text"].replace(a, "加入的生成数据固定为300张：每类10张时生成图像的数量是真实缺陷图像的5倍，每类50张时两者相同。结果见表2-6。")

json.dump(c, open(F, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(c), "items")
