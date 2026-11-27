# 中期报告第九轮（2026-11-26，用户要求）：改写真实数据与生成数据关系的描述——
# 每类50张时生成数据与只用真实数据相当且不干扰检测（其他方法会降低指标）；少样本时生成数据可替代一部分真实数据（约6～8张生成≈1张真实）。
# 用法：python -I update_v9.py content.json   （原文件先备份为 content_v8.json，本脚本总是从 content_v8.json 开始）
# 插值依据：选择集 mAP@50-95，原始数据每类10/20/30张 0.4169/0.4853/0.5020，加入生成数据每类10/20张 0.4708/0.4955
import json, shutil, sys
from pathlib import Path

F = Path(sys.argv[1])
if not (F.parent / "content_v8.json").exists():
    shutil.copy(F, F.parent / "content_v8.json")
c = json.load(open(F.parent / "content_v8.json", encoding="utf-8"))


def idx(start):
    hits = [i for i, x in enumerate(c) if x.get("text", "").startswith(start)]
    assert len(hits) == 1, (start, hits)
    return hits[0]


c[idx("真实数据越少，生成数据的作用越明显")]["text"] = (
    "真实数据越少，生成数据的作用越明显：每类10张时mAP@50-95提高15.2%，3个随机种子都高于只用真实数据的结果；每类20张时提高5.0%；"
    "每类30张时只提高1.7%；每类40张及以上时与只用真实数据相当，差别在随机种子的波动范围内。从替代真实数据的角度看，每类10张真实"
    "图像加入300张生成图像后，mAP@50-95与每类20张真实图像相当；按选择集的结果插值，300张生成图像约相当于每类增加6～8张真实图像，"
    "即约6～8张生成图像相当于1张真实图像。真实数据充足时，加入生成数据与只用真实数据相当，说明生成的缺陷和标签框与真实数据基本"
    "一致，不会干扰检测模型，但也不再带来新的信息。如果完全不用真实图像，只用300张生成图像训练，mAP@50-95只有0.335，召回率为"
    "0.644，生成数据只适合作为真实数据的补充，在真实缺陷样本少时替代一部分采集和标注工作。")

json.dump(c, open(F, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(c), "items")
