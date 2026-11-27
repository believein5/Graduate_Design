"""划分 A 训练集无缺陷图的来源检查（2026-11-25）：data/detection_A_cleanfix/
detection_A/train 的 50 张无缺陷图是从全部 free_256 中随机取的，其中 25 张来自划分 A 的测试照片、8 张来自选择照片
（缺陷已用电路板模板覆盖，但仍是测试照片的一部分）。这里把这 33 张换成来源为训练照片、未被使用的 free_256 背景块，
其余 317 张（300 张缺陷图 + 17 张无缺陷图）与 select / test 原样保留，文件名不变，只换图像内容。
python build_detection_cleanfix.py    之后 detect.py --split Aclean，与 detection_A_vbase 的结果对照
"""
import csv
import random
import shutil

from PIL import Image

from common import PROC, ROOT, write_sample

SRC, DST = ROOT / "data" / "detection_A", ROOT / "data" / "detection_A_cleanfix"


def main():
    split = {r["photo"]: r["split"] for r in csv.DictReader(open(SRC / "photo_split.csv"))}
    meta = list(csv.DictReader(open(SRC / "meta.csv")))
    free_rows = [r for r in meta if r["split"] == "train" and r["class"] == "free"]
    bad = [r for r in free_rows if split[r["source"]] != "train"]
    used = {(r["source"], r["left"], r["top"]) for r in free_rows}
    # 也不用 detection_A_n60 新加的 10 张，免得两组对照共用同一张替换图时出现混淆
    added = ROOT / "data" / "detection_A_n60" / "meta_added.csv"
    if added.exists():
        used |= {(r["source"], r["left"], r["top"]) for r in csv.DictReader(open(added)) if r["class"] == "free"}
    pool = [r for r in csv.DictReader(open(PROC / "free_manifest.csv"))
            if split[r["source"]] == "train" and (r["source"], r["left"], r["top"]) not in used]
    rng = random.Random(33)
    repl = rng.sample(pool, len(bad))

    if DST.exists():
        shutil.rmtree(DST)
    for s in ("train", "select", "test"):
        shutil.copytree(SRC / s, DST / s, ignore=shutil.ignore_patterns("*.cache"))
    out = []
    for r in meta:
        out.append(dict(r))
    by_name = {r["name"]: r for r in out}
    for b, n in zip(bad, repl):
        write_sample(Image.open(PROC / "free_256" / n["file"]).convert("RGB"), [],
                     DST / "train" / "images" / f"{b['name']}.png", DST / "train" / "labels" / f"{b['name']}.txt")
        by_name[b["name"]].update(source=n["source"], left=n["left"], top=n["top"])
    with open(DST / "meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0])); w.writeheader(); w.writerows(out)
    shutil.copy(SRC / "photo_split.csv", DST / "photo_split.csv")
    roles = {}
    for r in out:
        if r["split"] == "train":
            roles[split[r["source"]]] = roles.get(split[r["source"]], 0) + 1
    assert set(roles) == {"train"}, roles
    print(f"replaced {len(bad)} clean images ({sum(split[r['source']] == 'test' for r in bad)} from test photos, "
          f"{sum(split[r['source']] == 'select' for r in bad)} from select photos); every train image now from train photos: {roles}")


if __name__ == "__main__":
    main()
