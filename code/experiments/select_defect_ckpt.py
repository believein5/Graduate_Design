"""重训缺陷先验模型的权重选择与生成质量（2026-11-24）
python select_defect_ckpt.py [--run defect_64_maxcov_trainA]
- 与 2.5 节相同：每个保存的权重（原始与 EMA）用 50 步 DDIM 生成 600 个块（每类 100），与参考集算 KID，取最小者
- 参考集与 2.5 节不同：2.5 节用板 10～12 的照片，其中有划分 A 的测试照片；这里改用划分 A 选择照片的缺陷块
  （切法与训练集相同：未超大的取 defect_64/ 原文件，超大的按 0.5× 取覆盖最多的窗口，覆盖率 < 0.8 的不用）
- 对选中的权重、作者权重、旧自训权重（defect_64_50pc/model150000.pt）在同一参考集上算 KID、FID、生成精确率/召回率；
  选中权重另算最近邻距离比（池 = 它的训练块）
输出 results/comparison/checkpoint_selection_<run>.json，并把选中权重路径写到 runs/<run>/selected/best.txt
"""
import argparse
import csv
import json

import numpy as np
from PIL import Image

import build_defect_maxcov as M                                    # 先导入本目录模块，再由 notebook 把作者仓库加入 sys.path
from common import CKPT_64, PROC, ROOT, SCALE, photo
from train_defect import nb_namespace


def reference():
    split = {r["photo"]: r["split"] for r in csv.DictReader(open(ROOT / "data" / "detection_A" / "photo_split.csv"))}
    out, y = [], []
    for r in csv.DictReader(open(PROC / "defect_manifest.csv")):
        if split[r["source"]] != "select":
            continue
        if float(r["effective_scale"]) < SCALE:
            p, _, _, cov = M.maxcov_crop(photo(r["source"]), M.find_box(r))
            if cov < M.MIN_COV:
                continue
        else:
            p = Image.open(PROC / "defect_64" / r["file"]).convert("RGB")
        out.append(np.asarray(p)); y.append(M.CLASSES.index(r["class"]))
    return out, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="defect_64_maxcov_trainA")
    ap.add_argument("--data", default="defect_64_maxcov_trainA", help="训练块目录（data/pcb_processed/ 下），用作最近邻距离比的池")
    args = ap.parse_args()
    ns = nb_namespace(extra_def_cells=(18, 21))
    import torch
    import torch_fidelity
    ns["torch_fidelity"] = torch_fidelity
    to_t = lambda a: torch.from_numpy(np.stack(a)).permute(0, 3, 1, 2).contiguous()
    ref_l, ref_y = reference()
    ref = to_t(ref_l)
    print(f"reference: {len(ref)} defect patches from split-A select photos", flush=True)

    run = ROOT / "runs" / args.run
    step = lambda p: int(p.stem.split("_")[-1]) if p.stem.startswith("ema") else int(p.stem[5:])
    ckpts = sorted([*run.glob("model*.pt"), *run.glob("ema_*.pt")], key=lambda p: (step(p), p.name))
    y = ns["balanced_labels"](600)
    rows = []
    for ck in ckpts:
        net = ns["load_ckpt"]("64", ck)
        gen = ns["ddim_sample"](net, "64", y=y)
        del net; torch.cuda.empty_cache()
        m = ns["fidelity"](gen, ref, prc=False)
        rows.append(dict(file=ck.name, step=step(ck), weights="ema" if ck.stem.startswith("ema") else "model",
                         kid_x1e3=m["kernel_inception_distance_mean"] * 1e3, fid=m["frechet_inception_distance"]))
        print(f"{ck.name}: KID×1e3 {rows[-1]['kid_x1e3']:.2f} · FID {rows[-1]['fid']:.1f}", flush=True)
    best = min(rows, key=lambda r: r["kid_x1e3"])
    print("selected:", best, flush=True)

    # 同一参考集上比较三个缺陷模型（另一组初始噪声）
    pool = to_t([np.asarray(Image.open(f).convert("RGB")) for f in sorted((PROC / args.data).glob("*.jpg"))])
    compare = {}
    for name, path in [("retrained", run / best["file"]), ("authors", CKPT_64),
                       ("ours_50pc", ROOT / "runs" / "defect_64_50pc" / "model150000.pt")]:
        net = ns["load_ckpt"]("64", path)
        gen = ns["ddim_sample"](net, "64", y=y, seed=ns["SEED"] + 1)
        del net; torch.cuda.empty_cache()
        m = ns["fidelity"](gen, ref)
        compare[name] = dict(checkpoint=str(path), kid_x1e3=m["kernel_inception_distance_mean"] * 1e3,
                             fid=m["frechet_inception_distance"], precision=m["precision"], recall=m["recall"])
        if name == "retrained":
            compare[name]["nn_ratio"] = ns["nn_ratio"](gen, ref, pool)
        print(name, {k: round(v, 3) for k, v in compare[name].items() if isinstance(v, float)}, flush=True)

    res = ROOT / "results" / "comparison"; res.mkdir(parents=True, exist_ok=True)
    json.dump({"reference": "split-A select photos", "n_reference": len(ref), "selection": rows, "best": best, "compare": compare},
              open(res / f"checkpoint_selection_{run.name}.json", "w"), indent=1)
    (run / "selected").mkdir(exist_ok=True)
    (run / "selected" / "best.txt").write_text(str(run / best["file"]))


if __name__ == "__main__":
    main()
