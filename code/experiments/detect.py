"""P4/P5：训练 YOLOv11n 并在真实测试集上评估

用法: python detect.py --config crg_2x --seed 0
  config = original                      只用真实训练数据（300 缺陷 + 50 无缺陷）
           <method>_<k>x                 真实训练数据 + 前 300·k 个合成样本（k = 1, 2, 3；每类均衡）
           <method>_synonly              只用前 300 个合成样本（消融，按 CRG 论文 Table III 的做法）
           <method>_all                  真实训练数据 + 该方法目录下的全部合成样本（用于筛选后、数量不足 300 的集合）
           <method>_synall               只用该方法目录下的全部合成样本（两阶段训练的第一阶段，2026-11-25）
           <m1>+<m2>_<k>x                多个方法各取前 300·k 个（混合）
  --variant <名字>                       检测器设置变体（det_variants.py），结果写到 detection_<split>_v<名字>/；
                                         另外在 select 上评估 best.pt，写入 metrics.json 的 "select"
训练设置按 CRG 论文：200 epoch，batch 32；图像 256×256；训练中的验证用 select 半区（选 best.pt），
最终指标在 test 半区计算，结果写入 results/experiments/detection/<config>/seed<k>/metrics.json。
"""
import argparse
import gc
import json
import shutil
import sys
import time

from common import CLASSES, DET, RES, SYN

EPOCHS, BATCH, IMGSZ = 200, 32, 256
SPLIT_DIRS = {"board": DET, "A": DET.parent / "detection_A", "A10": DET.parent / "detection_A_n10",
              "A20": DET.parent / "detection_A_n20", "Aedge": DET.parent / "detection_A_edge",
              "Agrid": DET.parent / "detection_A_grid",
              # 2026-11-25 数据量趋势：每类 30 / 40 张（A30、A40 嵌套于 50 张）与 60 张（A60 = 50 张 + 每类 10 张新裁剪）
              "A30": DET.parent / "detection_A_n30", "A40": DET.parent / "detection_A_n40",
              "A60": DET.parent / "detection_A_n60",
              # 2026-11-25 检查：训练集 33 张来自测试 / 选择照片的无缺陷图换成训练照片的背景块（build_detection_cleanfix.py）
              "Aclean": DET.parent / "detection_A_cleanfix"}


def dataset_yaml(config, out, det=DET, suffix=""):
    train = [str(det / "train" / "images")]
    if config != "original":
        method, scale = config.rsplit("_", 1)
        n = None if scale in ("all", "synall") else 300 if scale == "synonly" else 300 * int(scale[:-1])
        imgs = []
        for m in method.split("+"):
            got = sorted((SYN / f"{m}{suffix}" / "images").glob("j*.png"))[:n]
            assert n is None or len(got) == n, f"{m}: only {len(got)} of {n} synthetic images exist"
            imgs += got
        lst = out / "synthetic.txt"
        lst.write_text("\n".join(str(p) for p in imgs) + "\n")
        train = [str(lst)] if scale in ("synonly", "synall") else train + [str(lst)]
    y = out / "data.yaml"
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    for split in ["select", "test"]:
        (out / f"data_{split}.yaml").write_text(
            f"train:\n" + "".join(f"  - {t}\n" for t in train) + f"val: {det / split / 'images'}\nnames:\n{names}\n")
    return out / "data_select.yaml", out / "data_test.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--keep-weights", action="store_true", help="诊断用：结果写到 detection_diag/，保留 best.pt")
    ap.add_argument("--split", default="board", choices=list(SPLIT_DIRS),
                    help="board：按板划分（默认）；A：论文式按照片划分；A10 / A20：划分 A 的训练集只取每类 10 / 20 张（select、test 不变）；"
                         "Aedge：划分 A 的训练与 select，测试裁剪中缺陷可被边缘截断（build_detection_edge_test.py）；"
                         "Agrid：同上，测试裁剪取 256 网格块（build_detection_grid_test.py）")
    ap.add_argument("--variant", default=None, help="det_variants.py 中的检测器设置变体，可用 + 组合")
    ap.add_argument("--init", default=None, help="从这个 .pt 开始训练（两阶段训练的第二阶段），代替 yolo11n.pt；须同时给 --variant")
    ap.add_argument("--proto", default="select", choices=["select", "valtest", "img640", "valtest640", "dmgr640"],
                    help="协议检查：select 用 select 选 best.pt（默认）；valtest 训练中直接用 test 验证并选 best.pt；img640 输入 640；"
                         "valtest640 两者同时；dmgr640 输入 640，按 DMGR 论文的检测器设置：AdamW、lr0 0.01、余弦学习率")
    args = ap.parse_args()
    import torch
    from ultralytics import YOLO

    sub = "detection_diag" if args.keep_weights else ("detection" if args.split == "board" else f"detection_{args.split}")
    if args.proto != "select":
        sub = f"{sub}_{args.proto}"
    imgsz = 640 if args.proto in ("img640", "valtest640", "dmgr640") else IMGSZ
    # 默认 optimizer=auto：迭代数 < 10k 时 Ultralytics 自动选 AdamW、lr0 = 0.002·5/(4+nc) = 0.001，并忽略 lr0 设置
    opt_kw = dict(optimizer="AdamW", lr0=0.01, cos_lr=True) if args.proto == "dmgr640" else {}
    weights, epochs = "yolo11n.pt", args.epochs
    if args.variant:
        import det_variants
        opt_kw = det_variants.get(args.variant)
        imgsz, epochs, weights = opt_kw.pop("imgsz", 640), opt_kw.pop("epochs", args.epochs), opt_kw.pop("model", weights)
        transpose = opt_kw.pop("transpose", 0)
        if transpose:                                                    # 随机转置，与上下、左右翻转组合为 8 种 90° 旋转 / 镜像
            import aug_rot90
            aug_rot90.enable(transpose)
        sub = f"detection_{args.split}_v{args.variant}"
    if args.init:
        assert args.variant, "--init 需要 --variant（结果另存目录）"
        weights = args.init
    out = RES / sub / args.config / f"seed{args.seed}"
    det = SPLIT_DIRS[args.split]
    if (out / "metrics.json").exists():
        print("done already:", out); return
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    y_sel, y_test = dataset_yaml(args.config, out, det, "" if args.split == "board" else "_A")
    # Ultralytics 在第 1 个 epoch 显存不足时会自动把 batch 减半并继续，结果与 batch 32 不可比（2026-11-19 发现一例）。
    # 这里一旦发现 batch 被减小就中止，等 5 分钟后从头重训，最多 6 次
    def guard_batch(trainer):
        if trainer.batch_size != BATCH:
            raise RuntimeError(f"batch reduced to {trainer.batch_size}")

    for attempt in range(6):
        model = YOLO(weights).load("yolo11n.pt") if weights.endswith(".yaml") else YOLO(weights)   # yaml：新结构，迁移可用的预训练权重
        model.add_callback("on_train_batch_end", guard_batch)
        try:
            model.train(data=str(y_test if args.proto in ("valtest", "valtest640") else y_sel), epochs=epochs, batch=BATCH, imgsz=imgsz,
                        seed=args.seed, deterministic=True, project=str(out), name="train", exist_ok=True, workers=4,
                        verbose=False, plots=False, **opt_kw)
            break
        except RuntimeError as e:
            if "batch reduced" not in str(e):
                raise
            print(f"attempt {attempt + 1}: {e}; restarting from scratch in 5 min", flush=True)
            del model; gc.collect(); torch.cuda.empty_cache()
            shutil.rmtree(out / "train", ignore_errors=True); time.sleep(300)
    else:
        sys.exit("batch was reduced in 6 attempts; no result written")
    best = YOLO(str(out / "train" / "weights" / "best.pt"))
    m = best.val(data=str(y_test), split="val", imgsz=imgsz, batch=BATCH, project=str(out), name="test", plots=False, verbose=False)
    b = m.box
    res = {"config": args.config, "seed": args.seed, "map50_95": float(b.map), "map50": float(b.map50),
           "precision": float(b.mp), "recall": float(b.mr),
           "per_class": {CLASSES[int(c)]: {"map50_95": float(b.maps[int(c)]), "recall": float(b.r[i]), "precision": float(b.p[i]),
                                          "map50": float(b.ap50[i])} for i, c in enumerate(b.ap_class_index)}}
    if args.variant:                                                     # 同一个 best.pt 在 select 上的分数，用于选设置
        ms = best.val(data=str(y_sel), split="val", imgsz=imgsz, batch=BATCH, project=str(out), name="select", plots=False, verbose=False).box
        res["select"] = {"map50_95": float(ms.map), "map50": float(ms.map50), "precision": float(ms.mp), "recall": float(ms.mr)}
        res["variant"] = args.variant
    json.dump(res, open(out / "metrics.json", "w"), indent=1)
    if not args.keep_weights:
        (out / "train" / "weights" / "last.pt").unlink(missing_ok=True)   # 保留 best.pt（5.5 MB），以后可在其他测试集上重新评估
    print(json.dumps({k: v for k, v in res.items() if k != "per_class"}))


if __name__ == "__main__":
    main()
