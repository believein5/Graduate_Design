# P6、P7 检测端改进的汇总（2026-11-26）：每个想法 3 个种子的 select 均值与对照比较，test 只用于报告
# select 分数统一取 train/results.csv 中 fitness 最高的一行（img640 对照没有 metrics.json 的 "select"，两边用同一口径）
# 用法：python -I summarize_p6.py
import csv, json
from pathlib import Path
from statistics import mean

R = Path(__file__).resolve().parents[2] / "results" / "experiments"


def sel(run):
    rows = list(csv.DictReader(open(run / "train" / "results.csv")))
    k = lambda r: 0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"])
    return float(max(rows, key=k)["metrics/mAP50-95(B)"])


def stats(d):
    runs = [R / d / f"seed{s}" for s in range(3)]
    m = [json.load(open(r / "metrics.json")) for r in runs]
    return dict(sel=mean(sel(r) for r in runs), test=mean(x["map50_95"] for x in m),
                P=mean(x["precision"] for x in m), R=mean(x["recall"] for x in m), ep=[len(list(open(r / "train" / "results.csv"))) - 1 for r in runs])


PAIRS = [  # (想法, 数据量, 新设置, 对照)
    ("two-stage ft", "50", "detection_Aclean_vft/original", "detection_Aclean_vbase/original"),
    ("two-stage ft", "10", "detection_A10_vft/original", "detection_A10_img640/original"),
    ("two-stage ft", "20", "detection_A20_vft/original", "detection_A20_img640/original"),
    ("relabel CRG", "50", "detection_Aclean_vbase/crgrelab_1x", "detection_Aclean_vbase/crg_1x"),
    ("relabel CRG", "10", "detection_A10_vbase/crgrelab_1x", "detection_A10_img640/crg_1x"),
    ("rot90 Original", "50", "detection_Aclean_vrot90/original", "detection_Aclean_vbase/original"),
    ("rot90 CRG", "50", "detection_Aclean_vrot90/crg_1x", "detection_Aclean_vbase/crg_1x"),
    ("rot90 Original", "10", "detection_A10_vrot90/original", "detection_A10_img640/original"),
    ("rot90 CRG", "10", "detection_A10_vrot90/crg_1x", "detection_A10_img640/crg_1x"),
    # P7（2026-11-26）：组合
    ("rot90 Original", "20", "detection_A20_vrot90/original", "detection_A20_img640/original"),
    ("rot90 CRG", "20", "detection_A20_vrot90/crg_1x", "detection_A20_img640/crg_1x"),
    ("rot90+ft vs rot90", "10", "detection_A10_vrot90+ft/original", "detection_A10_vrot90/original"),
    ("rot90+ft vs ft", "10", "detection_A10_vrot90+ft/original", "detection_A10_vft/original"),
    ("rot90+ft vs rot90", "20", "detection_A20_vrot90+ft/original", "detection_A20_vrot90/original"),
    ("rot90+ft vs ft", "20", "detection_A20_vrot90+ft/original", "detection_A20_vft/original"),
    ("rot90+ft vs rotCRG", "10", "detection_A10_vrot90+ft/original", "detection_A10_vrot90/crg_1x"),
    ("rot90+ft vs rotCRG", "20", "detection_A20_vrot90+ft/original", "detection_A20_vrot90/crg_1x"),
]
print(f"{'idea':16}{'n':>4}{'sel new':>9}{'sel ref':>9}{'d sel':>8}{'test new':>10}{'test ref':>10}  P/R new        epochs new")
for idea, n, new, ref in PAIRS:
    a, b = stats(new), stats(ref)
    print(f"{idea:16}{n:>4}{a['sel']:9.4f}{b['sel']:9.4f}{a['sel'] - b['sel']:+8.4f}{a['test']:10.4f}{b['test']:10.4f}  "
          f"{a['P']:.3f}/{a['R']:.3f}  {a['ep']}")
s = stats("detection_A_vbase/crg_synall")
print(f"stage 1 (900 CRG only): sel {s['sel']:.4f} test {s['test']:.4f}")
print("\nensembles (WBF, eval_ensemble.py)")
for k in ("original_ens3", "crg_1x_ens3", "mix6", "original_ens5", "crg_1x_ens5", "mix10", "clean_original_ens3", "clean_crg_1x_ens3",
          "clean_mix6", "rot90_original_ens3", "rot90_crg_1x_ens3", "rot90_mix6", "flipud_original_ens3", "flipud_crg_1x_ens3", "copypaste_2x_ens5"):
    e = json.load(open(R / "ensemble" / f"{k}.json"))
    print(f"{k:22} sel {e['select']['map50_95']:.4f}  test {e['test']['map50_95']:.4f}  "
          f"P {e['test']['precision']:.3f} R {e['test']['recall']:.3f}")
