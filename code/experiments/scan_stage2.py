"""扫描第 2 阶段：等 scan_configs 的全部设置都有代理分数后，选出前 3 个，补全到 300 张（与其他方法同样的任务），
再在 300 张上重新打分（60 张的代理分数误差约 ±0.07，300 张约 ±0.03）。不训练检测器：用哪个评估设置等第 2 步的结果出来再定。

选择规则（生成前定好）：窗口外背景变化不超过发布设置（ref_crg）的 1.5 倍；按检出率从高到低，相同时背景变化小的优先。
结果：results/experiments/scan/summary.csv（全部设置）、stage2.json（选中的设置与 300 张的分数）
"""
import csv
import json
import subprocess
import time

import scan_configs
from common import RES

PY = "/home/CNS2026014739/miniconda3/envs/crg-defect/bin/python"
P = RES / "scan" / "proxy"
LOG = RES.parents[1] / "runs" / "logs" / "claude_checks.log"
REFS = {"ref_crg": "crg_A", "ref_crgpaper": "crgpaper_A", "ref_ecics": "ecics_A", "ref_dpmpaste": "dpmpaste_A"}


def load(name, n):
    f = P / f"{name}_n{n}.json"
    return json.load(open(f)) if f.exists() else None


def main():
    names = list(scan_configs.SCAN)
    while any(load(k, 60) is None for k in names + ["ref_crg"]):
        time.sleep(120)
    rows = [load(k, 60) for k in names] + [load(k, 60) for k in sorted(p.stem[:-4] for p in P.glob("ref_*_n60.json"))]
    with open(RES / "scan" / "summary.csv", "w", newline="") as f:
        cols = ["name", "dir", "n", "det_rate", "mean_iou", "bg_change", "win_change"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        w.writerows(sorted(rows, key=lambda r: -r["det_rate"]))
    limit = 1.5 * load("ref_crg", 60)["bg_change"]
    ok = sorted((r for r in rows[:len(names)] if r["bg_change"] <= limit), key=lambda r: (-r["det_rate"], r["bg_change"]))
    top = [r["name"] for r in ok[:3]]
    with open(LOG, "a") as f:
        f.write(time.strftime("%a %H:%M ") + f"scan stage 1 done ({len(names)} settings); top 3 by proxy: {top}; generating 300 each\n")
    procs = [subprocess.Popen([PY, "generate.py", "--method", "crg", "--scan", k, "--start", "0", "--end", "300", "--jobs", "jobs_A"],
                              stdout=open(RES / "proto_logs" / f"scan2_gen_{k}.log", "w"), stderr=subprocess.STDOUT) for k in top]
    codes = [p.wait() for p in procs]
    for k, c in zip(top, codes):
        if c == 0:
            subprocess.run([PY, "score_proxy.py", "--dir", f"crgscan_{k}_A", "--name", k, "--n", "300"])
    for k, d in REFS.items():
        subprocess.run([PY, "score_proxy.py", "--dir", d, "--name", k, "--n", "300"])
    out = {"top": top, "exit_codes": codes, "n300": {k: load(k, 300) for k in top + list(REFS)}}
    json.dump(out, open(RES / "scan" / "stage2.json", "w"), indent=1)
    with open(LOG, "a") as f:
        f.write(time.strftime("%a %H:%M ") + "scan stage 2 done: " + ", ".join(
            f"{k} {v['det_rate']:.3f}" for k, v in out["n300"].items() if v) + "\n")


if __name__ == "__main__":
    main()
