"""简单 GPU 队列：按顺序启动任务，每个任务等到空闲显存 ≥ need + MARGIN 才启动（任务可并行），日志 results/experiments/gpu_queue.log

python gpu_queue.py   任务列表见 JOBS（命令, 需要的显存 MiB, 日志名[, 前提：(目录, 至少文件数)]）
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / "results" / "experiments"
SYN = ROOT / "data" / "synthetic"
PY = os.environ.get("CRG_PY", sys.executable)          # 主环境的 Python（env/requirements-main.txt），默认为运行本脚本的 Python
MARGIN = 2000                                   # 启动后至少留给其他进程的显存（ControlNet 生成在空闲 < 1500 MiB 时会自停）

# 划分 A 全部检测器按 640 输入重跑（select 选 best.pt）；种子优先：先得到每个方法的第一个种子
METHODS = ["crg", "dpmpaste", "repaint", "copaint", "controlnet", "copypaste", "ecics", "crgwin"]
BPY = os.environ.get("BASELINES_PY", "python")         # 对比方法环境的 Python（env/requirements-baselines.txt）
DEFFILLER = (f"{BPY} baselines/deffiller_baseline.py gen > ../../runs/baselines/deffiller/gen.log 2>&1 && "
             + " && ".join(f"{PY} detect.py --config deffiller_1x --seed {s} --split A --proto img640" for s in (0, 1, 2)))
CRGOURS = [([PY, "detect.py", "--config", "crgours_1x", "--seed", str(s), "--split", "A", "--proto", "img640"], 6000,
            f"img640_crgours_s{s}", (SYN / "crgours_A" / "labels", 300)) for s in (0, 1, 2)] + \
          [([PY, "detect.py", "--config", "crgours_1x", "--seed", "2", "--split", "A", "--proto", "img640"], 6000,
            "img640_crgours_s2_redo")]                                  # 第一次因显存不足被自动减为 batch 16，结果作废后重跑
JOBS = [(["bash", "-c", DEFFILLER], 6000, "deffiller_A_full")] + CRGOURS + \
       [([PY, "detect.py", "--config", f"{m}_1x", "--seed", str(s), "--split", "A", "--proto", "img640"], 6000, f"img640_{m}_s{s}")
        for s in (0, 1, 2) for m in METHODS] + \
       [(["bash", str(ROOT / "runs" / "logs" / "crg_checks.sh"), c], 6000, f"crg_{c}") for c in ("check1", "check2")]

# 第 2 步（2026-11-19）：找一个扩增还有提升空间的评估设置。640 输入，3 个种子，种子优先
#  按板划分（测试板从未见过）：Original、CRG、DDIM、copy-paste
#  划分 A 只用每类 10 / 20 张真实图：Original、CRG、DDIM（copy-paste 会用到子集之外的真实缺陷，不公平，不做）
STEP2 = [("board", c) for c in ("original", "crg_1x", "dpmpaste_1x", "copypaste_1x")] + \
        [(sp, c) for sp in ("A10", "A20") for c in ("original", "crg_1x", "dpmpaste_1x")]
JOBS += [([PY, "detect.py", "--config", c, "--seed", str(s), "--split", sp, "--proto", "img640"], 6000, f"img640_{sp}_{c}_s{s}")
         for s in (0, 1, 2) for sp, c in STEP2]

# CRG 设置扫描（scan_configs.py）：每个设置 60 张（每类 10 张）→ 代理评分；参考：已有各方法的前 60 张；最后自动选前 3 个补全到 300 张
import scan_configs                                                       # noqa: E402
REF_DIRS = ["crg", "crgpaper", "ecics", "dpmpaste", "copypaste", "crgours", "crgwin", "repaint", "copaint", "controlnet", "deffiller"]
JOBS += [(["bash", "-c", " ; ".join(f"{PY} score_proxy.py --dir {d}_A --name ref_{d}" for d in REF_DIRS)], 1500, "scan_refs")]
JOBS += [(["bash", "-c", f"{PY} generate.py --method crg --scan {k} --start 0 --end 60 --jobs jobs_A && "
                         f"{PY} score_proxy.py --dir crgscan_{k}_A --name {k}"], 4500, f"scan_{k}") for k in scan_configs.SCAN]
JOBS += [([PY, "scan_stage2.py"], 13500, "scan_stage2")]

# 扫描第 3 步（2026-11-20）：前 3 个设置在每类 10 张真实图的设置上训练检测器（640，3 个种子）；等 300 张生成完再开始。
# 选设置时只看 select 半区的 mAP（train/results.csv），test 只用于最后报告
TOP3 = ["g4", "thr08", "temp0.5_thr0.8"]                                 # scan_stage2 于 Fri 10:09 选出
JOBS += [([PY, "detect.py", "--config", f"crgscan_{k}_1x", "--seed", str(s), "--split", "A10", "--proto", "img640"], 6000,
          f"img640_A10_crgscan_{k}_s{s}", (SYN / f"crgscan_{k}_A" / "labels", 300)) for s in (0, 1, 2) for k in TOP3]

# 2026-11-20：每类 10 张真实图的设置下补全 Table I 的其余方法（合成图已有，只训练检测器）。copy-paste 不做（会用到子集外的真实缺陷）
A10_REST = ["repaint", "copaint", "controlnet", "deffiller", "ecics", "crgpaper", "crgours", "crgwin"]
JOBS += [([PY, "detect.py", "--config", f"{m}_1x", "--seed", str(s), "--split", "A10", "--proto", "img640"], 6000,
          f"img640_A10_{m}_s{s}") for s in (0, 1, 2) for m in A10_REST]

# 2026-11-20 论文未说明的步骤的检查（划分 A，640，3 个种子；每个种子先跑完所有检查，便于先看到第一轮结果）
#  B  更难的测试裁剪（缺陷可被边缘截断）：Original、CRG、DDIM
#  B2 按 256 网格切出的测试裁剪（论文图中的线索）：Original、CRG、DDIM
#  A  在测试集上选 best epoch（论文只描述了训练集和"验证集"）：Original、CRG
#  C1 合成样本的标签框：整个 64 窗口 / 实际改动区域的外接框
#  C2 每张合成图 2 个缺陷：先生成第二个缺陷（105 张）并合并，再训练检测器
CHECKS = [("Aedge", "img640", c) for c in ("original", "crg_1x", "dpmpaste_1x")] + \
         [("Agrid", "img640", c) for c in ("original", "crg_1x", "dpmpaste_1x")] + \
         [("A", "valtest640", c) for c in ("original", "crg_1x")] + \
         [("A", "img640", c) for c in ("crgwbox_1x", "crgcbox_1x")] + \
         [("A", "img640", "crg_synonly")]                                 # 只用 300 张 CRG 合成图训练（两篇论文中合成数据单独训练就超过真实数据）
C2_GEN = (f"{PY} generate.py --method crg --jobs jobs_A_multi --tag multi_A --start 0 --end 300 && "
          f"{PY} make_multi_jobs.py merge")
for s in (0, 1, 2):
    JOBS += [([PY, "detect.py", "--config", c, "--seed", str(s), "--split", sp, "--proto", pr], 6000, f"chk_{sp}_{pr}_{c}_s{s}")
             for sp, pr, c in CHECKS]
    if s == 0:
        JOBS += [(["bash", "-c", C2_GEN], 4500, "chk_C2_gen")]
JOBS += [([PY, "detect.py", "--config", "crgmd_1x", "--seed", str(s), "--split", "A", "--proto", "img640"], 6000,
          f"chk_A_img640_crgmd_1x_s{s}", (SYN / "crgmd_A" / "labels", 300)) for s in (0, 1, 2)]

# 2026-11-20 DMGR 论文（同一课题组）给出的检测器设置：AdamW、lr0 0.01、余弦学习率（我们默认 auto → AdamW lr 0.001、线性）
JOBS += [([PY, "detect.py", "--config", c, "--seed", str(s), "--split", "A", "--proto", "dmgr640"], 6000, f"chk_A_dmgr640_{c}_s{s}")
         for s in (0, 1, 2) for c in ("original", "crg_1x", "dpmpaste_1x")]

# ===== 两天搜索（2026-11-24 起，docs/superpowers/plans/2026-11-20-two-day-improvement-search.md）=====
import det_variants                                                       # noqa: E402
# P2 准备：划分 A 的任务 j0300–j0899 的合成图（CRG 分两片并行；DDIM 很快）
JOBS += [([PY, "generate.py", "--method", "crg", "--jobs", "jobs_A", "--tag", "_A", "--start", "300", "--end", "900",
           "--shard", f"{k}/2"], 4500, f"s2_gen_crg_A_{k}") for k in (0, 1)]
JOBS += [([PY, "generate.py", "--method", "dpmpaste", "--jobs", "jobs_A", "--tag", "_A", "--start", "300", "--end", "900"], 4500, "s2_gen_dpmpaste_A")]
# P1：检测器设置变体 × {Original, CRG 1×} × 种子 0、1（按 select 选）
P1 = ["base", "flipud", "img800", "img960", "ep300", "nomosaic", "close50", "yolo11s"]
JOBS += [([PY, "detect.py", "--config", c, "--seed", str(s), "--split", "A", "--variant", v], det_variants.mem(v), f"s1_{v}_{c}_s{s}")
         for s in (0, 1) for v in P1 for c in ("original", "crg_1x")]

# P1 结论（Tue 15:30）：没有变体在两种配置上都比 base 高 ≥ 0.005（select）；最接近的 flipud、img960 补种子 2 再定
JOBS += [([PY, "detect.py", "--config", c, "--seed", "2", "--split", "A", "--variant", v], det_variants.mem(v), f"s1_{v}_{c}_s2")
         for v in ("base", "flipud", "img960") for c in ("original", "crg_1x")]
# P2：在 base 检测器上比较合成数据的数量与混合（按 select 选），种子 0、1
P2 = ["crg_2x", "crg_3x", "dpmpaste_1x", "dpmpaste_2x", "dpmpaste_3x", "copypaste_1x", "copypaste_2x", "copypaste_3x", "crg+copypaste_1x"]
JOBS += [([PY, "detect.py", "--config", c, "--seed", str(s), "--split", "A", "--variant", "base"], 6000, f"s2_base_{c}_s{s}")
         for s in (0, 1) for c in P2]

# 补充（15:30）：CRG 与 copy-paste 各 600 张的混合
JOBS += [([PY, "detect.py", "--config", "crg+copypaste_2x", "--seed", str(s), "--split", "A", "--variant", "base"], 6000,
          f"s2_base_crg+copypaste_2x_s{s}") for s in (0, 1)]

# P3（2026-11-24 用户要求）：框精度相关的检测器变体，Original 与 CRG 1×，种子 0、1，按 select 选
JOBS += [([PY, "detect.py", "--config", c, "--seed", str(s), "--split", "A", "--variant", v], det_variants.mem(v), f"s3_{v}_{c}_s{s}")
         for s in (0, 1) for v in ("box10", "box15", "dfl3", "p2") for c in ("original", "crg_1x")]
# 推理增强（不重训）：base 检测器的单模型、TTA、3 种子集成、集成 + TTA；Original 与 CRG 1×
_R = str(ROOT / "results" / "experiments" / "detection_A_vbase")
for c in ("original", "crg_1x"):
    seeds = [f"{_R}/{c}/seed{s}" for s in (0, 1, 2)]
    JOBS += [([PY, "eval_ensemble.py", "--runs", seeds[0], "--name", f"{c}_single_s0"], 2000, f"s3_ens_{c}_single", (Path(seeds[2]) / "train" / "weights", 1)),
             ([PY, "eval_ensemble.py", "--runs", seeds[0], "--name", f"{c}_tta_s0", "--tta"], 2000, f"s3_ens_{c}_tta", (Path(seeds[2]) / "train" / "weights", 1)),
             ([PY, "eval_ensemble.py", "--runs", *seeds, "--name", f"{c}_ens3"], 2500, f"s3_ens_{c}_ens3", (Path(seeds[2]) / "train" / "weights", 1)),
             ([PY, "eval_ensemble.py", "--runs", *seeds, "--name", f"{c}_ens3_tta", "--tta"], 2500, f"s3_ens_{c}_ens3_tta", (Path(seeds[2]) / "train" / "weights", 1))]

# P2/P3 结论（Tue 18:50）：copypaste_2x / 3x 在 select 上比 Original 高 ≥ 0.005，补种子 2 确认；3 种子集成在 select 上 +0.02，
# P4：base 的 Original、CRG 1× 补种子 3、4（共 5 个），再看 5 模型集成与另一组 3 种子（2、3、4）的集成是否同样提高
JOBS += [([PY, "detect.py", "--config", c, "--seed", "2", "--split", "A", "--variant", "base"], 6000, f"s2_base_{c}_s2")
         for c in ("copypaste_2x", "copypaste_3x")]
JOBS += [([PY, "detect.py", "--config", c, "--seed", str(s), "--split", "A", "--variant", "base"], 6000, f"s4_base_{c}_s{s}")
         for s in (3, 4) for c in ("original", "crg_1x")]
for c, seeds in (("original", range(5)), ("crg_1x", range(5)), ("copypaste_2x", range(3))):
    d = [f"{_R}/{c}/seed{s}" for s in seeds]
    need = [Path(x) / "metrics.json" for x in d]
    if len(d) == 5:
        JOBS += [([PY, "eval_ensemble.py", "--runs", *d, "--name", f"{c}_ens5"], 3000, f"s4_ens_{c}_ens5", need),
                 ([PY, "eval_ensemble.py", "--runs", *d[2:], "--name", f"{c}_ens3_s234"], 2500, f"s4_ens_{c}_ens3_s234", need)]
    else:
        JOBS += [([PY, "eval_ensemble.py", "--runs", *d, "--name", f"{c}_ens3"], 2500, f"s4_ens_{c}_ens3", need)]

# P4 结论（Tue 20:50）：copypaste_2x 在 select 上比 Original 高 ≥ 0.005（3 种子），其 3 种子集成也最高；最终设置 = copypaste_2x + 集成，
# 按计划补到 5 个种子，再做 5 模型集成
JOBS += [([PY, "detect.py", "--config", "copypaste_2x", "--seed", str(s), "--split", "A", "--variant", "base"], 6000,
          f"s4_base_copypaste_2x_s{s}") for s in (3, 4)]
_d = [f"{_R}/copypaste_2x/seed{s}" for s in range(5)]
JOBS += [([PY, "eval_ensemble.py", "--runs", *_d, "--name", "copypaste_2x_ens5"], 3000, "s4_ens_copypaste_2x_ens5",
          [Path(x) / "metrics.json" for x in _d])]

# 泄漏诊断（2026-11-24 用户要求；只解释论文结果，不计入任务书指标）：用测试图像自身的背景与缺陷位置生成（build_leak_jobs.py）
JOBS += [([PY, "generate.py", "--method", "crg", "--jobs", "jobs_Aleak", "--tag", "leak_A", "--start", "0", "--end", "300",
           "--shard", f"{k}/2"], 4500, f"leak_gen_crg_{k}") for k in (0, 1)]
JOBS += [([PY, "detect.py", "--config", c, "--seed", str(s), "--split", "A", "--variant", "base"], 6000, f"leak_base_{c}_s{s}",
          (SYN / "crgleak_A" / "labels", 300)) for s in (0, 1, 2) for c in ("crgleak_1x", "crgleak_synonly")]

# 缺陷先验模型重训（2026-11-24 用户要求放在队列最后）：不二次缩放的 777 个划分 A 训练照片局部块，200000 次迭代
# 20:55 用户要求暂缓：前提改为放行目录，放行时 mkdir -p runs/release_maxcov && touch runs/release_maxcov/go
JOBS += [([PY, "train_defect.py"], 8000, "retrain_defect64_maxcov", (ROOT / "runs" / "release_maxcov", 1))]
# 重训之后：按 KID 选权重（参考集 = 划分 A 选择照片的缺陷块），用选中的 64 模型 + 作者 256 模型生成 300 张，
# 在划分 A（base 检测器，对照 detection_A_vbase 的 original / crg_1x）和每类 10 张（img640，对照表 2-6）上检测
_RUN = ROOT / "runs" / "defect_64_maxcov_trainA"
JOBS += [([PY, "select_defect_ckpt.py"], 6000, "maxcov_select_ckpt", (_RUN / "done", 1))]
# Thu 00:13 第一次在 AD 训练同时运行时显存不足（采样 200 张一批 + KID，峰值约 9 GB），以更大的显存需求重新排队
JOBS += [([PY, "select_defect_ckpt.py"], 11000, "maxcov_select_ckpt_retry", (_RUN / "done", 1))]
JOBS += [(["bash", "-c", f"{PY} generate.py --method crg --jobs jobs_A --tag maxcov_A --start 0 --end 300 --shard {k}/2 "
           f"--ckpt-64 $(cat {_RUN}/selected/best.txt)"], 4500, f"maxcov_gen_crg_{k}", (_RUN / "selected", 1)) for k in (0, 1)]
JOBS += [([PY, "detect.py", "--config", "crgmaxcov_1x", "--seed", str(s), "--split", sp, *a], 6000, f"maxcov_{sp}_crgmaxcov_1x_s{s}",
          (SYN / "crgmaxcov_A" / "labels", 300))
         for s in (0, 1, 2) for sp, a in (("A", ["--variant", "base"]), ("A10", ["--proto", "img640"]))]

# ===== 2026-11-25 毕业论文补充实验（用户要求全部运行） =====
import scan_configs as _sc                                                 # noqa: E402
_M4 = ("repaint", "copaint", "controlnet", "deffiller")
# T6 AnomalyDiffusion 冒烟测试（3 类各 60 步 + 生成 3 张，输出在 /tmp/ad_smoke；看过速度、显存和图像后再加完整训练与生成）
_ADS = "cd ../.. && export AD_CKPT=/tmp/ad_smoke/ckpt AD_OUT=/tmp/ad_smoke/out MAX_MIB=8000 && "
JOBS += [(["bash", "-c", _ADS + f"AD_MAX_STEPS=60 {BPY} code/experiments/baselines/anomalydiffusion_baseline.py train missinghole mousebite "
           f"opencircuit > /tmp/ad_smoke/train.log 2>&1 && {BPY} code/experiments/baselines/anomalydiffusion_baseline.py gen 3 "
           f"> /tmp/ad_smoke/gen.log 2>&1"], 7500, "t6_ad_smoke")]
# 12:59 第一次冒烟测试在 batch 4 的第一步超出 8 GB 上限；改为微批 2 × 累积 2 后重测（等效 batch 不变）
JOBS += [(["bash", "-c", _ADS + f"AD_MAX_STEPS=60 {BPY} code/experiments/baselines/anomalydiffusion_baseline.py train missinghole mousebite "
           f"opencircuit > /tmp/ad_smoke/train.log 2>&1 && {BPY} code/experiments/baselines/anomalydiffusion_baseline.py gen 3 "
           f"> /tmp/ad_smoke/gen.log 2>&1"], 7500, "t6b_ad_smoke")]
# 13:16 t6b：显存满足，但损失为 NaN（空间编码器输出超出 fp16 范围）；改为编码器 fp32、BERT bf16 后重测，每类 200 步看输出幅值趋势
JOBS += [(["bash", "-c", _ADS + f"AD_MAX_STEPS=200 {BPY} code/experiments/baselines/anomalydiffusion_baseline.py train missinghole mousebite "
           f"opencircuit > /tmp/ad_smoke/train.log 2>&1 && {BPY} code/experiments/baselines/anomalydiffusion_baseline.py gen 3 "
           f"> /tmp/ad_smoke/gen.log 2>&1"], 7500, "t6c_ad_smoke")]
# 13:56 t6c：损失有限，但同一进程训练第 2 类时显存不足（第 1 类未释放）→ 每类单独一个进程；
# 先用 200 步的缺孔模型生成 1 张（j0000 为缺孔）检查图像，确认后建 runs/release_ad 放行完整训练
JOBS += [(["bash", "-c", _ADS + f"{BPY} code/experiments/baselines/anomalydiffusion_baseline.py gen 1 > /tmp/ad_smoke/gen.log 2>&1"],
          7500, "t6d_ad_gen1")]
_AD = "cd ../.. && export MAX_MIB=8000 && "
_ADC = ROOT / "runs" / "baselines" / "anomalydiffusion" / "ckpt"
JOBS += [(["bash", "-c", _AD + f"{BPY} code/experiments/baselines/anomalydiffusion_baseline.py train {c} "
           f"> runs/baselines/anomalydiffusion/train_{c}.log 2>&1"], 8500, f"t6_ad_train_{c}", (ROOT / "runs" / "release_ad", 1))
         for c in ("missinghole", "mousebite", "opencircuit", "short", "spur", "spuriouscopper")]
JOBS += [(["bash", "-c", _AD + f"{BPY} code/experiments/baselines/anomalydiffusion_baseline.py gen 900 "
           f"> runs/baselines/anomalydiffusion/gen.log 2>&1"], 7000, "t6_ad_gen900",
          [_ADC / c / "done" for c in ("missinghole", "mousebite", "opencircuit", "short", "spur", "spuriouscopper")])]
_ADL = SYN / "anomalydiffusion_A" / "labels"
JOBS += [([PY, "detect.py", "--config", "anomalydiffusion_1x", "--seed", str(s), "--split", sp, "--proto", "img640"], 6000,
          f"t6_{sp}_ad_1x_s{s}", (_ADL, 300)) for s in (0, 1, 2) for sp in ("A", "A10", "A20")]
JOBS += [([PY, "detect.py", "--config", f"anomalydiffusion_{k}x", "--seed", str(s), "--split", "A", "--variant", "base"], 6000,
          f"t6_base_ad_{k}x_s{s}", (_ADL, 900)) for s in (0, 1, 2) for k in (2, 3)]
JOBS += [([PY, "gen_quality.py", "anomalydiffusion_A"], 3000, "t6_ad_quality", (_ADL, 300))]
# T1 每类 20 张：其余对比方法（原始数据、CRG、DDIM 已有），与 A20 已有结果同为 img640
JOBS += [([PY, "detect.py", "--config", f"{m}_1x", "--seed", str(s), "--split", "A20", "--proto", "img640"], 6000, f"t1_A20_{m}_s{s}")
         for s in (0, 1, 2) for m in (*_M4, "ecics")]
# T5 扩增倍率表补到 3 个种子（base 检测器；copypaste_2x / 3x 已有）
JOBS += [([PY, "detect.py", "--config", c, "--seed", "2", "--split", "A", "--variant", "base"], 6000, f"s2_base_{c}_s2")
         for c in ("crg_2x", "crg_3x", "dpmpaste_1x", "dpmpaste_2x", "dpmpaste_3x", "copypaste_1x")]
# T2 生成质量（已有的 1× 目录；参考集为选择集）
JOBS += [([PY, "gen_quality.py", "repaint_A", "copaint_A", "controlnet_A", "deffiller_A", "ecics_A", "copypaste_A", "crgwin_A",
           "crgpaper_A", "crgours_A"], 3000, "t2_gen_quality")]
# T3 消融：每个设置 300 张（两片并行），检测：划分 A base × 3 种子、每类 10 张 img640 × 3 种子；再算生成质量
for _k in _sc.ABLATION:
    _d = SYN / f"crgscan_{_k}_A" / "labels"
    JOBS += [([PY, "generate.py", "--method", "crg", "--scan", _k, "--jobs", "jobs_A", "--start", "0", "--end", "300",
               "--shard", f"{h}/2"], 4500, f"t3_gen_{_k}_{h}") for h in (0, 1)]
    if _k == "abl_nobg":                  # Thu 02:33 shard 0 在 60/150 张时显存不足（与选权重、另 3 个生成同时运行）；可续跑，提高显存需求
        JOBS += [([PY, "generate.py", "--method", "crg", "--scan", _k, "--jobs", "jobs_A", "--start", "0", "--end", "300",
                   "--shard", "0/2"], 6500, f"t3_gen_{_k}_0_resume")]
    JOBS += [([PY, "detect.py", "--config", f"crgscan_{_k}_1x", "--seed", str(s), "--split", sp, *a], 6000, f"t3_{sp}_{_k}_s{s}", (_d, 300))
             for s in (0, 1, 2) for sp, a in (("A", ["--variant", "base"]), ("A10", ["--proto", "img640"]))]
    JOBS += [([PY, "gen_quality.py", f"crgscan_{_k}_A"], 3000, f"t3_quality_{_k}", (_d, 300))]
# T4 扩增倍率：RePaint、CoPaint、ControlNet、DefFiller 补生成 j0300–j0899，再检测 2×、3×（1× 用已有的 img640 结果，设置与 base 相同）
JOBS += [([PY, "generate.py", "--method", "repaint", "--jobs", "jobs_A", "--tag", "_A", "--start", "300", "--end", "900"], 4500, "t4_gen_repaint")]
JOBS += [([PY, "generate.py", "--method", "copaint", "--jobs", "jobs_A", "--tag", "_A", "--start", "300", "--end", "900",
           "--shard", f"{h}/3"], 4500, f"t4_gen_copaint_{h}") for h in (0, 1, 2)]
JOBS += [(["bash", "-c", f"CN_N=900 CN_MIN_FREE=0 {BPY} baselines/controlnet_baseline.py gen A"], 6000, "t4_gen_controlnet")]
JOBS += [(["bash", "-c", f"DF_N=900 {BPY} baselines/deffiller_baseline.py gen"], 6000, "t4_gen_deffiller")]
JOBS += [([PY, "detect.py", "--config", f"{m}_{k}x", "--seed", str(s), "--split", "A", "--variant", "base"], 6000, f"t4_base_{m}_{k}x_s{s}",
          (SYN / f"{m}_A" / "labels", 900)) for s in (0, 1, 2) for k in (2, 3) for m in _M4]
# 重训模型的生成质量
JOBS += [([PY, "gen_quality.py", "crgmaxcov_A"], 3000, "t2_quality_crgmaxcov", (SYN / "crgmaxcov_A" / "labels", 300))]


# 2026-11-25 15:45 用户要求优先：输入尺寸表补齐（只用生成数据训练的检测器在 256/800/960 输入下各 3 个种子，640 已有；
# 800 输入补种子 2）。这些任务全部结束前不启动其他新任务（已在运行的任务不受影响）
_P = [([PY, "detect.py", "--config", "crg_synonly", "--seed", str(s), "--split", "A"], 6000, f"p_A256_crg_synonly_s{s}") for s in (0, 1, 2)]
_P += [([PY, "detect.py", "--config", c, "--seed", "2", "--split", "A", "--variant", "img800"], det_variants.mem("img800"),
        f"s1_img800_{c}_s2") for c in ("original", "crg_1x")]
_P += [([PY, "detect.py", "--config", "crg_synonly", "--seed", str(s), "--split", "A", "--variant", v], det_variants.mem(v),
        f"p_{v}_crg_synonly_s{s}") for v in ("img800", "img960") for s in (0, 1, 2)]
# 2026-11-25 用户要求：数据量趋势表补每类 30 / 40 / 60 张，全部方法 × 3 个种子（输入 640，与每类 10 / 20 / 50 张的结果相同设置），
# 排在上面的优先任务之后、其他任务之前
# 无缺陷图来源检查（用户要求，排在趋势实验之前）：与 detection_A_vbase 的 original / crg_1x 种子 0–2 对照
_P += [([PY, "detect.py", "--config", c, "--seed", str(s), "--split", "Aclean", "--variant", "base"], 6000, f"cf_{c}_s{s}")
       for s in (0, 1, 2) for c in ("original", "crg_1x")]
_TREND = ["original", "crg_1x", "ecics_1x", "dpmpaste_1x", "repaint_1x", "copaint_1x", "controlnet_1x", "deffiller_1x"]
_P += [([PY, "detect.py", "--config", c, "--seed", str(s), "--split", sp, "--proto", "img640"], 6000, f"tr_{sp}_{c}_s{s}")
       for s in (0, 1, 2) for sp in ("A30", "A40", "A60") for c in _TREND]
JOBS = _P + JOBS
PRIORITY = {j[2] for j in _P}
# t3_gen_abl_nodef_0 在 15:44 刚启动即为优先任务让出显存而停止（generate.py 跳过已生成的图像，可续跑），以新名字重新排队
JOBS += [([PY, "generate.py", "--method", "crg", "--scan", "abl_nodef", "--jobs", "jobs_A", "--start", "0", "--end", "300",
           "--shard", "0/2"], 4500, "t3_gen_abl_nodef_0_resume")]


# ===== 2026-11-25 P6：只改检测部分的改进（用户选择排在队列最后；只在其他任务都在等前提时才启动，填空闲）=====
_R6 = ROOT / "results" / "experiments"
# 思路 1 两阶段：第一阶段只用 900 张 CRG 合成图训练（select 选 best.pt），第二阶段从该权重出发只用真实数据训练（每类 50 / 10 / 20 张）
_L = [([PY, "detect.py", "--config", "crg_synall", "--seed", str(s), "--split", "A", "--variant", "base"], 6000, f"p6_pre_crg900_s{s}")
      for s in (0, 1, 2)]
_L += [([PY, "detect.py", "--config", "original", "--seed", str(s), "--split", sp, "--variant", "ft",
         "--init", str(_R6 / "detection_A_vbase" / "crg_synall" / f"seed{s}" / "train" / "weights" / "best.pt")], 6000,
        f"p6_ft_{sp}_s{s}", [_R6 / "detection_A_vbase" / "crg_synall" / f"seed{s}" / "metrics.json"])
       for s in (0, 1, 2) for sp in ("A", "A10", "A20")]
# 思路 2 修正合成图标签框（relabel_synthetic.py 已生成 data/synthetic/crgrelab_A，142/300 张换框）
_L += [([PY, "detect.py", "--config", "crgrelab_1x", "--seed", str(s), "--split", sp, "--variant", "base"], 6000, f"p6_relab_{sp}_s{s}")
       for s in (0, 1, 2) for sp in ("A", "A10")]
# 思路 3 90° 旋转增强：原始数据与原始数据 + CRG 成对比较，每类 50 / 10 张
_L += [([PY, "detect.py", "--config", c, "--seed", str(s), "--split", sp, "--variant", "rot90"], 6000, f"p6_rot90_{sp}_{c}_s{s}")
       for s in (0, 1, 2) for sp in ("A", "A10") for c in ("original", "crg_1x")]
# 思路 6 混合集成：只用真实数据的模型与原始数据 + CRG 的模型一起做加权框融合（不训练）
_VB = _R6 / "detection_A_vbase"
for _k, _seeds in (("mix6", range(3)), ("mix10", range(5))):
    _runs = [str(_VB / c / f"seed{s}") for c in ("original", "crg_1x") for s in _seeds]
    _L += [([PY, "eval_ensemble.py", "--runs", *_runs, "--name", _k], 3000, f"p6_ens_{_k}", [Path(r) / "metrics.json" for r in _runs])]
# 18:55 无缺陷图来源检查完成：换图后单个数值变化最多约 0.009，方向不一致（在种子波动范围内），但按计划规则 4，
# P6 在划分 A 上的真实数据训练改用修正后的训练集 Aclean，对照为 detection_Aclean_vbase 的 original / crg_1x 种子 0–2（已有）；
# 第一阶段（只用合成图）不用真实训练图，不变；A10 / A20 与已有基线保持一致，不变
for _j in _L:
    if _j[2].startswith(("p6_ft_A_", "p6_relab_A_", "p6_rot90_A_")):
        _k = _j[0].index("--split"); assert _j[0][_k + 1] == "A"; _j[0][_k + 1] = "Aclean"
_VC = _R6 / "detection_Aclean_vbase"
_cl = {c: [str(_VC / c / f"seed{s}") for s in range(3)] for c in ("original", "crg_1x")}
_L += [([PY, "eval_ensemble.py", "--runs", *_cl["original"], "--name", "clean_original_ens3"], 2500, "p6_ens_clean_orig3"),
       ([PY, "eval_ensemble.py", "--runs", *_cl["crg_1x"], "--name", "clean_crg_1x_ens3"], 2500, "p6_ens_clean_crg3"),
       ([PY, "eval_ensemble.py", "--runs", *_cl["original"], *_cl["crg_1x"], "--name", "clean_mix6"], 3000, "p6_ens_clean_mix6")]
JOBS += _L
LATE = {j[2] for j in _L}

# ===== 2026-11-26 P7：P6 通过的想法组合（用户同意）=====
# 旋转模型的集成（不训练，对照 clean_*_ens3 / clean_mix6，模型数相同）
_VR = _R6 / "detection_Aclean_vrot90"
_rr = {c: [str(_VR / c / f"seed{s}") for s in range(3)] for c in ("original", "crg_1x")}
_C = [([PY, "eval_ensemble.py", "--runs", *_rr["original"], "--name", "rot90_original_ens3"], 2500, "p7_ens_rot90_orig3"),
      ([PY, "eval_ensemble.py", "--runs", *_rr["crg_1x"], "--name", "rot90_crg_1x_ens3"], 2500, "p7_ens_rot90_crg3"),
      ([PY, "eval_ensemble.py", "--runs", *_rr["original"], *_rr["crg_1x"], "--name", "rot90_mix6"], 3000, "p7_ens_rot90_mix6")]
# 两阶段 + 旋转：第一阶段权重沿用 p6_pre_crg900（未加旋转），第二阶段加旋转，每类 10 / 20 张；对照为单独的旋转与单独的两阶段
_C += [([PY, "detect.py", "--config", "original", "--seed", str(s), "--split", sp, "--variant", "rot90+ft",
         "--init", str(_R6 / "detection_A_vbase" / "crg_synall" / f"seed{s}" / "train" / "weights" / "best.pt")], 6000,
        f"p7_rot90ft_{sp}_s{s}") for s in (0, 1, 2) for sp in ("A10", "A20")]
# 每类 20 张的旋转（补全 10 / 20 / 50 三个数据量）
_C += [([PY, "detect.py", "--config", c, "--seed", str(s), "--split", "A20", "--variant", "rot90"], 6000, f"p7_rot90_A20_{c}_s{s}")
       for s in (0, 1, 2) for c in ("original", "crg_1x")]
JOBS += _C


def free_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(out.split()[0])


def log(msg):
    with open(RES / "gpu_queue.log", "a") as f:
        f.write(time.strftime("%a %H:%M:%S ") + msg + "\n")


def reap(running):
    for p, name in [r for r in running if r[0].poll() is not None]:
        log(f"done {name} (exit {p.returncode})"); running.remove((p, name))


def started():
    """日志里已经启动过的任务（重启队列时跳过，避免 detect.py 删除正在运行任务的输出目录）"""
    f = RES / "gpu_queue.log"
    return {l.split()[3] for l in f.read_text().splitlines() if l.split()[2:3] == ["start"]} if f.exists() else set()


def ready(job, pending=(), running=()):
    """前提为 "last" 的任务：没有任务在运行、其余待启动任务都在等前提时才启动（后续任务可以依赖它的输出）"""
    pre = job[3:]
    if pre == ("last",):
        return not running and not any(ready(j) for j in pending if j is not job and j[3:] != ("last",))
    if pre and isinstance(pre[0], list):                              # 一组 metrics.json 都已写出
        return all(f.exists() for f in pre[0])
    return not pre or len(list(pre[0][0].glob("*"))) >= pre[0][1]


def is_scan(name):
    return name.startswith("scan_") and name not in ("scan_refs", "scan_stage2")


def main():
    """按顺序取第一个"显存够且前提满足"的任务启动（先到先得，但小任务可以填进大任务等不到的空隙）。
    扫描生成任务同时最多 2 个（还有其他任务在等时），以免拖慢第 2 步的检测器"""
    time.sleep(180)                       # 重启队列时，刚启动的任务可能还没占满显存；先等它们稳定再判断空闲显存
    skip = started()
    pending, running = [j for j in JOBS if j[2] not in skip], []
    while pending:
        reap(running)
        free = free_mib()
        n_scan = sum(is_scan(n) for _, n in running)
        cap = 2 if any(not is_scan(j[2]) and not j[2].startswith("scan_") for j in pending) else 4   # scan_stage2 不算"其他任务"
        pri = [j for j in pending if j[2] in PRIORITY]
        cand = pri if pri or any(n in PRIORITY for _, n in running) else pending       # 优先任务未全部结束时只启动优先任务
        if any(j[2] not in LATE and ready(j, pending, running) for j in cand):   # P6 只在其他任务都在等前提时启动
            cand = [j for j in cand if j[2] not in LATE]
        job = next((j for j in cand if free >= j[1] + MARGIN and ready(j, pending, running) and not (is_scan(j[2]) and n_scan >= cap)), None)
        if job is None:
            time.sleep(60); continue
        cmd, need, name = job[:3]
        p = subprocess.Popen(cmd, cwd=HERE, stdout=open(RES / "proto_logs" / f"{name}.log", "w"), stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, env={**os.environ, "MPLBACKEND": "Agg"})
        running.append((p, name)); pending.remove(job); log(f"start {name} (pid {p.pid}, free {free} MiB)")
        time.sleep(180)                                               # 等显存占用稳定后再判断下一个
    while running:
        reap(running); time.sleep(60)
    log("queue finished")


if __name__ == "__main__":
    main()
