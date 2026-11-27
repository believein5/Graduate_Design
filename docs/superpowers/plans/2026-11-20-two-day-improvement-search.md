# Two-day search to reach the targets (2026-11-20 → 2026-11-22)

Goal: on split A (photo split, 50 real crops per class, real test set of 300 crops), a detector trained on real + synthetic data reaches **precision > 0.90, recall > 0.90 and mAP@50–95 > 0.50**, and, if possible, beats the same detector without synthetic data (Original) by more than noise (about 0.01).

Starting point (input 640, 3 seeds): Original P 0.956, R 0.936, mAP 0.497; CRG 1× P 0.969, R 0.921, mAP 0.494.

## Rules fixed before any result is seen

1. **No test data in training, ever.** Test crops come from test photos only; no test image or test photo is used for training or for any choice.
2. **Choices are made on the select set** (300 real crops from photos disjoint from train and test). Score of a run = best select mAP@50–95 over epochs (`train/results.csv`). A setting's score = mean over its seeds.
3. **The test set is read only to report** settings that have been chosen, and the final setting, with 3 seeds (5 for the final table). All settings that were tried are listed, including failures.
4. A setting counts as better than another only if its select mean is higher by ≥ 0.005 and the test result does not contradict it by more than noise.
5. **Leakage check last and labelled:** a crop-level split (train and test crops from the same photos, as the paper may have done) is run only to measure how much such a leak inflates scores. Its numbers are never reported as reaching the targets.

## Phases

| Phase | What | Runs | Decision |
|---|---|---|---|
| P0 (queued 20 Nov) | box labels C1, two defects C2, synthetic only, DMGR detector settings | 21 | feeds P2 |
| P1 detector protocol | variants on Original and CRG 1×, seeds 0–1: `flipud`, `img800`, `img960`, `ep300`, `nomosaic`, `close50`, `yolo11s`; plus DMGR settings from P0 | 28 | keep variants ≥ 0.005 above base on select for both methods; combine the best two; confirm with seed 2 |
| P2 synthetic data | at the best P1 detector: CRG 2× and 3× (jobs j0300–j0899 generated for split A), copy-paste 2×/3×, CRG + copy-paste mix, change-box labels if C1 helps, two-defect images if C2 helps, weak-class weighting | ~30 + generation | best by select |
| P3 test-time augmentation | re-evaluate kept `best.pt` with `augment=True` on select, then test | eval only | keep if select improves |
| P4 final | best setting and Original with the same detector, 5 seeds, test | 10 | report against targets |
| P5 diagnostic | crop-level split, Original and CRG | 6 | labelled, not a result |

Low-data track (thesis contribution): the best P1/P2 detector is also run at 10 real crops per class, Original vs CRG vs DDIM, 3 seeds.

## Log

(appended at each check-in: time, what finished, what was decided, what was queued)

### 2026-11-24 10:50 · P0 results, P1 started

Paused 20 Nov 21:00 – 24 Nov 10:30 (user stopped tool use; resumed on "go on").

P0 (split A, input 640, 3 seeds, test mAP@50–95): box labels 64-window 0.493 / changed-area 0.495; two defects per image 0.490; CRG synthetic only 0.335 (recall 0.644); best epoch picked on test (not allowed for results) Original 0.505, CRG 0.506. **DMGR detector settings** (AdamW, lr0 0.01, cosine): Original 0.431 (recall 0.800), CRG 0.469, DDIM 0.453 — the higher learning rate weakens the baseline and synthetic data compensates; this reproduces the paper's ordering, not its levels.

Targets already met without synthetic-data gains: copy-paste 1× P 0.949, R 0.936, mAP 0.501 (3 seeds); CRG 1× P 0.969, R 0.921, mAP 0.494 (mAP short by 0.006).

Queued: CRG and DDIM jobs j0300–j0899 on split A (for 2×/3×); copy-paste 900 done. P1: 8 detector variants × {Original, CRG 1×} × seeds 0–1, scored on select.

### 2026-11-24 15:30 · P1 done (detector variants, seeds 0–1)

| Variant | Select mAP: Original | Select mAP: CRG 1× | Test mAP: Original / CRG |
|---|---|---|---|
| base | 0.5149 | 0.5199 | 0.497 / 0.494 |
| flipud | 0.5188 | 0.5190 | 0.517 / 0.503 |
| img800 | 0.5202 | 0.5190 | 0.487 / 0.491 |
| img960 | 0.5183 | 0.5247 | 0.485 / 0.495 |
| ep300 | 0.5141 | 0.5173 | 0.484 / 0.492 |
| nomosaic | 0.5126 | 0.5134 | 0.496 / 0.475 |
| close50 | 0.5151 | 0.5157 | 0.499 / 0.491 |
| yolo11s | 0.5068 | 0.5202 | 0.503 / 0.502 |

Decision (rule 4): no variant is ≥ 0.005 above base on select for both methods; seed-to-seed select spread is about 0.003–0.008, so all variants are within noise. Base detector kept. flipud and img960 (closest) get seed 2 to confirm. Test numbers are shown for completeness only and were not used: flipud Original 0.517 would be the best test score so far, but select does not support it.

Queued: P1 seed 2 for base/flipud/img960; P2 on the base detector: CRG 2×/3×, DDIM 1×/2×/3×, copy-paste 1×/2×/3×, CRG + copy-paste 1× each, seeds 0–1.
 (+ CRG + copy-paste 2× each, added 15:31).

### 2026-11-24 16:50 · P3 added (box precision, inference-time)

mAP@50 is about 0.96 for every method, so mAP@50–95 is limited by box precision at strict IoU. Added, all chosen on select:
- detector variants `box10`, `box15` (box-loss weight, default 7.5), `dfl3` (DFL weight, default 1.5), `p2` (YOLO11 with an extra stride-4 head, `yolo11-p2.yaml`, weights transferred from yolo11n), × Original / CRG 1× × seeds 0–1;
- `eval_ensemble.py`: test-time augmentation and a 3-seed weighted-boxes-fusion ensemble of the base models, no retraining. Its single-model scores are about 0.005 higher than `detect.py`'s (per-image predict vs batched val), so TTA and ensembles are compared only against the same tool's single-model score.
- 2026-11-24 18:13 P5 泄漏诊断入队（用户要求）：build_leak_jobs.py → data/synthetic/jobs_Aleak（300 个任务 = 测试裁剪去缺陷后的背景 + 测试缺陷的类别、位置和框；板 10～12 模板新建）；
  leak_gen_crg_0/1 生成 crgleak_A，之后 crgleak_1x、crgleak_synonly × 种子 0–2（variant base）。只用于解释论文结果，不计入指标。

### 2026-11-24 18:50 · P2 done, P3 mostly done (P2 head pending)

Select mAP@50–95 = mean over seeds (base Original / CRG 1× over seeds 0–1: 0.5149 / 0.5199). Test shown for completeness only.

| Setting (base detector) | Seeds | Select | Test mAP / P / R |
|---|---|---|---|
| Original | 3 | 0.5145 | 0.497 / 0.956 / 0.936 |
| CRG 1× | 3 | 0.5193 | 0.494 / 0.969 / 0.921 |
| CRG 2× / 3× | 2 | 0.5178 / 0.5118 | 0.488 / 0.492 |
| DDIM 1× / 2× / 3× | 2 | 0.5181 / 0.5192 / 0.5142 | 0.494 / 0.492 / 0.491 |
| copy-paste 1× / **2×** / **3×** | 2 | 0.5121 / **0.5239** / **0.5234** | 0.500 / 0.490 / 0.495 |
| CRG + copy-paste 1× / 2× | 2 | 0.5154 / 0.5170 | 0.494 / 0.502 |

| Detector variant | Select Original / CRG 1× (2 seeds) | Test Original / CRG |
|---|---|---|
| box10 | 0.5160 / 0.5191 | 0.496 / 0.497 |
| box15 | 0.5180 / 0.5151 | 0.500 / 0.494 |
| dfl3 | 0.5164 / 0.5222 | 0.491 / 0.497 |
| p2 | 0.5065 (1 seed) / pending | 0.479 / – |

Inference-time (eval_ensemble.py, compare only within the tool; select / test):

| | Original | CRG 1× |
|---|---|---|
| single (seed 0) | 0.5181 / 0.500 | 0.5148 / 0.496 |
| TTA | 0.5164 / 0.507 | 0.5102 / 0.499 |
| **3-seed ensemble (WBF)** | **0.5363 / 0.521** (P 0.951, R 0.963) | **0.5376 / 0.519** (P 0.955, R 0.951) |
| ensemble + TTA | 0.5285 / 0.519 | 0.5335 / 0.521 |

Decisions (rule 4):
- copy-paste 2× and 3× are ≥ 0.005 above Original on select (+0.009, +0.009) but not above CRG 1× (+0.004); test is 0.007 below Original, inside the bootstrap SD (0.012). Not decided: seed 2 queued.
- box/DFL weights: no variant ≥ 0.005 above base for both configs. Rejected. P2 head: first seed 0.008 below base; wait for the other 3 runs.
- TTA: lower on select for both, alone and on top of the ensemble. Rejected.
- **3-seed ensemble: +0.018 (Original) and +0.023 (CRG) on select. Accepted.** Test confirms (+0.021 / +0.022 over the single model in the same tool). With it, Original reaches all three targets on test: P 0.951, R 0.963, mAP 0.521. The ensemble needs 3 models at inference (3× time), which should be stated.
- Synthetic data still adds nothing on top: CRG ensemble ≈ Original ensemble on both sets.

Queued (P4): copy-paste 2×/3× seed 2; base Original and CRG 1× seeds 3–4 (5 seeds each); ensembles of all 5 seeds and of seeds 2–4 (a second, disjoint-ish triple, to check the gain repeats); copy-paste 2× 3-seed ensemble. Then the leak diagnostic, then the defect-model retrain (already queued).

### 2026-11-24 20:50 · P4 seeds and ensembles done; final setting chosen

Single detectors (base, select mean ± SD over seeds; test for completeness):

| Setting | Seeds | Select | Test mAP / P / R |
|---|---|---|---|
| Original | 5 | 0.5165 ± 0.0030 | 0.492 / 0.947 / 0.928 |
| CRG 1× | 5 | 0.5206 ± 0.0036 | 0.497 / 0.962 / 0.924 |
| copy-paste 2× | 3 | 0.5214 ± 0.0041 | 0.497 / 0.947 / 0.916 |
| copy-paste 3× | 3 | 0.5204 ± 0.0043 | 0.494 / 0.955 / 0.931 |
| P2 head: Original / CRG | 2 | 0.5094 / 0.5123 | 0.478 / 0.489 |

Ensembles (eval_ensemble.py; select / test, P, R on test):

| Members | Original | CRG 1× | copy-paste 2× |
|---|---|---|---|
| seeds 0–2 | 0.5363 / 0.521 | 0.5376 / 0.519 | **0.5566 / 0.523** (P 0.948, R 0.956) |
| seeds 2–4 | 0.5418 / 0.520 | 0.5398 / 0.514 | – |
| seeds 0–4 | 0.5468 / 0.529 (P 0.958, R 0.968) | 0.5421 / 0.526 (P 0.959, R 0.968) | queued |

Decisions:
- P2 head rejected (select −0.007 vs base).
- Ensemble gain repeats with a different set of seeds (2–4): +0.02 on select for both. Five members add +0.005–0.010 on select over three for Original, +0.002–0.005 for CRG.
- copy-paste 2× vs Original: +0.005 on select over 5 Original seeds (+0.007 over the same 3 seeds), test equal (0.497) → passes rule 4, narrowly. As an ensemble the select gap is +0.020 (0.5566 vs 0.5363), but test only +0.002. Select and test disagree on how much copy-paste helps; reported as such.
- **Final setting: base YOLO11n, real data + copy-paste 2×, 5-seed ensemble.** Fallback that needs no synthetic data: Original 5-seed ensemble (test P 0.958, R 0.968, mAP 0.529 — all targets met).

Queued: copy-paste 2× seeds 3–4 and its 5-seed ensemble (final number). Leak diagnostic running: first seed CRG-leak 1× = **test 0.660**, select 0.529 (vs Original 0.497 / 0.5165) — wait for seeds 1–2 and synthetic-only before reading it.

### 2026-11-24 20:55 · P4 done (5 seeds, ensembles); P2 head rejected

Select mAP@50–95 (mean over seeds); test only for the chosen setting.

| Setting (base detector) | Seeds | Select | Test mAP / P / R |
|---|---|---|---|
| Original | 5 | 0.5165 | 0.492 / 0.947 / 0.928 |
| CRG 1× | 5 | 0.5206 | 0.497 / 0.962 / 0.924 |
| copy-paste 2× | 3 | 0.5214 (seeds 0–2 Original: 0.5145, +0.007) | 0.497 |
| copy-paste 3× | 3 | 0.5204 | 0.494 |
| P2 head, Original / CRG | 2 | 0.5094 / 0.5123 | – |

| Ensemble (eval_ensemble.py) | Select | Test mAP / P / R |
|---|---|---|
| Original ens3 (s0–2) / ens3 (s2–4) / **ens5** | 0.5363 / 0.5418 / **0.5468** | 0.521 / 0.520 / **0.529 / 0.958 / 0.968** |
| CRG 1× ens3 / ens3 (s2–4) / ens5 | 0.5376 / 0.5398 / 0.5421 | 0.519 / 0.514 / 0.526 / 0.959 / 0.968 |
| copy-paste 2× ens3 | 0.5566 | 0.523 / 0.948 / 0.956 |

Decisions:
- CRG 1× vs Original over 5 seeds: +0.004 on select, below the rule. Synthetic CRG data is not an improvement.
- P2 head: below base for both configs. Rejected.
- Ensemble gain repeats on a second seed triple (+0.025 / +0.020); 5 models add a further +0.005 / +0.002. **Ensemble accepted; with 5 models, Original meets all three targets on test (P 0.958, R 0.968, mAP 0.529).**
- copy-paste 2× passes the rule against seed-matched Original (+0.007) and its 3-model ensemble is highest on select (+0.020 over Original ens3). The 20:49 check-in queued seeds 3–4 and a 5-model ensemble to confirm; decide when they finish.
- Leak diagnostic (P5, labelled, not counted toward targets): first run crgleak_1x seed 0 gives test 0.660 vs select 0.529. Remaining seeds running.
- The defect-model retrain is on hold at the user's request (release: mkdir -p runs/release_maxcov && touch runs/release_maxcov/go).

### 2026-11-24 22:50 · copy-paste 2× resolved; P5 leak diagnostic done; search plan complete

| Setting (base detector) | Seeds | Select (mean ± SD) | Test mAP / P / R |
|---|---|---|---|
| Original | 5 | 0.5165 ± 0.0030 | 0.492 / 0.947 / 0.928 |
| CRG 1× | 5 | 0.5206 ± 0.0036 | 0.497 / 0.962 / 0.924 |
| copy-paste 2× | 5 | 0.5182 ± 0.0054 | 0.497 / 0.955 / 0.920 |
| **Original, 5-model ensemble** | 5 | **0.5468** | **0.529 / 0.958 / 0.968** |
| CRG 1×, 5-model ensemble | 5 | 0.5421 | 0.526 / 0.959 / 0.968 |
| copy-paste 2×, 5-model ensemble | 5 | 0.5512 | 0.533 / 0.952 / 0.962 |

Decisions:
- copy-paste 2×: seeds 3–4 (0.5165, 0.5105) pull the 5-seed mean to +0.0017 over Original; its 5-model ensemble is +0.0044 over Original's. Both below the 0.005 rule. Rejected; the 3-seed lead was seed noise.
- **Final setting: Original (real data only), YOLO11n base, 5-model WBF ensemble. Test: P 0.958, R 0.968, mAP@50–95 0.529, all three targets met.** Without the ensemble, no setting reaches mAP 0.50 on test (best single-model mean 0.497).
- No synthetic-data method improves on Original under the rule, single or ensembled.

P5 leak diagnostic (labelled, NOT a result, not counted toward targets). Synthetic images built from the test crops' own backgrounds and defect positions/boxes (jobs_Aleak), 3 seeds:

| | Select | Test mAP / P / R |
|---|---|---|
| Original + crgleak 1× | 0.5257 | **0.650** / 0.979 / 0.968 |
| crgleak only (no real data) | 0.3898 | **0.570** / 0.894 / 0.829 |

On select (no leak) crgleak behaves like ordinary CRG (+0.009); on test it jumps +0.159, and synthetic-only beats real data (0.570 vs 0.492). That is the pattern of the paper (large CRG gain; synthetic-only 0.527–0.549 > real 0.496), with a larger magnitude. It shows that such leakage is enough to produce the paper's pattern; it does not show that the authors did it.

Queue: all search jobs finished; only retrain_defect64_maxcov remains, on hold per the user. GPU idle from 21:16.

### 2026-11-24 22:50 · Final numbers; leakage diagnostic done; queue idle (on hold)

Final setting chosen on select (rule 2): **base YOLO11n, real data + copy-paste 2×, 5-model ensemble.**

| 5-model ensemble (seeds 0–4) | Select mAP | Test mAP@50–95 | Test P | Test R |
|---|---|---|---|---|
| Original (real only) | 0.5468 | 0.529 | 0.958 | 0.968 |
| CRG 1× | 0.5421 | 0.526 | 0.960 | 0.968 |
| **copy-paste 2× (chosen)** | **0.5512** | **0.533** | **0.952** | **0.962** |

Single models, 5 seeds: copy-paste 2× select 0.5182 / test 0.497; Original 0.5165 / 0.492; CRG 1× 0.5206 / 0.497.

All three targets are met by the chosen setting (P 0.952, R 0.962, mAP 0.533), and also by Original alone with the same ensemble. Synthetic data adds at most +0.004 on top (copy-paste 2×) and CRG adds nothing (−0.003). The gain comes from ensembling five detectors.

Leakage diagnostic (P5, labelled, **not a result**): CRG images generated on the test crops' own backgrounds with the test defects' class and position (`jobs_Aleak`), 3 seeds: real + CRG-leak 1× test **0.651** (select 0.526, R 0.968); CRG-leak synthetic only test **0.570** (select 0.390). With leaked backgrounds the test score jumps +0.15 while select does not move, and synthetic-only beats real data, as both papers report. This is consistent with the paper's synthetic backgrounds overlapping its test images; it does not prove it.

Queue idle since 21:16: only the defect-model retrain (maxcov) and its follow-ups remain, on hold at the user's request (release: `mkdir -p runs/release_maxcov && touch runs/release_maxcov/go`).

### 2026-11-25 · P6 queued: detector-side improvements (user: keep the authors' diffusion checkpoint, change only the detector part; option b = end of queue)

Rules 1–4 above still bind. Every idea is paired: real only vs real + generated, same detector settings.

| Idea | Jobs | Compare against |
|---|---|---|
| P6-1 two-stage: pretrain on 900 CRG images only (`crg_synall`, select picks best.pt), then train on real only from those weights (`--variant ft --init`) | `p6_pre_crg900_s0-2`, `p6_ft_{A,A10,A20}_s0-2` | Original from yolo11n.pt: A `detection_A_vbase/original` s0–2; A10/A20 `detection_A{10,20}_img640/original` (select from results.csv) |
| P6-2 refined synthetic boxes (`relabel_synthetic.py`: real-only detector's box replaces the label when same class, IoU ≥ 0.3, conf ≥ 0.25; 142/300 replaced) | `p6_relab_{A,A10}_s0-2` (`crgrelab_1x`, variant base) | CRG 1× with original boxes (A vbase; A10 img640) |
| P6-3 90° rotations (`rot90` = flipud 0.5 + transpose 0.5 → 8 orientations; `aug_rot90.py`, checked visually) | `p6_rot90_{A,A10}_{original,crg_1x}_s0-2` | same configs, base |
| P6-6 mixed ensemble (WBF, no training) | `p6_ens_mix6` (orig+CRG s0–2), `p6_ens_mix10` (s0–4) | `original_ens3/ens5`, `crg_1x_ens3/ens5` (same tool) |

**Final analysis when the queue is empty** (whoever runs the check-in): 
1. For each P6 idea and each data level, select mean over 3 seeds vs its comparison; accept only if ≥ +0.005 on select (rule 4). 
2. Combine accepted ideas only if two or more pass (e.g. rot90 + two-stage), as one more round of runs, and ask the user before queueing it. 
3. "Best method for the requirement": the task book targets on test (P ≥ 0.90, R ≥ 0.90, mAP@50–95 ≥ 0.50) for the setting chosen on select; report the best single model and the best ensemble separately (ensembles cost N× inference). Report separately whether generated data helps at 50/class and at 10–20/class under the chosen detector. 
4. Note the clean-image check (`cf_*`, detection_A_cleanfix): if it moved Original or CRG by > 0.005, P6 results on split A need rerunning on the fixed set. 
5. Write the result table and the decision into this log, update `docs/CRG-DefectDiffuser_architecture.html` only if asked, and report to the user.

### 2026-11-25 18:55 · clean-image check done; detect.py output-path bug fixed

Clean-image check (`detection_A_cleanfix`: the 33 clean training images cut from test/select photos replaced by train-photo crops; base detector, seeds 0–2):

| | Select: old set → fixed set | Test: old set → fixed set |
|---|---|---|
| Original | 0.5145 → 0.5237 (+0.009; one seed 0.538) | 0.497 → 0.490 (−0.006) |
| CRG 1× | 0.5193 → 0.5153 (−0.004) | 0.494 → 0.503 (+0.008) |

No consistent direction (select and test move oppositely; Original and CRG oppositely); all changes are within seed noise (Original select SD on the fixed set 0.010). Conclusion unchanged: at 50/class neither configuration is reliably better. The leak had no systematic effect. Per rule 4 of the P6 entry, P6 split-A real-data jobs now use the fixed set (`--split Aclean`, comparison `detection_Aclean_vbase`); added `clean_{original,crg_1x}_ens3`, `clean_mix6` ensembles. A10/A20/A30/A40/A60 keep the old clean images (identical across methods, so the trend comparison is unaffected) — note this in the thesis.

Bug: my 16:47 P6 edit put `sub = detection_{split}_v{variant}` under `if args.init`, so `--variant` runs started 16:47–18:55 wrote to `detection_{split}/`. Only cf_crg_1x_s1, cf_original_s2, cf_crg_1x_s2 were affected (moved to `detection_Aclean_vbase/`, results valid). Fixed before the img800 / synonly variant jobs started; they would have found old 256-input results in `detection_A/` and been skipped as "done already".

### 2026-11-26 00:50 · trend table (10–60 per class, all methods) and input-size runs done

Select mAP@50–95 (mean of 3 seeds; Δ vs Original at the same level). Test in brackets for CRG only.

| Per class | Original | CRG | ECICS | DDIM | RePaint | CoPaint | ControlNet | DefFiller |
|---|---|---|---|---|---|---|---|---|
| 10 | 0.4169 | **+0.054** (test +0.059) | +0.053 | +0.024 | +0.000 | −0.034 | +0.004 | −0.012 |
| 20 | 0.4853 | **+0.010** (+0.022) | +0.005 | −0.006 | −0.012 | −0.012 | −0.005 | −0.013 |
| 30 | 0.5020 | +0.005 (+0.008) | +0.007 | +0.006 | −0.004 | −0.004 | −0.003 | −0.004 |
| 40 | 0.5105 | +0.003 (−0.007) | +0.001 | −0.001 | +0.005 | −0.003 | −0.004 | −0.012 |
| 50 | 0.5140 | +0.005 (−0.003) | +0.004 | +0.004 | −0.005 | −0.005 | −0.008 | −0.008 |
| 60 | 0.5245 | −0.003 (−0.001) | −0.001 | +0.000 | −0.005 | −0.002 | −0.004 | −0.005 |

Reading (rule 4): CRG/ECICS help clearly at 10 and 20 per class (select and test agree); from 30 per class on the gain is at the 0.005 threshold or below and select/test no longer agree; at 60 it is zero. Inpainting and whole-image methods (RePaint, CoPaint, ControlNet, DefFiller) never pass the threshold and mostly hurt. Original test: 0.388 / 0.446 / 0.478 / 0.494 / 0.497 / 0.490 (60 lower than 50 on test, higher on select — test noise ±0.012). The 30/40/60 sets keep the old 50 clean images (same for every method).

Input size (3 seeds now for 800): Original / CRG select 0.5209 / 0.5200, test 0.489 / 0.498 — no gain over 640. Generated-only (300 CRG images): test 0.364 (256), 0.335 (640), 0.324 (800), 0.339 (960); recall 0.63–0.73.

Failure: maxcov_select_ckpt OOM next to two AD trainings; re-queued as `maxcov_select_ckpt_retry` (11 GB need).

### 2026-11-26 11:10 · status: retrained defect model, generation quality, ablation (partial)

- Retrained defect model (`crgmaxcov`, 777 patches, ema 180k): no better than the old own model; select 0.5090 at 50/class, 0.4303 at 10/class vs authors' CRG 0.5206 / 0.4708. It memorises its patches (NN ratio 0.055). Rejected; authors' checkpoint stays.
- Generation quality (`gen_quality.py`, reference = split-A select crops; KID×10³ full image / 64 window; LPIPS diversity, real 0.476): CRG 12.3 / 12.4, ECICS 13.3 / 24.7, DDIM 11.4 / 30.0, RePaint 10.2 / 23.1, CoPaint 12.2 / 24.0, ControlNet 16.5 / 20.3, DefFiller 56.0 / 62.0, copy-paste 9.4 / 25.6. CRG has the lowest window KID: its defects look most like real ones; DefFiller is far off.
- Ablation so far (select, 3 seeds unless noted): no defect loss L_d: 50/class 0.5109, 10/class 0.3926 (below Original 0.4169: the guidance without L_d mostly smooths the window); no outer background loss L_o: 1 seed, 0.5127 / 0.4606. no inner background loss and no mask averaging: detection pending. Window KID: no L_d 19.0, no L_b 15.8, no L_o 11.8, no mask EMA 12.2 (full CRG 12.4).

### 2026-11-26 12:50 · AnomalyDiffusion 1× done (2×/3× running)

AD trained per class (6 × 5000 steps), 900 images generated. Select (test) mAP@50–95, 3 seeds: 50/class 0.5132 (0.483) vs Original 0.5140 (0.497); 20/class 0.4657 (0.433) vs 0.4853 (0.446); 10/class 0.4053 (0.379) vs 0.4169 (0.388). Never above Original; like the other whole-image / inpainting baselines, unlike CRG (+0.054 at 10/class).

### 2026-11-26 14:50 · ablation done (4 settings × 50/10 per class × 3 seeds)

Each setting removes one part of the guidance; 300 images each, same jobs as CRG 1×. Select mAP@50–95 (test in brackets); window KID×10³ (gen_quality, lower = more realistic defect window).

| Setting | Window KID | 50/class | 10/class | Δ vs full CRG at 10/class (select) |
|---|---|---|---|---|
| real only | — | 0.5145 (0.497) | 0.4169 (0.388) | −0.054 |
| **full CRG** | 12.4 | 0.5193 (0.494) | **0.4708 (0.447)** | — |
| no defect loss L_d | 19.0 | 0.5109 (0.483) | 0.3926 (0.348) | **−0.078** (below real only) |
| no inner background loss L_b | 15.8 | 0.5145 (0.486) | 0.4554 (0.428) | −0.015 |
| no outer background loss L_o | 11.8 | 0.5165 (0.497) | 0.4648 (0.432) | −0.006 |
| no mask EMA | 12.2 | 0.5170 (0.492) | 0.4597 (0.436) | −0.011 |
| no attention mask at all (ECICS) | 24.7 | 0.5181 (0.490) | 0.4695 (0.447) | −0.001 |

Reading (rule 4, at 10/class where generated data matters): L_d is essential (removing it makes generated data harmful); L_b and the mask EMA each contribute ~0.01–0.015 (≥ 0.005, test agrees); L_o is at the threshold (−0.006 select, −0.015 test). Removing the attention mask entirely (ECICS) costs nothing on detection although its window KID is twice as high — so the mask improves how the defect window looks, not what the detector learns. At 50/class no setting differs from real only by more than 0.005, consistent with the trend table.

### 2026-11-26 16:50 · augmentation ratio (1×/2×/3×) done for all methods, 50/class, 3 seeds

Select mAP@50–95 (Δ vs Original 0.5145; test in brackets). 1× for RePaint/CoPaint/ControlNet/DefFiller/AD from the img640 runs (same settings as base).

| Method | 1× (300) | 2× (600) | 3× (900) |
|---|---|---|---|
| CRG | 0.5193 (+0.005; 0.494) | 0.5166 (+0.002; 0.485) | 0.5159 (+0.001; 0.492) |
| DDIM fine-tune | 0.5157 (+0.001; 0.495) | 0.5197 (+0.005; 0.496) | 0.5179 (+0.003; 0.493) |
| copy-paste | 0.5150 (+0.001; 0.501) | 0.5214 (+0.007; 0.497) | 0.5204 (+0.006; 0.494) |
| RePaint | 0.5086 (−0.006; 0.474) | 0.5109 (−0.004; 0.481) | 0.5139 (−0.001; 0.478) |
| CoPaint | 0.5089 (−0.006; 0.487) | 0.5149 (+0.000; 0.488) | 0.5134 (−0.001; 0.486) |
| ControlNet | 0.5059 (−0.009; 0.482) | 0.5108 (−0.004; 0.475) | 0.5053 (−0.009; 0.470) |
| DefFiller | 0.5065 (−0.008; 0.486) | 0.4997 (−0.015; 0.476) | 0.4918 (−0.023; 0.464) |
| AnomalyDiffusion | 0.5132 (−0.001; 0.483) | 0.5096 (−0.005; 0.490) | 0.5055 (−0.009; 0.481) |

Reading (rule 4): at 50/class more generated data does not help any method; nothing passes +0.005 on select with test agreeing (copy-paste 2×/3× pass on select but test is equal or lower, as in P4). Methods with poor window realism get worse with more images (DefFiller −0.023 at 3×, ControlNet −0.009, AD −0.009); CRG, DDIM and copy-paste stay at the Original level at any ratio. Consistent with the trend table: at 50/class the detector is not data-limited.

### 2026-11-26 18:50 · P6 done (queue finished Thu 18:29, 35 jobs, all exit 0) — final analysis

`code/experiments/summarize_p6.py`. Select = best-fitness row of train/results.csv (same measure as the trend table; A10 Original 0.4169 reproduced). 3 seeds each; test only reported. Split A uses the fixed clean set (Aclean).

| Idea | Data | Select new vs ref (Δ) | Test new vs ref | Rule 4 |
|---|---|---|---|---|
| two-stage (900 CRG pretrain → real) | 50/class | 0.5106 vs 0.5234 (−0.013) | 0.494 vs 0.490 | reject |
| | 10/class | 0.4702 vs 0.4169 (+0.053) | 0.439 vs 0.388 | pass (≈ CRG 1× mixed, 0.4708) |
| | 20/class | 0.5028 vs 0.4853 (+0.018) | 0.467 vs 0.446 | pass |
| relabelled CRG boxes | 50 / 10 | +0.004 / −0.002 | 0.495 vs 0.503 / 0.459 vs 0.447 | reject |
| **rot90 (8 orientations)**, Original | 50 / 10 | **+0.025 / +0.057** | 0.516 vs 0.490 / 0.437 vs 0.388 | **pass** |
| **rot90**, CRG 1× | 50 / 10 | **+0.026 / +0.025** | 0.510 vs 0.503 / 0.477 vs 0.447 | **pass** |
| mixed ensemble mix6 / mix10 | 50 | 0.5510 / 0.5574 vs Original ens5 0.5468 | 0.534 / 0.540 vs 0.529 | +0.004 / +0.011, but more models (6/10 vs 5): confounded |

Stage 1 alone (900 CRG images, no real): select 0.405, test 0.394.

Requirement (P ≥ 0.90, R ≥ 0.90, mAP@50–95 ≥ 0.50 on test):
- Best single model, chosen on select: rot90 Original at 50/class, select 0.5484; test mAP 0.516, P 0.965, R 0.960 (mean); every seed passes (0.521 / 0.513 / 0.513). First single-model setting that meets all three targets; no ensemble or TTA needed.
- Best ensemble on select: mix10 (Original + CRG base, 5 seeds each), test mAP 0.540, P 0.967, R 0.975. rot90 ensembles not yet evaluated.
- Generated data under rot90: 50/class CRG −0.008 select vs Original (no help); 10/class CRG +0.022 select, +0.040 test (helps). Same picture as with the base detector: generated data matters when real defects are few.

Rule 2: rot90 and two-stage both pass (two-stage only at 10–20/class). Candidate combination round (not queued, ask the user): rot90 ensembles (eval only, ~3 min each), rot90 + two-stage at A10/A20, rot90 at A20.

### 2026-11-26 19:00 · P7 queued (user approved the combination round)

| Jobs | Compare against |
|---|---|
| `p7_ens_rot90_{orig3,crg3,mix6}` (WBF of the Aclean rot90 runs, no training) | `clean_original_ens3`, `clean_crg_1x_ens3`, `clean_mix6` (same model counts) |
| `p7_rot90ft_{A10,A20}_s0-2` (`--variant rot90+ft`, stage-1 weights from `p6_pre_crg900`, not rotated) | rot90 alone (A10 `detection_A10_vrot90/original`, A20 below) and two-stage alone (`detection_A{10,20}_vft/original`) |
| `p7_rot90_A20_{original,crg_1x}_s0-2` | `detection_A20_img640/{original,crg_1x}` |

Rule 4 as before (≥ +0.005 select). The final choice is the best single setting on select at each data level; test is reported only for those.

### 2026-11-26 19:45 · P7 done (queue finished 19:41, 15 jobs, all exit 0)

Select mean of 3 seeds (test in brackets), `summarize_p6.py`.

| Setting | 10/class | 20/class | 50/class (Aclean) |
|---|---|---|---|
| base, Original | 0.4169 (0.388) | 0.4853 (0.446) | 0.5234 (0.490) |
| base, CRG 1× | 0.4708 (0.447) | 0.4955 (0.468) | 0.5146 (0.503) |
| rot90, Original | 0.4738 (0.437) | **0.5113** (0.476) | **0.5484** (0.516) |
| rot90, CRG 1× | **0.4955** (0.477) | 0.5094 (0.493) | 0.5403 (0.510) |
| rot90 + two-stage | 0.4919 (0.450) | 0.5058 (0.475) | — |

- rot90 at 20/class passes (+0.026 Original, +0.014 CRG).
- rot90 + two-stage: better than rot90 Original at 10/class (+0.018) but below rot90 + CRG mixed (−0.004 at 10 and 20); at 20/class below rot90 alone. Not accepted; mixing generated data into training stays the way to use it.
- Generated data under rot90: 10/class +0.022 select (+0.040 test) — helps; 20/class −0.002 (test +0.017) — tie on select; 50/class −0.008 — no help.
- rot90 ensembles (same model counts as the clean_* ones): Original ens3 0.5609 vs 0.5451 (+0.016), CRG ens3 0.5574 vs 0.5411 (+0.016), mix6 0.5673 vs 0.5567 (+0.011). Test: 0.530 / 0.523 / 0.542; mix6 P 0.960 R 0.965.

**Final choice (on select)**: single model = rot90 Original at 50/class (test P 0.965, R 0.960, mAP 0.516; every seed ≥ 0.513). Ensemble = rot90 mix6 (test P 0.960, R 0.965, mAP 0.542). At 10/class: rot90 + CRG (test 0.477, P 0.939, R 0.919). Search complete; thesis chapter 5 updated from these numbers.
