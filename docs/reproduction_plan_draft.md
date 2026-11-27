# Reproduction plan: CRG-DefectDiffuser on PKU-Market-PCB (draft)

Draft of 2026-11-18. It reproduces **CRG-DefectDiffuser** (Luo et al., TCSVT 2026), the journal version of **DEFectDiffuser** (ECICS 2025), and judges every generated image by how much it helps a defect detector trained on real data.

Paper references: Table I (comparison), Fig. 4–5 (augmentation scale), Table II (perceptual quality), Table III (ablation), Table IV (inference time).

---

## 1. Models

### 1.1 Our own models (trained from scratch, repo code, `torch.compile`)

| | DPM: Defect Prior Model | CSM: Contextual Synthesis Model |
|---|---|---|
| Learns | what a defect looks like | what a defect-free board looks like |
| Training data | 300 defect patches, 64×64, 50 per class (`defect_64_50pc`) | 3,000 defect-free crops, 256×256 (`free_256`) |
| Network | guided-diffusion UNet, 64 ch, mult 1-2-3-4, attention at 16/8 | guided-diffusion UNet, 128 ch, mult 1-1-2-3-4-4, attention at 16/8 |
| Diffusion | 1,000 steps, linear schedule, learn_sigma (6 output channels) | same |
| Training | 200k steps, batch 64, lr 1e-4, EMA 0.9999, fp16 | 500k steps, batch 8, lr 1e-4, EMA 0.9999, fp16 |
| Status | **done** (200k, `runs/defect_64_50pc/`) | **running**: 356k / 500k, ends Thu 19 Nov ≈ 10:30 (`runs/free_256/`) |
| Checkpoint choice | model vs EMA weights, every saved step, by KID on the selection split | same |

### 1.2 The authors' checkpoints

`checkpoint/checkpoint-64-defect.pt` (DPM) and `checkpoint/checkpoint-256-defect-free.pt` (CSM). All results to date use them. Our own models replace them in E5.

### 1.3 Known differences from the paper

| Item | Paper | Ours | Effect |
|---|---|---|---|
| CSM training set | 1,800 defect-free images | 3,000 crops | more background variety; same step count |
| Image scale | not stated | 0.5× (confirmed by matching the repo's sample image to board 01) | — |
| Detector split | not stated | split A: by photo, all boards seen in training (paper-like) · board split: test boards unseen (stricter) | board split gives lower absolute mAP |
| Box of a synthetic defect | not stated | placement box of a real defect at the same board position | — |
| ControlNet condition | text prompt | text + Canny edge map of the background, box area blanked | — |
| Loss weights in code | λ 1 / 0.1 / 0.01 | code uses 1e5 / 1e4 / 5e4 (kept as released) | tested in E6 |

---

## 2. Data

- **Source:** PKU-Market-PCB: 693 photos, 10 boards, 6 classes (missing hole, mouse bite, open circuit, short, spur, spurious copper). Scaled to 0.5×.
- **Generator data:** 64×64 defect patches (2,185 available, 300 used: 50 per class) and 256×256 defect-free crops (3,000; defects erased with a per-board median template; every crop checked pixel by pixel).
- **Detector data, split A (main):** train = 300 defect crops (50 per class) + 50 defect-free crops; select = 300 crops for best-epoch choice and tuning; test = 300 crops (50 per class), one per photo. Test photos never supply training crops.
- **Detector data, board split (secondary):** train on boards 01, 04–09, test on boards 10–12.
- **Synthetic samples:** each sample is a defect-free background + one defect at the position of a real defect of the same class on the same board. Every method gets the same jobs (background, position, class), so differences come only from the generator. The classes are balanced, and the scales are nested (1× ⊂ 2× ⊂ 3×).

---

## 3. Methods compared (paper Table I)

| Method | Paradigm | Backbone | Implementation | Status (split A, 1×) |
|---|---|---|---|---|
| Original | no synthetic data: detector trained on the 300 + 50 real crops only (paper: "Original (No Aug.)"); every gain is measured against it | — | — | done |
| DDIM (fine-tune) | generated patch pasted | authors' DPM | ours (`dpmpaste`) | done |
| RePaint | inpainting, resampling (2 steps) | authors' CSM | repo | done |
| CoPaint | inpainting, Bayesian | authors' CSM | repo approximation | done |
| ControlNet | text-conditioned full synthesis | SD 1.5 + ControlNet, fine-tuned 3k steps | diffusers | done |
| AnomalyDiffusion | mask-guided editing of a reference | LDM text2img-large | official repo (needs port from PyTorch 1.12) | **to do** (after 256 training) |
| DefFiller | mask-conditioned synthesis | GLIGEN-sem (SD 1.4) | official repo | **to do** (after 256 training) |
| **CRG-DefectDiffuser** | collaborative refinement guidance | authors' DPM + CSM | repo `sample_generation.py` | done |
| Copy-paste · ECICS · CRG-window | extra references (not in paper) | — | ours | done |

---

## 4. Metrics

### 4.1 Downstream detection (main criterion, as in the paper)

Detector: YOLOv11n, 200 epochs, batch 32, **640 input** (Ultralytics default; decided 2026-11-19, Section 10; earlier runs used 256). Trained on the real training set + synthetic samples, tested on real test crops only. **3 seeds per setting, reported as mean ± std.**

| Metric | In paper | Note |
|---|---|---|
| **mAP@50–95** | yes (headline) | mean AP over IoU 0.50, 0.55, …, 0.95 |
| **Recall** | yes | share of real defects found |
| Precision | no | reported for completeness |
| mAP@50 | no | reported for completeness |
| Per-class AP@50–95 and Recall | yes | Table I, Fig. 4 |

### 4.2 Generation quality (paper Table II, IV)

| Metric | In paper | Note |
|---|---|---|
| **KID ×10³** | yes | unbiased with few samples; main quality metric |
| **LPIPS** | yes | perceptual distance to real crops |
| FID | no | extra; biased at 300 images, so report together with KID |
| Generative precision / recall (Kynkäänniemi) | no | extra; fidelity vs diversity. These are not the detector's precision and recall |
| NN ratio | no | memorisation check: distance to the nearest training patch |
| Time per 256×256 image | yes (Table IV) | single GPU job |

The paper's own conclusion (Sec. V-E): KID and LPIPS do not rank methods the way detection does, so detection decides.

---

## 5. Targets

Main setting: split A, CRG with the authors' checkpoints, 1×, mean of 3 seeds.

| Target | Paper (CRG) | Goal | Now | Met |
|---|---|---|---|---|
| Recall | 0.935 | **≥ 0.90** | 0.866 | no |
| mAP@50–95 | 0.557 | **≥ 0.50** | 0.459 | no |
| Precision | not reported | ≥ 0.90 | 0.933 | yes (Original already 0.929) |
| mAP@50 | not reported | — | 0.921 | (all methods 0.89–0.93) |
| Gain over Original (mAP@50–95) | +12.3 % | **CRG best of the 8 Table I methods, gain > 2 × seed std** | +1.7 %; 2nd of 6 done (DDIM first) | no |
| Scale 1× / 2× / 3× | 0.557 / 0.573 / 0.579 | CRG increases with scale; baselines flatten or drop | board split running | — |
| Ablation (synthetic only) | 0.527 → 0.549, each step ≥ previous | same order | 0.35–0.37 with 300 images | no |
| Own models | — | own-checkpoint CRG within seed noise of authors' checkpoints | pending | — |

Notes on the targets:
- **"mAP > 0.5" means mAP@50–95.** mAP@50 is already 0.89–0.93 for every method, so a 0.5 bar on it says nothing.
- **Our detector baseline is 0.045 below the paper's** (Original 0.451 vs 0.496) before any synthetic data is added. Reaching 0.50 therefore needs either a closer detector setup (check pretrained weights, input size, YOLO augmentation defaults, how test crops are cut) or a +11 % gain from augmentation. The relative targets matter more than the absolute ones.
- Precision ≥ 0.90 holds for every method including Original, so it works only as a floor.

---

## 6. Experiments

| ID | Reproduces | Setting | Status |
|---|---|---|---|
| **E1** Comparison | Table I | split A, 1×, 8 methods + 3 references, 3 seeds | 6 / 8 paper methods done; AnomalyDiffusion, DefFiller to do |
| **E2** Augmentation scale | Fig. 4–5 | 1× / 2× / 3× per method, 3 seeds; on split A only after the split is chosen (Section 7b) | board split: running (vs Original: DDIM 2× +8.5 %, RePaint 2× +1.2 % (2 seeds), copy-paste 2× +2.4 %, same as its 1×); split A: not started |
| **E3** Ablation | Table III | detector on synthetic only: base → +sigmoid → +Gaussian smoothing → +connected components → +EMA | board split, 300 per setting: partly done (pilot). Final: **900 per setting** (nested 300 ⊂ 900), 3 seeds, on the chosen split. The paper's steps differ by 0.004–0.007 mAP, about our seed std, so single steps may not separate; the recall jump at connected-component filtering (+0.031) should |
| **E4** Quality and cost | Tables II, IV | KID, LPIPS, FID, generative P/R, NN ratio, time, per method and class, against real test crops | to do (CPU/GPU < 1 h per method) |
| **E5** Own models | — | checkpoint selection, benchmark against the authors' checkpoints (KID, samples), then E1 CRG row with own checkpoints | DPM done; CSM ends Thu; selection + benchmark run automatically |
| **E6** Optimisation | — | on the select split only: T_start (20/30/50), λ weights (code vs paper), mask temperature, `t[0] / 10` → `% 10` fix, spurious-copper scale; best setting once on test | to do |
| **E7** Analysis | Figs. 6–8 | per-class results, false positives (spur), visual comparison grid, guidance-map evolution, failure cases | partly done (spur FP analysis, box check) |

---

## 7. Results so far (split A, 1×, 3 seeds, authors' checkpoints)

| Method | Precision | Recall | mAP@50 | mAP@50–95 | vs Original | Paper Recall | Paper mAP@50–95 |
|---|---|---|---|---|---|---|---|
| Original | 0.929 | 0.843 | 0.903 | 0.451 | — | 0.869 | 0.496 |
| DDIM (fine-tune) | 0.943 | 0.880 | 0.925 | 0.468 | +3.8 % | 0.876 | 0.495 |
| RePaint | 0.913 | 0.828 | 0.887 | 0.447 | −0.9 % | 0.874 | 0.492 |
| CoPaint | 0.927 | 0.843 | 0.896 | 0.438 | −2.9 % | 0.850 | 0.488 |
| ControlNet | 0.931 | 0.835 | 0.894 | 0.446 | −1.1 % | 0.897 | 0.526 |
| AnomalyDiffusion | — | — | — | — | — | 0.896 | 0.524 |
| DefFiller | — | — | — | — | — | 0.909 | 0.532 |
| **CRG** | 0.933 | 0.866 | 0.921 | 0.459 | +1.7 % | 0.935 | 0.557 |
| Copy-paste | 0.926 | 0.876 | 0.912 | 0.466 | +3.3 % | — | — |
| ECICS | 0.940 | 0.871 | 0.918 | 0.466 | +3.3 % | — | — |
| CRG-window | 0.931 | 0.863 | 0.916 | 0.469 | +4.0 % | — | — |

Seed std is 0.002–0.011 mAP@50–95.

## 7b. Which split fits the paper (1×, 3 seeds)

| Method | Board: Recall | Board: mAP@50–95 | Board: gain | A: Recall | A: mAP@50–95 | A: gain | Paper: Recall | Paper: mAP@50–95 | Paper: gain |
|---|---|---|---|---|---|---|---|---|---|
| Original | 0.842 | 0.413 | — | 0.843 | 0.451 | — | 0.869 | 0.496 | — |
| DDIM | 0.852 | 0.419 | +1.4 % | 0.880 | 0.468 | +3.8 % | 0.876 | 0.495 | −0.2 % |
| RePaint | 0.840 | 0.418 | +1.2 % | 0.828 | 0.447 | −0.9 % | 0.874 | 0.492 | −0.8 % |
| CoPaint | 0.851 | 0.432 | +4.5 % | 0.843 | 0.438 | −2.9 % | 0.850 | 0.488 | −1.6 % |
| ControlNet | — | — | — | 0.835 | 0.446 | −1.1 % | 0.897 | 0.526 | +6.0 % |
| CRG | 0.846 | 0.410 | −0.7 % | 0.866 | 0.459 | +1.7 % | 0.935 | 0.557 | +12.3 % |
| Copy-paste | 0.863 | 0.422 | +2.3 % | 0.876 | 0.466 | +3.2 % | — | — | — |
| ECICS | 0.832 | 0.412 | −0.2 % | 0.871 | 0.466 | +3.4 % | — | — | — |
| CRG-window | 0.837 | 0.418 | +1.1 % | 0.863 | 0.469 | +3.9 % | — | — | — |

- Method pairs ordered as in the paper: board 1 of 10, split A 10 of 15.
- Split A is closer to the paper on absolute level, on the sign of the CRG gain, and on the negative RePaint/CoPaint gains.
- The board split is the harder test: test boards are never seen. It is kept as the secondary result.
- To complete the comparison on the board split, ControlNet (and later DefFiller, AnomalyDiffusion) must be trained again on the board-split training crops.

---

## 8. Order of work

1. **Now → Thu 19 Nov ≈ 10:30:** CSM training finishes; the board-split E2/E3 queue keeps running alongside it.
2. **Thu–Fri:** with the GPU free, port and train DefFiller and AnomalyDiffusion, generate 300 split-A images each, 3 detector seeds → E1 complete.
3. **E5:** own-model checkpoint selection and benchmark (automatic after training), then the CRG row with own checkpoints.
4. **Choose the split** from the 1× comparison (Section 7b), then E3 (900 per setting) and E2 on that split.
5. **E4** quality metrics for all methods.
6. **E6** optimisation on the select split, then one final test run.
7. **E7** figures and tables; update the HTML report.

---

### Update 2026-11-19

With the detector at 640, no method beats Original on split A (Original 0.497, recall 0.936; authors' CRG 0.494). The test set leaves little room for augmentation. Queued, in order:
1. **CRG checks** (`runs/logs/crg_checks.sh`): check 1 uses the paper's Eq. 17 settings (`% 10` fix, no background term inside the ROI, λ = 1 / 0.1 / 0.01); check 2 filters CRG images with the Original detector (IoU ≥ 0.6).
2. **Evaluation with room for gain:** board split at 640 (Original, CRG, DDIM, copy-paste), and split A with only 10 or 20 real crops per class (Original, CRG, DDIM). 3 seeds each.
3. **Then:** if step 2 shows room, a CRG setting scan with a cheap proxy, tuned on the select set (T_start, threshold, temperature, loss weights, guidance strength). The test set is used once, for the final setting.

~~Dropped: tighter synthetic boxes.~~ Corrected 2026-11-20: the box check's 0.14 is the area of the bounding box of what CRG actually changed, divided by the label box area. So CRG's visible defect is much smaller than our label box (median IoU 0.14). How the paper drew synthetic boxes is not stated, so both alternatives are now tested (check C1 below).

### Update 2026-11-20 · checks for steps the paper does not state (split A, 640, 3 seeds)

| Check | What it tests | Data |
|---|---|---|
| A | best epoch picked on the test set (the paper describes only a training and a "validation" set) | `detect.py --proto valtest640`; Original, CRG |
| B | harder test crops: defects anywhere, 26 % of main defects cut by the crop edge | `data/detection_A_edge` (`build_detection_edge_test.py`); Original, CRG, DDIM |
| C1 | synthetic box labels: whole 64×64 window, or bounding box of the changed region | `crgwbox_A`, `crgcbox_A` (`build_label_variants.py`) |
| C2 | two defects per synthetic image (105 of 300 images get a second real placement) | `crgmd_A` (`make_multi_jobs.py`) |
| D | which boards the authors' 256 model saw (photos of a board are aligned copies, so train vs test photos cannot be told apart) | `check_csm_boards.py` |
| B2 | test crops cut on a 256-px grid of the 0.5× photos, as the paper's figures suggest | `data/detection_A_grid` (`build_detection_grid_test.py`); Original, CRG, DDIM |

Check D result: nearest real board of the authors' 256-model samples is 10–12 in 10 % of cases vs 3.3 % for ours (never saw them), so their background model most likely saw boards 10–12. Clues about the authors' data preparation (code, figures): `results/experiments/paper_figs/clues.md`, also in the HTML report's log of 2026-11-20.

## 9. Risks

- **The authors' checkpoints may have seen our test photos.** Their training data is unknown. E5 (own models) gives the clean result.
- **Small test set.** 300 crops, 50 per class, so a 1-point mAP difference is within noise. 3 seeds and reporting std are mandatory.
- **Baselines with their own repos.** AnomalyDiffusion targets PyTorch 1.12 / CUDA 11.3 and needs porting. DefFiller needs GLIGEN weights. Both are trained with default settings, as the paper states.
- **GPU sharing.** The CSM training has priority until it ends (watchdog in `runs/logs/watchdog.py`). Baseline training waits for it.

---

## 10. Decisions

Decided 2026-11-18:
1. **Target metric:** "mAP > 0.5" is mAP@50–95.
2. **E2 on split A:** not now. First compare 1× on both splits (Section 7b) and choose the split.
3. **E3 size:** 900 per setting, 3 seeds, on the chosen split (Section 6).
4. **Board split:** kept as the secondary, stricter result.

Decided 2026-11-19:
5. **Detector input size 640** (Ultralytics default) for every split-A detector run, best epoch still chosen on the select set. Original at 640 scores mAP@50–95 0.497 (paper 0.496), recall 0.936. Choosing the best epoch on the test set explains only 0.010 (0.461). All split-A methods are re-run at 640 with the same synthetic images; the 256 results are kept as a secondary table.

Background (2026-11-18):
- **Detector gap.** Our Original (detector on real data only) scores 0.451 mAP@50–95; the paper's scores 0.496. No generator is involved in that row, so the gap comes from the detector protocol. Candidate causes, each testable with 3 Original runs:
   - best epoch chosen on the test set instead of a separate select set (the Ultralytics default when only train/val exist);
   - input size 256 vs the Ultralytics default 640;
   - how the 300 test crops are cut.

   If one of them closes the gap, all detector runs are repeated with that protocol. The synthetic images stay the same: about 27 runs for split A, 1×.
