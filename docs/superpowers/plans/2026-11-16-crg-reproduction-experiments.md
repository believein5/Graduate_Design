# Reproduction experiments with the authors' checkpoints

Date: 2026-11-16 · Status: **approved 2026-11-16** — D1 CRG protocol · D2 board split · D3 CoPaint included · D4 start now, alongside training

Goal: reproduce the downstream experiments of CRG-DefectDiffuser (TCSVT 2026) and DEFectDiffuser (ECICS 2025) with the authors' released checkpoints: comparison with baselines, ablation of the guidance components, augmentation scales 1×/2×/3×, analysis, and tuning of the generation. The same pipeline must later run with our own checkpoints by changing two paths.

## 1. The core idea

Generated defect images are only useful if they help a **defect detector** trained on them. So every generation method is judged the same way:

```
real training crops (300 + 50 defect-free)  ─┐
                                              ├─► train YOLOv11n ─► test on REAL held-out crops ─► mAP / Recall
synthetic defect crops (0 / 300 / 600 / 900) ─┘
```

The detector is a supervised object detector (box + class), not an unsupervised anomaly detector. Test images are always real and never seen by any generator or detector during training or tuning.

## 2. What the papers did

| | CRG (TCSVT 2026) — matches the checkpoints | DEFectDiffuser (ECICS 2025) — thesis title |
|---|---|---|
| Real training set | 300 defect images (256×256) + 50 defect-free negatives | 300 samples, 50 per class |
| Test set | 300 images, 50 per class | real images (size not stated) |
| Detector | YOLOv11n, 200 epochs, batch 32 | YOLOv11 |
| Detection metrics | Recall, mAP@50–95, per class + overall | AP50, Recall, AUC-PR, per class |
| Augmentation scales | 1×, 2×, 3× synthetic (balanced) | 50 / 100 / 150 per class |
| Baselines | DDIM fine-tune, RePaint, CoPaint, ControlNet, AnomalyDiffusion, DefFiller | Original, RePaint, CoPaint |
| Ablation | cumulative CGM components (base, +sigmoid, +Gaussian smoothing, +connected components, +EMA); detector trained **only** on 1,800 synthetic per config | none |
| Quality metrics | KID×10³, LPIPS, inference time | KID×10³ per class, ResNet50 classification |
| Generation | DDIM 250 steps, T_start = 30, λ_defect 1.0, λ_bg 0.1, λ_reg 0.01, N_opt 2 | DDIM, Gaussian mask |

Not stated anywhere: how synthetic images get their bounding boxes, where defects are placed, the train/test split level.

## 3. Our setup

**Real data for the detector** (new, `data/detection/`), all at 0.5×, 256×256:
- `train`: 300 crops from boards 01, 04–09 that contain defects, 50 per main class (all boxes inside the crop labelled; boxes cut by the border kept if ≥ 50 % visible) + 50 defect-free crops from `free_256/`.
- `select`: 300 crops (50 per class) from the selection half of boards 10–12 → used only for tuning (Phase 7).
- `test`: 300 crops (50 per class) from the test half of boards 10–12 → every reported number.

**Synthetic data** (`data/synthetic/<method>/`): each image is a defect-free background with one generated defect.
- Background: a crop from `free_256/` (training boards, cleaned).
- Placement: a real label position **of the same board at the same coordinates**, taken from another training photo. All photos of a board are aligned copies, so the defect lands exactly where that defect type occurs on that board (a short between real traces, a missing hole on a real pad).
- Class balance: equal number per class; 900 per method, so 1× / 2× / 3× are nested subsets (300 ⊂ 600 ⊂ 900).
- Bounding box: from the final guidance mask of the sampler (largest connected component), falling back to the real box size of the placement if the mask is empty. Checked visually on a sample before mass generation.

**Methods**

| Method | What it is | Cost per image |
|---|---|---|
| Original | no synthetic data | — |
| Copy-paste | real defect patch from the training set pasted at the placement | ~0 |
| DPM-paste | 64 model sample pasted, no background guidance (ablation of the dual-model idea) | <1 s |
| RePaint | authors' 256 model, inpainting the 64×64 window, 2 resampling steps | ~9 s |
| CoPaint | authors' 256 model, Bayesian inpainting | ~140 s (paper) — 1× only |
| ECICS loss | CRG code with attention off: Eq. 4 loss of DEFectDiffuser | ~16 s |
| **CRG (full)** | repo `sample_generation.py`, authors' settings | ~16 s (measured) |
| ControlNet, AnomalyDiffusion, DefFiller | separate repos, each needs its own training | **out of scope** (cite paper numbers) |

**Ablation** (CRG paper Table III style, detector trained on synthetic only, 300 per config instead of 1,800 to fit the budget): base (no CGM) → +sigmoid → +Gaussian smoothing → +connected components → +EMA (= full). Needs small switches in `ddim.py` (in our own copy, not in `third_party/`).

**Detector protocol:** YOLOv11n (ultralytics), 200 epochs, batch 32, imgsz 256, **3 seeds per configuration** (300 test images give noisy mAP; report mean ± std). Metrics: mAP@50–95, mAP@50, Recall, AUC-PR, per class and overall.

**Quality metrics:** KID×10³ (torch-fidelity) and LPIPS against real `test` crops, per class; NN ratio for copying; time per image.

## 4. Phases

| Phase | Work | GPU | Output | Gate |
|---|---|---|---|---|
| P0 Setup | install `ultralytics`, `lpips`; build `data/detection/` (train/select/test, YOLO format); build the placement list | none | dataset + label previews | labels look right |
| P1 Pilot | 12 CRG images (2 per class) with the authors' checkpoints; box extraction; 1 YOLO run on Original | small | timings, box check, baseline mAP | boxes fit defects; time per image known |
| P2 Main generation | CRG full, 900 images | ~4 h | `synthetic/crg/` | visual spot check |
| P3 Baselines | copy-paste 900, DPM-paste 900, RePaint 900, ECICS loss 900, CoPaint 300 | ~2.5 h + ~4 h + ~12 h | `synthetic/<method>/` | |
| P4 Detection | Original + each method at 1×/2×/3× (CoPaint 1×), 3 seeds | ~16 configs × 3 | `results/detection/` | |
| P5 Ablation | 4 extra configs × 300, synthetic-only detectors, 3 seeds | ~5.5 h + detectors | `results/ablation/` | |
| P6 Quality | KID, LPIPS, NN ratio, time per method | <1 h | `results/quality/` | |
| P7 Optimisation | tune on `select` only: T_start (20/30/50), λ weights (code 1e5/1e4/5e4 vs paper 1/0.1/0.01), mask temperature, fix `t[0] / 10` → `% 10`, scale of spurious copper; re-run best on `test` | ~1 h per variant (100 images) | best config | must beat full CRG on `select` |
| P8 Analysis | tables and figures for the thesis; HTML report section | none | report | |
| P9 (later) | rerun P2–P6 with our own checkpoints when training finishes | | | |

Each phase writes its config and results to disk and can be resumed.

## 5. Budget and the running training

Our 64 and 256 training runs use the GPU until about Wednesday morning. Generation and detector training on the same GPU make both run slower (roughly 1.5–2× each). Total for P2–P7 alone: about **30–40 GPU hours** (CoPaint is the largest single item). Two idle notebook kernels hold ~8 GB of GPU memory and should be closed.

## 6. Known caveats to report

- The authors' checkpoints were trained on unknown boards, possibly including boards 10–12. Synthetic data from them may carry test-board appearance; all methods using their 256 model share this, copy-paste does not. Repeat with our own checkpoints (P9) for a clean result.
- Our split is by board (stricter than the paper's image split), so absolute mAP will be lower than the paper's; compare relative gains.
- Boxes of synthetic defects are our choice (mask-derived); the papers do not say.
- Baselines needing separate repos (ControlNet, AnomalyDiffusion, DefFiller) are not reproduced.

## 7. Decisions needed before P0

- **D1 Main protocol:** CRG (recommended; same detector runs also give the ECICS metrics) or ECICS.
- **D2 Split:** board-level (recommended) or the paper's image-level split for closer comparability (or both).
- **D3 Baselines:** as in the table above (recommended), or add/remove CoPaint (12 h).
- **D4 GPU:** start now alongside training (both slower) or start P0–P1 now and the heavy phases after the 64 run finishes (~19:30 today).


## Amendment 2026-11-17 · Split A and the full baseline set (1× only)

Decisions: split **A** first (paper-style, by photo); baselines **as in the paper**; 256 training **not paused**.

**Split A (`data/detection_A/`).** Photos of all 10 boards are split at photo level. Test: 300 crops, 50 per class, from a random set of photos; train: 300 + 50 crops (as before) from the remaining photos; select: 300 crops from a third disjoint set of photos (best epoch only). Test photos never supply training crops, but their boards are seen in training, as in the paper. Synthetic data is unchanged (generated on the training boards 01, 04–09; reused from `data/synthetic/`).

**Methods at 1×, 3 seeds:** original · DDIM fine-tune (= our dpmpaste: the authors' 64 model, DDIM, pasted patch; matches the paper's "Fine-tuned Patch") · RePaint · CoPaint (approximation) · ControlNet · AnomalyDiffusion · DefFiller · CRG; extra references: copypaste, ECICS, crgwin.

**New baselines, each trained on the same 300 real training crops (50 per class):**

| Method | Base model | Condition | Output |
|---|---|---|---|
| ControlNet | Stable Diffusion 1.5 + ControlNet, fine-tuned | text "a {class} defect on a printed circuit board" + structural map of the defect-free background (seg/edge) | full 256 image |
| AnomalyDiffusion | official repo, LDM text2img-large | defect-free reference + mask + label | edited reference |
| DefFiller | official repo, GLIGEN-sem | mask + label | full image from scratch |

Masks and positions come from the same 300 jobs as the other methods. Each new method gets one training run with the repos' default settings (as the CRG paper states for its baselines); no tuning on the test split.

Order: split A + existing methods first (results within ~1 day); the three new baselines run after, one at a time, sharing the GPU with the 256 training.
