# CRG-DefectDiffuser: Repository Report

- Repo: https://github.com/weihang-luo/CRG-DefectDiffuser (MIT; some files inherited from RePaint/Huawei are CC BY-NC-SA 4.0)
- Local copy: `Graduate_Design/third_party/CRG-DefectDiffuser/` (commit `724199e`, 2026-06-14)
- Paper: Luo et al., *Collaborative Refinement Guidance for High-Fidelity Defect Image Synthesis*, IEEE TCSVT 2026. It is the journal follow-up to *DEFectDiffuser* (ECICS 2025), the paper being reproduced.
- Status: the code has been read but not run. The checkpoints are not downloaded.

## 1. What it does

The repo contains only the generation side. It takes a defect-free 256×256 PCB image and a list of `(class, x, y)` points, and paints one realistic defect of that class at each point. The background is left unchanged.

**Input** (`data/sample/`): a defect-free image plus a `.txt` with the same name, one line per defect: `class_id cx cy`, normalized like YOLO but with no width or height. The bundled sample has 6 lines, one per class.

**Output** (`images/`):

| Folder       | Contents                                 |
| ------------ | ---------------------------------------- |
| `samples/` | full 256×256 image, one defect per file |
| `crop/`    | the 64×64 defect patch                  |
| `grid/`    | original / noised / result side by side  |
| `temp/`    | attention-mask visualizations            |
| log          | loss values per step                     |

Finished `(image, class, x, y)` combinations are skipped on re-run. The script works them out from the output filenames (`sample_generation.py:166`).

## 2. The two models

Both are OpenAI guided-diffusion UNets: 1000 steps, linear β, ε-prediction, `learn_sigma: true`.

|                                     | Contextual Synthesis Model        | Defect Prior Model                       |
| ----------------------------------- | --------------------------------- | ---------------------------------------- |
| Checkpoint                          | `checkpoint-256-defect-free.pt` | `checkpoint-64-defect.pt`              |
| Trained on                          | defect-free PCB crops, 256×256   | defect patches, 64×64                   |
| Conditioning                        | none                              | defect class (6, from `data/pcb.json`) |
| `num_channels` / `channel_mult` | 128 / 1,1,2,3,4,4                 | 64 / 1,2,3,4                             |
| Training batch size                 | 8                                 | 64                                       |
| Role                                | keeps the board realistic         | decides what the defect looks like       |

The checkpoints are not in the repo. They are Google Drive links in the README.

## 3. How one defect is generated

### Setup (`sample_generation.py:430-477`)

1. Place a 64×64 window centered on `(cx, cy)`, clamped inside the image.
2. Noise the whole defect-free image to step 29 of 250 (about t≈116 of 1000). The board stays recognizable. This is SDEdit-style, not starting from pure noise.
3. Cut the same window from the noised image. It is the starting point for the 64 defect model.

### Each denoising step (`A_DDIMSampler.p_sample`, `guided_diffusion/ddim.py:443`)

1. The 256 model predicts the clean full image `x̂₀` from `x_t`.
2. The 64 model predicts a clean defect patch `ĉ₀` from `crop_t`, given the class label.
3. Compute the attention mask (see below).
4. Take two gradient-descent steps on `x_t` (CoPaint-style), minimizing:

   ```
   L = 1e5 · mean(A_def ⊙ (crop(x̂₀) − ĉ₀)²)           # defect area: follow the defect model
     + 1e4 · mean(A_bg  ⊙ (crop(x̂₀) − crop(x_free))²)  # rest of the window: match the clean board
     + 5e4 · mean((1−window) ⊙ (x̂₀ − x_free)²)        # outside the window: match the clean board
     + 1e-4 · ‖x_t − x_t_original‖²                    # stay close to the original x_t
   ```

   A backtracking line search multiplies the learning rate by 0.8 and retries if the loss increases (`ddim.py:766-803`).
5. Take a DDIM step with σ=0 for both `x_t` and `crop_t`.

Only the 256 image is optimized. With `lr_crop: false` the crop's learning rate is 0, so the defect model denoises freely and the 256 image is pulled toward it.

### Attention mask (`compute_attention_masks`, `ddim.py:25`)

This is the "collaborative refinement" part of the paper title.

- Starts from `|clean background crop − defect prediction|`, normalized and sharpened with a sigmoid (`temperature`).
- Adds a Gaussian center boost and a blur.
- Keeps only connected regions that touch the center.
- Applies an EMA across timesteps (`accumulation_beta`, which decreases as t falls).
- The background mask is wherever the defect mask is below 0.01.

The mask is not fixed. It follows wherever the defect model's output differs from the board.

### Schedule (`new_scheduler.py: ddim_repaint_timesteps`)

These are RePaint-style time-travel steps. I ran the function with the repo config, and it gives 110 steps:

- 29→10, then back up to 20 and down again (twice).
- Then 10→0, with two more jumps back to 10.
- On each jump back the image is re-noised by one step (`_undo`), after one extra optimization pass.

## 4. Comparison with the ECICS DEFectDiffuser paper

|            | ECICS DEFectDiffuser                                      | This repo (TCSVT CRG)                                  |
| ---------- | --------------------------------------------------------- | ------------------------------------------------------ |
| Models     | 64 class-conditional defect model + 256 defect-free model | same                                                   |
| Sampler    | DDIM, about 50 steps from `x_T` (Fig. 1)                | DDIM σ=0, 250-step respacing, starts at step 29       |
| Loss       | Eq. 6: defect term + Gaussian-masked background term      | 3 terms with attention masks                           |
| Mask       | fixed Gaussian `m_t` that shrinks over time (Eq. 5)     | from the pixel difference, plus a center prior and EMA |
| Refinement | R repeats of k steps                                      | RePaint time travel (jump 10, 3 passes)                |

The ECICS loss is already in the code. With `attention_features.enabled: false`, `loss_fn` uses
`10·MSE(crop(x̂₀), ĉ₀) + 0.06·MSE_masked(x̂₀, x_free)` (`ddim.py:476-480`). That is Eq. 4 with λ_defect=10 and λ_normal=0.06; the paper never states these values. Multiplying the background term by the Gaussian `m_t` from Eq. 5 gives Eq. 6.

## 5. Training: the pieces exist, but there is no script

- `guided_diffusion/train_util.py` has a working `TrainLoop` (AMP, EMA, checkpointing, optional DDP), but no `train.py` calls it. You would write roughly 40 lines: build the model and diffusion with `script_util`, then call `load_data(...)` and `TrainLoop(...).run_loop()`.
- `load_data` (`guided_diffusion/image_datasets.py`):
  - It only finds `**/*.jpg`.
  - The class is the filename prefix before the first `_`, and it must match a `pcb.json` key (e.g. `missinghole_0001.jpg`).
  - Larger images are shrunk with `thumbnail`, not cropped, so save patches at exactly 64 or 256.
- `confs/train-64.yml` has leftover paths (`data/lsd.json`, `checkpoint/laether_64_cond/`) from a leather dataset experiment. Change them to `data/pcb.json` and your own folder, and clear `resume_checkpoint`.
- Neither the paper nor the configs state the number of training steps.

## 6. Issues found in the code

- `n_samples` and `n_iter` (in the README) are never read. `batch_size = 1` is hard-coded (`sample_generation.py:315`), so you always get one image per label line. Use a different `seed` to get more variations.
- `num_iteration_optimize_xt: 4` is ignored. The actual number of gradient steps per step comes from `g_scha` (start=end=2), which gives 2. `gd_mask_radius` and `max_iter` are also unused.
- `if t[0] / 10 == 0` (`ddim.py:848`) is true only at t=0. It was probably meant to be `% 10`, which would resync the crop from the large image every 10 steps.
- `lr_xt_decay: 1.012` multiplies the learning rate on every step, so the rate grows.
- `from tkinter import Image` (`ddim.py:5`) is unused and fails on Python builds without Tk.
- `Default_Conf.__init__` is `pass`, which skips `NoneDict`'s setup (`conf_mgt/conf_base.py:58`). `conf.key` still returns `None` for missing keys, but `conf['key']` raises `KeyError`.
- `conf_mgt/conf_base.py` is leftover from RePaint. `get_dataloader`, `eval_imswrite` and `get_default_eval_name` are unused.

## 7. How to run generation

1. `pip install -r requirements.txt`. `mpi4py` is listed but never imported, so skip it if it fails to install.
2. Download the two checkpoints into `checkpoint/`.
3. From the repo folder:
   ```bash
   cd Graduate_Design/third_party/CRG-DefectDiffuser
   python sample_generation.py --conf_path confs/pcb_gen.yml
   ```

   This runs on the bundled sample and produces 6 images. It falls back to CPU without CUDA, which is very slow: each defect takes 110 steps × about 2–3 forward and backward passes of the 256 UNet.

## 8. Useful ablations in `confs/pcb_gen.yml`

| Change                                         | What it shows                    |
| ---------------------------------------------- | -------------------------------- |
| `attention_features.enabled: false`          | the ECICS Eq. 4 loss             |
| `ddim.schedule_params.use_timetravel: false` | effect of RePaint re-noising     |
| `start_step` 30 → 100 → 200                | more noise and more freedom      |
| `g_scha` start/end 2 → 0                    | no guidance, the 256 model alone |
| `debug: true` + `temp/` images             | how the mask forms over time     |
