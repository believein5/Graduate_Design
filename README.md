# 基于扩散模型的印刷电路板缺陷图像生成方法研究

本科毕业论文（设计）的代码、实验记录与报告。用背景合成模型（256×256）和缺陷先验模型（64×64）两个扩散模型在去噪过程中协同引导，
在无缺陷 PCB 图像的指定位置生成缺陷，并用 YOLO11n 检测实验评价生成数据的作用。

Code, experiment logs and reports for a bachelor's thesis on diffusion-based PCB defect image generation.

## 主要结果（测试集，mAP@50-95，3 个随机种子的均值）

| 每类真实图像 | 只用真实数据 | 加入 300 张生成图像 |
|---|---|---|
| 10 张 | 0.388 | 0.447 |
| 10 张，90° 旋转增强 | 0.437 | 0.477 |
| 50 张，90° 旋转增强 | 0.516（P 0.965，R 0.960） | 0.510 |

详细结果与分析见论文第 4、5 章和 `docs/superpowers/plans/2026-11-20-two-day-improvement-search.md`。

## 目录

| 目录 | 内容 |
|---|---|
| `code/experiments/` | 数据划分与检测数据构建（`build_detection_*.py`）、生成（`generate.py`）、检测训练（`detect.py`、`det_variants.py`、`aug_rot90.py`）、多模型平均（`eval_ensemble.py`）、生成质量（`gen_quality.py`）、缺陷先验模型训练（`train_defect.py`）、GPU 任务队列（`gpu_queue.py`） |
| `code/experiments/baselines/` | 对比方法：ControlNet、DefFiller、AnomalyDiffusion |
| `code/*.ipynb` | 预处理、训练、采样的分步说明 |
| `docs/superpowers/plans/` | 各阶段的实验计划与记录 |
| `docs/midterm_rebuild/`、`docs/thesis_build/`、`docs/kaiti_rebuild/` | 中期报告、毕业论文、开题报告的生成脚本（内容 → .docx） |
| `report/` | 开题报告、中期报告、毕业论文（.docx / .pdf） |
| `tools/libreoffice/` | 把 .docx 导出为预览 PDF 的脚本（LibreOffice 本身不包含） |

## 运行步骤

### 1. 环境

两个 conda 环境：主环境（Python 3.11）用于预处理、生成、检测和报告脚本；对比方法环境（Python 3.10）只用于 `code/experiments/baselines/`。
实验使用 RTX 4090（24 GB）、CUDA 13.0。

```bash
conda create -n crg-defect python=3.11 -y && conda activate crg-defect
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r env/requirements-main.txt

conda create -n baselines python=3.10 -y && conda activate baselines   # 只在运行对比方法时需要
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r env/requirements-baselines.txt
```

`gpu_queue.py` 用环境变量 `CRG_PY`、`BASELINES_PY` 指定两个环境的 Python（默认分别为当前 Python 和 `python`）。

### 2. 数据与第三方代码

按“未包含的内容”一节，把数据集放到 `data/PKU-Market-PCB-corrected/`，把双扩散模型的实现克隆到 `third_party/CRG-DefectDiffuser/`，
两个预训练权重 `checkpoint-64-defect.pt`、`checkpoint-256-defect-free.pt` 放到 `checkpoint/`。

### 3. 流程（在 `code/experiments/` 下运行，除第 1 步外）

| 步骤 | 命令 | 输出 |
|---|---|---|
| 1 预处理：0.5 倍缩放、缺陷局部块、模板去除缺陷后的背景块 | 运行 `code/preprocess_walkthrough.ipynb` | `data/pcb_processed/` |
| 2 检测数据：按照片划分（训练 / 选择 / 测试） | `python build_detection_data_A.py` | `data/detection_A/` |
| 3 每类 10～60 张的训练子集；无缺陷图替换后的训练集 | `python build_detection_subset.py --n 10 20 30 40`；`python build_detection_n60.py`；`python build_detection_cleanfix.py` | `data/detection_A_n*/`、`data/detection_A_cleanfix/` |
| 4 生成任务（背景、类别、位置，所有方法共用） | `python make_jobs.py --out jobs_A --photo-split ../../data/detection_A/photo_split.csv` | `data/synthetic/jobs_A/` |
| 5 协同引导生成 | `python generate.py --method crg --jobs jobs_A --tag _A --start 0 --end 300` | `data/synthetic/crg_A/` |
| 6 检测训练与评估（3 个随机种子） | `python detect.py --config crg_1x --split A --proto img640 --seed 0`；只用真实数据时 `--config original` | `results/experiments/detection_A_img640/` |
| 7 90° 旋转增强 | `python detect.py --config original --split Aclean --variant rot90 --seed 0` | `results/experiments/detection_Aclean_vrot90/` |
| 8 多模型平均 | `python eval_ensemble.py --runs <结果目录>... --name <名字>` | `results/experiments/ensemble/` |
| 9 生成质量（KID、LPIPS） | `python gen_quality.py crg_A` | `results/experiments/gen_quality/` |
| 10 汇总第 5 章的比较 | `python -I summarize_p6.py` | 输出到终端 |

全部实验的任务列表见 `gpu_queue.py`（按顺序排队、按显存并行）。设置一律按选择集的 mAP@50-95 选取，测试集只用于报告选定设置的结果。
`results/` 中只提交了每次训练的 `metrics.json`、`train/results.csv` 以及集成、生成质量和扫描的结果文件，论文表格中的数字都可以由这些文件核对。

## 未包含的内容

体积或许可原因，以下内容不在仓库中，需要自行获取：

- 数据集：PKU-Market-PCB，整理版本 https://huggingface.co/datasets/KeenForgeAI/PKU-Market-PCB-corrected ，放在 `data/PKU-Market-PCB-corrected/`
- 双扩散模型的实现与预训练权重：https://github.com/weihang-luo/CRG-DefectDiffuser ，放在 `third_party/CRG-DefectDiffuser/` 与 `checkpoint/`
- 对比方法：https://github.com/CC-T/DefFiller 、https://github.com/sjtuplayer/anomalydiffusion ，放在 `third_party/`
- YOLO11 权重（`yolo11n.pt`，Ultralytics 自动下载）
- 预处理后的数据、生成图像、检测模型权重和训练过程的图像（`data/`、`runs/`、`results/` 中的图像与权重）

## 许可

本仓库的代码采用 MIT 许可（`LICENSE`）。`report/` 中的报告与论文、`docs/` 中报告的正文内容不在此许可范围内，版权归作者所有。第三方代码、数据集和预训练权重遵循各自的许可。

## 引用

双扩散模型的结构、协同引导方法与预训练权重来自 CRG-DefectDiffuser；数据集为北京大学智能机器人开放实验室发布的 PKU-Market-PCB。
