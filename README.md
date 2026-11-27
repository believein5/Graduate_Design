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

## 未包含的内容

体积或许可原因，以下内容不在仓库中，需要自行获取：

- 数据集：PKU-Market-PCB，整理版本 https://huggingface.co/datasets/KeenForgeAI/PKU-Market-PCB-corrected ，放在 `data/PKU-Market-PCB-corrected/`
- 双扩散模型的实现与预训练权重：https://github.com/weihang-luo/CRG-DefectDiffuser ，放在 `third_party/CRG-DefectDiffuser/` 与 `checkpoint/`
- 对比方法：https://github.com/CC-T/DefFiller 、https://github.com/sjtuplayer/anomalydiffusion ，放在 `third_party/`
- YOLO11 权重（`yolo11n.pt`，Ultralytics 自动下载）
- 预处理后的数据、生成图像、检测训练输出（`data/`、`results/`、`runs/`）

## 引用

双扩散模型的结构、协同引导方法与预训练权重来自 CRG-DefectDiffuser；数据集为北京大学智能机器人开放实验室发布的 PKU-Market-PCB。
