# 2026-10-26（二）：加入 Luo 等 2025（DEFectDiffuser）并置于参考文献首位；去掉正文中的英文论文全称。
# 用法：python revise_1026b.py <输入 content.json（revise_1026.py 的输出）> <输出 content.json>
# 文献号先用临时键 [L]（Luo）书写，最后按首次出现顺序整体重新编号；参考文献随之重排。
import sys, json, re

SRC, OUT = sys.argv[1:3]
content = json.load(open(SRC))

LUO_REF = ("LUO W, ZHANG Z, HUANG J, et al. DEFectDiffuser: A Text-Free Framework for Semantically Consistent "
           "PCB Defect Image Generation via Dual Diffusion Alignment[C]//2025 IEEE 2nd International Conference on "
           "Electronics, Communications and Intelligent Science (ECIS). 2025: 1-6.")
LUO = 99            # 临时编号，重编号后为 1

EDITS = [
    # 1.1 首段：Luo 为全文第一处引用 → 新编号 [1]（经三视角核验，按原文 “diverse and intricate morphologies …
    # strict consistency with local background textures, making their detection particularly challenging”）
    ("多余铜箔等多种表面缺陷。已有PCB缺陷检测研究表明",
     f"多余铜箔等多种表面缺陷。这些缺陷形态多样且复杂，并与局部背景纹理紧密关联，使PCB缺陷检测尤为困难^[{LUO}]^。"
     "已有PCB缺陷检测研究表明"),
    # 2.2：去掉英文论文全称，改为“作者等”；Song Y 与上一段 Song J（DDIM）区分
    ("Score-Based Generative Modeling through Stochastic Differential Equations从随机微分方程角度统一描述"
     "score-based生成过程与扩散生成过程，为分析不同噪声水平下的数据分布学习提供了连续时间理论框架^[18]^；"
     "Latent Diffusion Models则将扩散过程转移到潜在空间，",
     "Song Y等从随机微分方程（Stochastic Differential Equation，SDE）角度将基于分数的生成模型与扩散概率模型"
     "纳入统一框架，把离散的噪声尺度推广到连续时间^[18]^；Rombach等提出的潜在扩散模型（Latent Diffusion Models，"
     "LDM）则将扩散过程转移到预训练自编码器的潜在空间，"),
    # 2.3 首段末句：同样去掉两处英文全称（该句与 2.2 重复，改为简称）
    ("Score-Based Generative Modeling through Stochastic Differential Equations和Latent Diffusion Models分别"
     "从连续时间建模和潜在空间建模角度扩展了扩散模型的表达与计算能力^[18,19]^。",
     "基于SDE的连续时间框架^[18]^与LDM^[19]^分别从连续时间建模和潜在空间建模角度扩展了扩散模型的表达与计算能力。"),
    # 2.3：在 CoPaint 段后新增 PCB 专用方法（措辞经三视角核验，限定为“均优于 RePaint、CoPaint 等基线”，
    # 因原文表 I 中真实数据基线的平均 Recall 更高）
    ("而让生成区域与已知区域共同参与优化有助于提高整体一致性。",
     "而让生成区域与已知区域共同参与优化有助于提高整体一致性。\n"
     "针对PCB缺陷图像生成，Luo等提出DEFectDiffuser框架，将基于少量缺陷样本训练的低分辨率缺陷扩散模型与利用大量"
     "无缺陷样本的高分辨率纹理重建模型相结合，并在DDIM采样过程中通过动态优化的权重掩码逐步融合缺陷区域与背景，"
     f"从而在指定区域生成与背景纹理一致的缺陷^[{LUO}]^；实验表明，使用其生成样本训练的模型在缺陷分类与检测任务中"
     "均优于RePaint^[16]^、CoPaint^[17]^等基线方法。"),            # 点名即标注（指导意见）；16/17 为输入文件中的旧编号
    # 2.3 末段：本课题思路与 DEFectDiffuser 的双扩散融合直接相关，须注明
    ("本课题拟参考扩散图像修复中的条件约束思想，将背景信息",
     f"本课题拟参考扩散图像修复中的条件约束思想及DEFectDiffuser的双扩散融合思路^[{LUO}]^，将背景信息"),
    # 2.4：加入 PCB 专用扩散方法后，“现有扩散模型主要面向自然图像”需限定
    ("其二，现有扩散模型主要面向自然图像，难以兼顾",
     "其二，现有扩散模型主要面向自然图像，面向PCB缺陷的专门研究仍较少，通用方法难以兼顾"),
    # 2.4 两处综述性范围引用纳入新文献
    ("提供新的方向^[1-8,10-20]^。", f"提供新的方向^[1-8,10-20,{LUO}]^。"),
    # 用户于 2026-10-26 16:30 在 WPS 中将 1.1 首段末句另起一段，保留该修改
    ("重要研究方向^[1-8]^。图1-1展示PCB电路板整体图像",
     "重要研究方向^[1-8]^。\n图1-1展示PCB电路板整体图像"),
    ("验证其增强价值^[3-7,9-20]^。", f"验证其增强价值^[3-7,9-20,{LUO}]^。"),
    # 3.2.2：1386 = 693 张模板方向 + 693 张人工旋转，出自数据集原论文 Huang & Wei（输入文件中为 [9]，重编号后为 [10]）
    # 原文：“1386 naked PCB images. Half of them are in right orientation as templates … other half are manually rotated
    # to simulate the situation when PCBs are not correctly placed.” 来源是数据集作者自述，故写“Huang等指出”而非“已有研究指出”
    ("已有研究指出，该数据集的1386张图像包括693张原始方向图像和693张旋转图像。",
     "Huang等指出，该数据集的1386张图像中，一半（693张）为与模板方向一致的原始方向图像，另一半为人工旋转的图像，"
     "用于模拟PCB未正确放置的情况^[9]^。"),
    # 3.2.2：[9]（Huang）未提及发布实验室；Xia 等原文 “the PCB defect dataset released by the Open Lab of Peking University”
    # （输入文件中为 [6]，重编号后为 [7]；2.1 已先引，不影响编号顺序）
    ("公开发布的PKU-Market-PCB数据集作为主要实验数据来源", "公开发布的PKU-Market-PCB数据集^[6]^作为主要实验数据来源"),
]

for old, new in EDITS:
    hits = [c for c in content if c["t"] == "p" and old in c["text"]]
    assert len(hits) == 1 and hits[0]["text"].count(old) == 1, f"锚点不唯一或不存在：{old[:30]}（{len(hits)} 处）"
    c = hits[0]
    if "\n" in new:                                  # 新增段落：拆为两个 p 块
        head, tail = new.split("\n", 1)
        before, after = c["text"].split(old, 1)
        c["text"] = before + head
        content.insert(content.index(c) + 1, {"t": "p", "text": tail + after})
    else:
        c["text"] = c["text"].replace(old, new)

refs = [c for c in content if c["t"] == "ref"]
content.insert(content.index(refs[0]), {"t": "ref", "n": LUO, "text": LUO_REF})

CITE = re.compile(r"\^\[([^\]]+)\]\^")


def parse(s):
    nums = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        nums += range(int(a), int(b or a) + 1)
    return nums


def compress(nums):
    nums, out, i = sorted(set(nums)), [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out += [f"{nums[i]}-{nums[j]}"] if j - i >= 2 else [str(n) for n in nums[i:j + 1]]
        i = j + 1
    return ",".join(out)


order = []
for c in content:
    if c["t"] in ("p", "cap"):
        for m in CITE.finditer(c["text"]):
            for n in parse(m.group(1)):
                if n not in order:
                    order.append(n)
refs = [c for c in content if c["t"] == "ref"]
assert sorted(order) == sorted(r["n"] for r in refs), "存在未被引用或不存在的文献"
NEW = {old: i + 1 for i, old in enumerate(order)}
assert NEW[LUO] == 1, "Luo 等 2025 未成为第一处引用"

for c in content:
    if c["t"] in ("p", "cap"):
        c["text"] = CITE.sub(lambda m: "^[" + compress(NEW[n] for n in parse(m.group(1))) + "]^", c["text"])
for r in refs:
    r["n"] = NEW[r["n"]]
start = content.index(refs[0])
content[start:] = sorted(content[start:], key=lambda r: r["n"])
assert not any("Score-Based Generative Modeling" in c.get("text", "") for c in content if c["t"] in ("p", "cap"))

json.dump(content, open(OUT, "w"), ensure_ascii=False, indent=1)
print(f"{len(EDITS)} 处修改；文献 {len(refs)} 篇；重新编号（旧→新）：{ {o if o != LUO else 'Luo': n for o, n in NEW.items() if o != n} }")
