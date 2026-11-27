# 中期报告第十二轮（2026-11-26，用户要求）：补充原理与方法介绍（图和公式），并加入技术路线图。
# - 1.1 加图1-1 技术路线（make_route.py --mid，栏标题不写章号）
# - 2.2.2 加回较大缺陷二次缩放图；2.2.3 加回背景块覆盖次数图
# - 2.3.1 扩散模型原理：前向/反向过程示意图、单步加噪、累乘系数、反向分布、训练目标；DDIM 的确定性采样
# - 2.3.3 加回短路缺陷采样过程图
# - 2.4.1 注意力掩码的计算（像素差归一化、Sigmoid 锐化与中心加权、平滑与连通域、滑动平均）与 x_t 的梯度更新
# - 2.7.1 YOLO11n 原理简介；IoU、精确率、召回率、mAP@50-95 的定义
# 图、表、公式在第1、2章内按出现顺序重新编号，正文中的“图/表/式”引用随之更新；参考文献按首次引用重新编号。
# 掩码公式与参数取自 third_party/CRG-DefectDiffuser/guided_diffusion/ddim.py 的 compute_attention_masks 与 confs/pcb_gen.yml
# 用法：python -I update_v12.py content.json   （原文件先备份为 content_v11.json，本脚本总是从 content_v11.json 开始）
import json, re, shutil, sys
from pathlib import Path

F = Path(sys.argv[1])
if not (F.parent / "content_v11.json").exists():
    shutil.copy(F, F.parent / "content_v11.json")
c = json.load(open(F.parent / "content_v11.json", encoding="utf-8"))


def P(t): return {"t": "p", "text": t}
def FIG(name, cap): return [{"t": "fig", "name": name}, {"t": "cap", "num": "", "key": name, "text": cap}]
def EQ(key, tex): return {"t": "eq", "num": "", "key": key, "tex": tex}
def NOTE(items): return {"t": "note", "items": [list(i) for i in items]}


# ---------- 1. 现有图、表、公式改为按键引用 ----------
for k, x in enumerate(c):
    if x["t"] == "cap":
        x["key"] = c[k - 1]["name"] if c[k - 1]["t"] == "fig" else c[k - 2]["name"]
    elif x["t"] == "table":
        x["key"] = "t" + x["num"]
    elif x["t"] == "eq":
        x["key"] = "e" + x["num"]
figkey = {x["num"]: x["key"] for x in c if x["t"] == "cap"}
REFPAT = re.compile(r"图(\d+-\d+)|表(\d+-\d+)|式\((\d+-\d+)\)")


def to_keys(s):
    def f(m):
        if m.group(1):
            return "{图:" + figkey[m.group(1)] + "}"
        if m.group(2):
            return "{表:t" + m.group(2) + "}"
        return "{式:e" + m.group(3) + "}"
    return REFPAT.sub(f, s)


def texts(items):
    for x in items:
        if x["t"] == "ref":
            continue
        if isinstance(x.get("text"), str) and x["t"] != "cap":
            yield x, "text"
        if isinstance(x.get("note"), str):
            yield x, "note"
        if x["t"] == "note":
            for it in x["items"]:
                yield it, 1
        if x["t"] == "table":
            for r in x["rows"]:
                for j in range(len(r)):
                    yield r, j


for obj, key in texts(c):
    obj[key] = to_keys(obj[key])


def at(start):
    hits = [i for i, x in enumerate(c) if x.get("text", "").startswith(start)]
    assert len(hits) == 1, (start, hits)
    return hits[0]


def after_key(key):
    """该图题（或表、公式）之后的位置"""
    return next(i for i, x in enumerate(c) if x.get("key") == key) + 1


# ---------- 2. 1.1 技术路线 ----------
i = at("总体方案采用双扩散结构") + 1
c[i:i] = [P("本课题的技术路线见{图:route_mid}。首先整理数据集，构建缺陷局部块和无缺陷背景块两类训练数据；然后搭建双扩散模型，"
            "在去噪过程中由缺陷先验模型和背景合成模型协同引导生成缺陷图像；最后将生成图像加入检测模型的训练集，评价其对检测性能"
            "的影响，并改进检测模型的训练与推理方式，使检测指标达到任务书要求。")] + FIG("route_mid", "本课题技术路线")

# ---------- 3. 2.2.2 二次缩放图 ----------
i = at("以每个缺陷标注框的中心为基准截取64×64的局部块")
t = c[i]["text"]
a = "缩小后仍放不下的115个较大缺陷，先放大截取窗口，再缩小到64×64。"
assert t.endswith(a)
c[i]["text"] = t[:-len(a)]
j = after_key("fig_pre_overview")
c[j:j] = [P("缩小后仍放不下的115个较大缺陷，其中多余铜箔89个、短路15个、毛刺10个、缺孔1个，先放大截取窗口，使标注框长边占窗口边长的"
            "75%，再缩小到64×64，见{图:fig_pre_rescale}，上行为放大后的截取窗口，下行为得到的局部块。这部分局部块相对原图的缩放比例为"
            "0.17～0.37倍，小于其余局部块的0.5倍。")] + FIG("fig_pre_rescale", "较大缺陷的二次缩放")

# ---------- 4. 2.2.3 覆盖次数图 ----------
i = at("去除缺陷后，在每块电路板上按网格均匀截取256×256的背景块")
c[i:i + 1] = ([P("去除缺陷后，在每块电路板距图像边缘至少32像素的有效区域内划分网格，每格在随机位置取一个256×256的窗口，再把各格轮流"
                 "分配给该电路板的各张图像，共截取3000个背景块。与避开缺陷随机截取相比，这样截取的背景块覆盖电路板各部分的次数更均匀，"
                 "见{图:fig_pre_coverage}，左为避开缺陷随机截取，右为按网格截取，颜色越亮表示覆盖该像素的背景块越多；在中部区域，覆盖次数的"
                 "变异系数由0.096降到0.031。")] + FIG("fig_pre_coverage", "电路板01上背景块的覆盖次数")
              + [P("最后将每个背景块与模板的对应区域逐像素比较，个别因图像错位或细小的未标注缺陷而不合格的背景块，改从同一电路板的其他"
                   "图像截取。{图:fig_pre_free}为一张训练图像去除缺陷后截取的背景块。")])

# ---------- 5. 2.3.1 扩散模型原理 ----------
a, b = at("DDPM的前向过程逐步向图像加入高斯噪声"), at("DDIM与DDPM的训练目标相同")
old_eq1, old_note1 = c[a + 1], c[a + 2]
old_eq2 = c[a + 4]
assert old_eq1["key"] == "e2-1" and old_note1["t"] == "note" and old_eq2["key"] == "e2-2"
old_note1["items"] = [it for it in old_note1["items"] if it[0] != r"\bar{\alpha}_{t}"]
c[a:b] = [
    P("去噪扩散概率模型（DDPM）包括前向加噪和反向去噪两个过程，如{图:op_ddpm}所示^[3]^。前向过程在$T=1000$步内逐步向图像加入少量"
      "高斯噪声，第$t$步的加噪如{式:e_q}所示。"),
    *FIG("op_ddpm", "DDPM的前向加噪与反向去噪过程^[3]^"),
    EQ("e_q", r"q(x_{t}\mid x_{t-1})=\mathrm{N}(x_{t};\sqrt{1-\beta_{t}}x_{t-1},\beta_{t}\mathbf{I})"),
    NOTE([(r"\beta_{t}", "第$t$步加入噪声的方差，采用线性调度，由$10^{-4}$增大到0.02；"), (r"\mathbf{I}", "单位矩阵。")]),
    P("把各步的系数连乘，记为{式:e_abar}，由{式:e_q}逐步代入可得，任意时间步的加噪图像可以由原始图像一步得到，如{式:e2-1}所示。"
      "$t$越大，$\\bar{\\alpha}_{t}$越接近0，第$T$步的加噪图像接近纯噪声。"),
    EQ("e_abar", r"\bar{\alpha}_{t}=\prod_{s=1}^{t}{(1-\beta_{s})}"),
    old_eq1, old_note1,
    P("反向过程从纯噪声出发逐步去噪，每一步的分布由网络参数化为高斯分布，如{式:e_rev}所示。训练时随机选取时间步$t$和噪声"
      "$\\epsilon$，按{式:e2-1}得到加噪图像，网络$\\epsilon_{\\theta}$以$x_{t}$和$t$为输入预测加入的噪声，训练目标为预测噪声与真实"
      "噪声的均方误差，如{式:e_loss}所示^[3]^。本课题使用的模型还同时学习反向过程的方差^[6]^。"),
    EQ("e_rev", r"p_{\theta}(x_{t-1}\mid x_{t})=\mathrm{N}(x_{t-1};\mu_{\theta}(x_{t},t),\Sigma_{\theta}(x_{t},t))"),
    EQ("e_loss", r"L_{\mathrm{simple}}=\mathrm{E}_{x_{0},\epsilon ,t}[\lVert \epsilon -\epsilon_{\theta}(x_{t},t)\rVert ^{2}]"),
    NOTE([(r"\mu_{\theta},\Sigma_{\theta}", "网络给出的反向过程均值与协方差；"), (r"\epsilon_{\theta}", "噪声预测网络（U-Net）；"),
          (r"\mathrm{E}", "对原始图像、噪声和时间步取期望，$t$在1～$T$中均匀采样。")]),
    P("由预测的噪声可以得到对原始图像的估计，如{式:e2-2}所示。"),
    old_eq2,
]
i = at("DDIM与DDPM的训练目标相同")
j = i + 3                                   # 段落、公式、符号说明之后
assert c[i + 1]["key"] == "e2-3" and c[i + 2]["t"] == "note"
c[j:j] = [P("取随机项为0后，DDIM的采样轨迹是确定的：同一初始噪声总得到同一结果。因此既可以从纯噪声出发生成新图像，也可以先把已有图像"
            "加噪到某一时间步再去噪，在保持图像整体结构的同时修改局部内容^[8]^，这是2.4节协同引导的基础。本课题按{式:e2-2}和{式:e2-3}"
            "编写了独立的DDIM采样程序，用于检查每个模型单独学到的内容。")]

# ---------- 6. 2.3.3 采样过程图 ----------
j = after_key("fig_64_samples")
c[j:j] = [P("以短路类为例，{图:fig_64_process}给出了采样过程中模型对$\\hat{x}_{0}$的估计，每行为一个样本，从左到右对应的时间步由999逐步"
            "减小到0。时间步较大时估计接近模糊的平均颜色；时间步在500～350之间，线路、焊盘和缺陷的布局基本确定；此后只有细节和清晰度"
            "变化。可见缺陷结构主要在中等噪声水平下形成，这也是协同引导从中等噪声水平开始去噪的原因。")] + FIG("fig_64_process", "短路缺陷采样过程中的原始图像估计")

# ---------- 7. 2.4.1 掩码计算与梯度更新 ----------
i = at("$L_{\\mathrm{d}}$使窗口内的缺陷区域跟随缺陷先验模型的预测")
c[i]["text"] = ("$L_{\\mathrm{d}}$使窗口内的缺陷区域跟随缺陷先验模型的预测，$L_{\\mathrm{b}}$和$L_{\\mathrm{o}}$使窗口内的背景和窗口外的区域"
                "保持原样。缺陷区域的掩码$A_{\\mathrm{d}}$不是预先给定的，而是在每个去噪步由无缺陷图像与缺陷预测之间的差异计算。先计算窗口内"
                "两者之差，对三个颜色通道取平均并线性归一化到0～1，如{式:e_D}所示；再经Sigmoid函数锐化，并按到窗口中心的距离加权，使窗口"
                "中部的差异更容易被保留，如{式:e_A}所示。")
c[i + 1:i + 1] = [
    EQ("e_D", r"D=\mathrm{norm}(\frac{1}{3}\sum_{k=1}^{3}{\mid r(x_{\mathrm{free}})_{k}-\hat{c}_{0,k}\mid })"),
    EQ("e_A", r"A^{\prime}=\mathrm{min}(1,\mathrm{sigmoid}(\frac{D-0.5}{\tau})(1+w\,\mathrm{exp}(-\frac{\rho^{2}}{2\sigma_{\mathrm{r}}^{2}})))"),
    NOTE([("D", "归一化的像素差，$k$为颜色通道；"), (r"\mathrm{norm}(\cdot )", "按最小值和最大值线性归一化到0～1；"),
          (r"\tau", "锐化温度，取0.3；"), (r"\rho", "像素到窗口中心的归一化距离；"),
          (r"w,\sigma_{\mathrm{r}}", "中心权重与衰减系数，分别取1.0和0.15。")]),
    P("然后对$A^{\\prime}$做高斯平滑（标准差为1像素），以窗口中心区域平均值的0.6倍为阈值二值化，只保留与中心区域相交且面积足够的"
      "连通域，得到当前时间步的掩码$A^{(t)}$。为了避免掩码在相邻时间步之间跳变，再与上一步的掩码做指数滑动平均，如{式:e_ema}所示，"
      "平均后重新归一化到0～1；窗口内背景区域的掩码$A_{\\mathrm{b}}$取$A_{\\mathrm{d}}$小于0.01的部分。"),
    EQ("e_ema", r"A_{\mathrm{d}}^{(t)}=\gamma A_{\mathrm{d}}^{(t+1)}+(1-\gamma)A^{(t)}"),
    NOTE([(r"\gamma", "滑动平均系数，取0.9。")]),
    P("每个去噪步中，先按{式:e_upd}对$x_{t}$做两次梯度下降，使{式:e2-4}的损失减小，再由背景合成模型做一步DDIM更新得到下一时间步的"
      "图像。直接对$x_{t}$做梯度优化的思路与CoPaint^[9]^相近，部分时间步还按RePaint^[10]^的方式重新加噪后再去噪，使缺陷边缘与周围"
      "的线路和铜箔衔接自然。"),
    EQ("e_upd", r"x_{t}\leftarrow x_{t}-\eta \nabla_{x_{t}}L"),
    NOTE([(r"\eta", "引导步长，取0.1。")]),
]

# ---------- 8. 2.7.1 检测模型与评价指标 ----------
i = at("检测训练集由300张真实缺陷图像（每类50张）和50张无缺陷图像组成")
c[i:i] = [P("检测模型采用YOLO11n^[12]^。YOLO系列是单阶段检测器：骨干网络提取多尺度特征，颈部网络融合不同尺度的特征，检测头在步长为8、"
            "16和32的三个特征图上直接预测缺陷的类别和边界框，不需要先生成候选区域，速度快，适合工业在线检测。YOLO11n是其中参数量最小的"
            "版本，约260万个参数。")]
c[i + 1]["text"] = c[i + 1]["text"].replace("检测训练集由", "检测模型输入640，训练200个epoch，批次为32。检测训练集由")
c[i + 2:i + 2] = [
    P("检测结果用精确率$P$、召回率$R$和mAP@50-95评价。预测框与标注框的交并比如{式:e_iou}所示，交并比不低于阈值且类别正确的预测框"
      "记为正确检出，精确率和召回率如{式:e_prec}和{式:e_rec}所示。"),
    EQ("e_iou", r"\mathrm{IoU}=\frac{\mid B_{\mathrm{p}}\cap B_{\mathrm{g}}\mid }{\mid B_{\mathrm{p}}\cup B_{\mathrm{g}}\mid }"),
    EQ("e_prec", r"P=\frac{TP}{TP+FP}"),
    EQ("e_rec", r"R=\frac{TP}{TP+FN}"),
    NOTE([(r"B_{\mathrm{p}},B_{\mathrm{g}}", "预测框与标注框的面积区域；"), ("TP", "正确检出的缺陷数；"), ("FP", "误检数；"),
          ("FN", "漏检数。")]),
    P("按置信度从高到低依次取阈值，可得到精确率-召回率曲线，曲线下的面积为该类的平均精度AP。mAP@50-95在交并比阈值0.50、0.55、…、"
      "0.95下分别计算6类AP的平均值，再对10个阈值取平均，如{式:e_map}所示，同时反映检出能力和框的定位精度；mAP@50只取阈值0.50。"),
    EQ("e_map", r"\mathrm{mAP}_{50-95}=\frac{1}{10}\sum_{j=0}^{9}{\frac{1}{6}\sum_{k=1}^{6}{\mathrm{AP}_{k}(0.5+0.05j)}}"),
    NOTE([(r"\mathrm{AP}_{k}(u)", "第$k$类缺陷在交并比阈值$u$下的平均精度。")]),
]

# ---------- 9. 第1、2章的图、表、公式按顺序编号，替换引用 ----------
num, ch, cnt = {}, 0, {}
for x in c:
    if x["t"] == "h1" and x["num"]:
        ch = int(x["num"]); cnt = {"cap": 0, "table": 0, "eq": 0}
    if x["t"] in ("cap", "table", "eq"):
        if ch in (1, 2):
            cnt[x["t"]] += 1
            x["num"] = f"{ch}-{cnt[x['t']]}"
        num[x["key"]] = x["num"]                        # 第3章及以后的编号不变
KEYPAT = re.compile(r"\{(图|表|式):([^}]+)\}")
for obj, key in texts(c):
    obj[key] = KEYPAT.sub(lambda m: ("式(" + num[m.group(2)] + ")") if m.group(1) == "式" else m.group(1) + num[m.group(2)], obj[key])
for x in c:
    x.pop("key", None)
left = [obj[key] for obj, key in texts(c) if "{图:" in obj[key] or "{表:" in obj[key] or "{式:" in obj[key]]
assert not left, left

# ---------- 10. 参考文献按首次引用重新编号（同 update_v6.py）----------
CITE = re.compile(r"\^\[([\d,\-–]+)\]\^")


def expand(s):
    out = []
    for part in s.split(","):
        if re.search(r"[-–]", part):
            x, y = map(int, re.split(r"[-–]", part)); out += list(range(x, y + 1))
        else:
            out.append(int(part))
    return out


def compress(ns):
    ns, out, k = sorted(ns), [], 0
    while k < len(ns):
        j = k
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        out.append(str(ns[k]) if j == k else (f"{ns[k]},{ns[j]}" if j == k + 1 else f"{ns[k]}-{ns[j]}")); k = j + 1
    return ",".join(out)


allt = [(x, "text") for x in c if x["t"] != "ref" and isinstance(x.get("text"), str)] + list(
    (o, k) for o, k in texts(c) if not (isinstance(o, dict) and k == "text"))
order = []
for obj, key in allt:
    for m in CITE.finditer(obj[key]):
        for n in expand(m.group(1)):
            if n not in order:
                order.append(n)
remap = {o: n for n, o in enumerate(order, 1)}
for obj, key in allt:
    obj[key] = CITE.sub(lambda m: f"^[{compress([remap[n] for n in expand(m.group(1))])}]^", obj[key])
refs = [x for x in c if x["t"] == "ref"]
start = c.index(refs[0])
byn = {r["n"]: r for r in refs}
assert set(order) == set(byn), (set(byn) - set(order))
c[start:start + len(refs)] = [{"t": "ref", "n": remap[o], "text": byn[o]["text"]} for o in order]
print("refs remap:", {k: v for k, v in remap.items() if k != v})
print("figures:", [x["num"] for x in c if x["t"] == "cap"])
print("equations:", [x["num"] for x in c if x["t"] == "eq"])
json.dump(c, open(F, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(c), "items")
