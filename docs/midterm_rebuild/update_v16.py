# 中期报告第十六轮（2026-11-27）：更正掩码滑动平均系数与引导步长的说明，与实际运行的配置一致。
# confs/pcb_gen.yml 中 adaptive_beta: true、beta_min: 0.5，ddim.py 按 γ_t = 0.5 + 0.4·t/250 计算（t 为 250 步序列中的序号），
# 在使用的时间步 t ≤ 29 内为 0.50～0.55，而不是 0.9；引导步长 lr_xt 初值 0.1，随时间步调整（smart decay 与每步 ×1.012），
# 损失增大时乘以 0.8 重新更新（use_adaptive_lr_xt）。
# 用法：python -I update_v16.py content.json   （原文件先备份为 content_v15.json，本脚本总是从 content_v15.json 开始）
import json, shutil, sys
from pathlib import Path

F = Path(sys.argv[1])
if not (F.parent / "content_v15.json").exists():
    shutil.copy(F, F.parent / "content_v15.json")
c = json.load(open(F.parent / "content_v15.json", encoding="utf-8"))

eq = [x for x in c if x["t"] == "eq" and x["tex"].startswith(r"A_{\mathrm{d}}^{(t)}=\gamma")]
assert len(eq) == 1
eq[0]["tex"] = r"A_{\mathrm{d}}^{(t)}=\gamma_{t}A_{\mathrm{d}}^{(t+1)}+(1-\gamma_{t})A^{(t)}"
n = [x for x in c if x["t"] == "note" and x["items"] == [["\\gamma", "滑动平均系数，取0.9。"]]]
assert len(n) == 1
n[0]["items"] = [["\\gamma_{t}", "滑动平均系数，$\\gamma_{t}=0.5+0.4t/250$，在使用的时间步范围内为0.50～0.55。"]]
eu = [x for x in c if x["t"] == "eq" and x["tex"] == r"x_{t}\leftarrow x_{t}-\eta \nabla_{x_{t}}L"]
assert len(eu) == 1
eu[0]["tex"] = r"x_{t}\leftarrow x_{t}-\eta_{t}\nabla_{x_{t}}L"
n = [x for x in c if x["t"] == "note" and x["items"] == [["\\eta", "引导步长，取0.1。"]]]
assert len(n) == 1
n[0]["items"] = [["\\eta_{t}", "引导步长，初值为0.1，随时间步调整；若更新后损失增大，步长乘以0.8后重新更新。"]]
json.dump(c, open(F, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(c), "items")
