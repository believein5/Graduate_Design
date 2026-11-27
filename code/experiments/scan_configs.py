"""CRG 设置扫描（2026-11-19）：以发布的 confs/pcb_gen.yml 为中心，每个设置在划分 A 的前 60 个任务（每类 10 个）上生成
（generate.py --method crg --scan <名字>），再用只在真实训练数据上训练的 Original 检测器打分（score_proxy.py）。
值是对配置的覆盖项；"_code" 是对 ddim.py 源码的修改：w_def / w_bg / w_out 为三项损失权重的倍数，fix10 为 % 10 修正。
"""
import copy
import json


def _merge(*ds):
    out = {}
    for d in ds:
        for k, v in d.items():
            out[k] = _merge(out.get(k, {}), v) if isinstance(v, dict) else v
    return out


def _t(v):
    return {"ddim": {"schedule_params": {"start_step": v}}}


def _af(**k):
    return {"attention_features": k}


def _opt(**k):
    return {"optimize_xt": k}


def _g(v):
    return {"g_scha": {"start": v, "end": v}}


def _code(**k):
    return {"_code": k}


SCAN = {
    # 单因素：每次只改一项
    "t20": _t(20), "t50": _t(50), "t80": _t(80),                                  # T_start（发布值 30）
    "thr04": _af(threshold_factor=0.4), "thr08": _af(threshold_factor=0.8),       # 掩码阈值（0.6）
    "temp01": _af(temperature=0.1), "temp05": _af(temperature=0.5),               # 掩码锐度（0.3）
    "blur05": _af(blur_sigma=0.5), "blur20": _af(blur_sigma=2.0),                 # 高斯平滑（1.0）
    "beta07": _af(accumulation_beta=0.7), "beta097": _af(accumulation_beta=0.97), # EMA 系数（0.9）
    "wdef10": _code(w_def=10), "wdef01": _code(w_def=0.1),                        # 缺陷项权重
    "wbg10": _code(w_bg=10, w_out=10), "wbg01": _code(w_bg=0.1, w_out=0.1),       # 背景项权重（窗口内外一起）
    "lr005": _opt(lr_xt=0.05), "lr02": _opt(lr_xt=0.2),                           # 引导步长（0.1）
    "g1": _g(1), "g4": _g(4),                                                     # 每个去噪步的梯度步数（2）
    "fix10": _code(fix10=True),                                                   # 只做 % 10 修正
}
_seen = {json.dumps(v, sort_keys=True) for v in SCAN.values()} | {json.dumps({}, sort_keys=True)}


def _add(name, cfg):
    key = json.dumps(cfg, sort_keys=True)
    if key not in _seen:
        SCAN[name] = cfg; _seen.add(key)


# 小网格 1：T_start × 阈值 × 缺陷权重（只列出上面没有的组合）
for t in (20, 30, 50):
    for thr in (0.4, 0.6, 0.8):
        for wd in (1, 10):
            parts = [_t(t)] if t != 30 else []
            parts += [_af(threshold_factor=thr)] if thr != 0.6 else []
            parts += [_code(w_def=wd)] if wd != 1 else []
            _add(f"t{t}_thr{int(thr * 10):02d}_wd{wd}", _merge(*parts))
# 小网格 2：引导步长 × 梯度步数；掩码锐度 × 阈值
for lr in (0.05, 0.2):
    for g in (1, 4):
        _add(f"lr{lr}_g{g}", _merge(_opt(lr_xt=lr), _g(g)))
for temp in (0.1, 0.5):
    for thr in (0.4, 0.8):
        _add(f"temp{temp}_thr{thr}", _af(temperature=temp, threshold_factor=thr))


# 消融（2026-11-25）：每次去掉协同引导的一个组成部分，在 300 个任务上生成（generate.py --scan abl_*）。
# 不放进 SCAN：gpu_queue.py 会为 SCAN 中每个设置建 60 个任务的扫描作业。去掉注意力掩码引导的版本即 ECICS（ecics_A）
ABLATION = {
    "abl_nodef": _code(w_def=0),                 # 去掉缺陷损失 L_d：窗口内不再跟随缺陷先验模型
    "abl_nobg": _code(w_bg=0),                   # 去掉窗口内背景损失 L_b
    "abl_noout": _code(w_out=0),                 # 去掉窗口外背景损失 L_o
    "abl_noema": _af(accumulation_beta=0.0),     # 掩码不做滑动平均，每步只用当前掩码
}


def get(name):
    return copy.deepcopy(SCAN[name] if name in SCAN else ABLATION[name])


if __name__ == "__main__":
    for k, v in SCAN.items():
        print(f"{k:18s} {json.dumps(v)}")
    print(len(SCAN), "settings")
