"""后台运行全部生成与检测任务（可中断、可重复运行：已完成的部分会跳过）

两个工作线程并行：
  生成线程：按 300 个一轮生成各方法（先让所有方法的 1× 可用），消融配置只生成前 300 个
  检测线程：数据就绪的配置依次训练 YOLOv11n（3 个 seed）
两者在启动每个任务前都会等待 GPU 空闲显存足够，并在 256 模型训练刚启动（64 模型训练结束、256 训练
尚未出第一条日志）期间暂停，避免与正在进行的扩散模型训练争抢显存导致其崩溃。
日志：results/experiments/pipeline.log
"""
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from common import RES, ROOT, SYN

PY = sys.executable
RESERVE_MIB = 3000                 # 任务启动后，GPU 上仍须为扩散训练留出的空闲显存
GEN_MIB, DET_MIB = 4500, 1600      # 实测：生成约 4.3 GB，检测约 1.5 GB
PAUSE = RES / "pipeline.pause"     # 存在时不启动新的 GPU 任务（看门狗在重启训练时使用）
HERE = Path(__file__).parent
LOG = RES / "pipeline.log"
MAIN = ["copypaste", "dpmpaste", "repaint", "crg", "ecics", "copaint"]
ABLATION = ["abl_sig", "abl_gs", "abl_ccf"]
SEEDS = [0, 1, 2]
lock = threading.Lock()


def log(msg):
    with lock:
        with open(LOG, "a") as f:
            f.write(time.strftime("%a %H:%M:%S ") + msg + "\n")


def free_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def diffusion_training_starting():
    """256 模型训练进程在运行、但本次启动后还没写出第一条步数日志（编译与显存峰值阶段）"""
    alive = subprocess.run(["pgrep", "-f", "run_nb.py train_walkthrough"], capture_output=True).returncode == 0
    l256 = ROOT / "runs" / "free_256" / "train_log.txt"
    if not alive or not l256.exists():
        return False
    last = [l for l in l256.read_text().splitlines() if l.strip()][-1:]
    return bool(last) and not last[0].startswith("step")


def wait_gpu(need_mib, who):
    """need_mib 为任务自身的显存；0 表示纯 CPU 任务，不等待"""
    if need_mib == 0:
        return
    need_mib += RESERVE_MIB
    waited = False
    while PAUSE.exists() or diffusion_training_starting() or free_mib() < need_mib:
        if not waited:
            log(f"{who}: waiting for GPU (free {free_mib()} MiB, need {need_mib})"); waited = True
        time.sleep(60)


def run(cmd, who, need_mib):
    wait_gpu(need_mib, who)
    log(f"{who}: start {' '.join(cmd[1:])}")
    t = time.time()
    p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    tail = (p.stdout + p.stderr).strip().splitlines()[-1:] if p.returncode else []
    log(f"{who}: {'done' if p.returncode == 0 else 'FAILED'} in {(time.time() - t) / 60:.0f} min "
        f"{' '.join(cmd[1:])} {tail}")
    return p.returncode == 0


def orphans():
    """仍在运行的 detect.py / generate.py 进程（按 /proc 中的命令行精确匹配，不会匹配到 shell）"""
    found = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                args = (d / "cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            if len(args) > 1 and args[0].endswith(b"python") and args[1] in (b"detect.py", b"generate.py", b"diagnose_crg.py"):
                found.append(int(d.name))
    return found


def n_done(method):
    d = SYN / method / "labels"
    return len(list(d.glob("j*.txt"))) if d.exists() else 0


def generation():
    for m in MAIN + ["crgwin"]:                                               # 划分 A：300 个任务，只用 A 的训练照片
        if n_done(m + "_A") < 300:
            run([PY, "generate.py", "--method", m, "--start", "0", "--end", "300", "--jobs", "jobs_A", "--tag", "_A"],
                "gen", 0 if m == "crgwin" else GEN_MIB)
    for lo in (0, 300, 600):
        for m in MAIN + (ABLATION if lo == 0 else []):
            if n_done(m) < lo + 300:
                for _ in range(3):                                            # 被中断的生成最多重试两次
                    if run([PY, "generate.py", "--method", m, "--start", str(lo), "--end", str(lo + 300)], "gen", GEN_MIB):
                        break
            if m == "crg" and n_done("crgwin") < n_done("crg"):              # 诊断：CRG 窗口贴回原背景（CPU）
                run([PY, "generate.py", "--method", "crgwin", "--start", "0", "--end", "900"], "gen", 0)
    log("gen: all generation finished")


DIAG_CONFIGS = ["crgwin_1x", "crg_synonly", "crgwin_synonly"]             # 2026-11-17 加入的 CRG 诊断，优先运行
# 2026-11-17：划分 A（论文式）上的 1× 比较，排在诊断之后、其余按板划分任务之前
SPLIT_A = ["original", "dpmpaste_1x", "repaint_1x", "copaint_1x", "crg_1x", "copypaste_1x", "ecics_1x", "crgwin_1x",
           "controlnet_1x"]                                            # 2026-11-18：外部基线（数据就绪后才会运行）


def configs():
    yield "original"
    yield from DIAG_CONFIGS
    for k in (1, 2, 3):
        for m in MAIN:
            yield f"{m}_{k}x"
    for m in ["ecics", *ABLATION, "crg"]:                     # 消融：base → +SA → +GS → +CCF → +EMA（完整）
        if f"{m}_synonly" not in DIAG_CONFIGS:
            yield f"{m}_synonly"


def needed(cfg):
    cfg = cfg.split(":")[-1]
    if cfg == "original":
        return None, 0
    m, s = cfg.rsplit("_", 1)
    return m, 300 if s == "synonly" else 300 * int(s[:-1])


def diagnostics():
    """保留权重的 original / crg_1x 检测器（seed 0），再统计误检（diagnose_crg.py fp）"""
    for cfg in ["original", "crg_1x"]:
        if not (RES / "detection_diag" / cfg / "seed0" / "metrics.json").exists():
            run([PY, "detect.py", "--config", cfg, "--seed", "0", "--keep-weights"], "det", DET_MIB)
    if not (RES / "diag_crg" / "false_positives.json").exists():
        run([PY, "diagnose_crg.py", "fp"], "det", DET_MIB)


def detection():
    pending = [(c, s) for c in configs() for s in SEEDS]
    first = [(c, s) for c, s in pending if c in DIAG_CONFIGS]                 # 诊断配置排在最前
    split_a = [("A:" + c, s) for c in SPLIT_A for s in SEEDS]                 # 然后是划分 A
    pending = first + split_a + [x for x in pending if x not in first]
    diag_done = False
    retries = {}
    while pending:
        ready = [(c, s) for c, s in pending if needed(c)[0] is None
                 or n_done(needed(c)[0] + ("_A" if c.startswith("A:") else "")) >= needed(c)[1]]
        if not ready:
            time.sleep(120); continue
        if not diag_done and not any(c in DIAG_CONFIGS for c, _ in pending):   # 诊断配置跑完后做误检分析
            diagnostics(); diag_done = True
        c, s = ready[0]
        split, cfg = (c.split(":") if ":" in c else ("board", c))
        sub = "detection_A" if split == "A" else "detection"
        pending.remove((c, s))
        if not (RES / sub / cfg / f"seed{s}" / "metrics.json").exists():
            ok = run([PY, "detect.py", "--config", cfg, "--seed", str(s), "--split", split], "det", DET_MIB)
            retries[(c, s)] = retries.get((c, s), 0) + (not ok)
            if not ok and retries[(c, s)] <= 2:                                # 失败（例如被看门狗中断）则排到队尾重试
                pending.append((c, s))
    log("det: all detection finished")


if __name__ == "__main__":
    import fcntl
    RES.mkdir(parents=True, exist_ok=True)
    lock_file = open(RES / "pipeline.lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)            # 同一时间只允许一个 pipeline
    except BlockingIOError:
        sys.exit("another run_pipeline.py is already running")
    while orphans():                                                          # 等上一个 pipeline 留下的任务结束，避免重复运行
        time.sleep(30)
    log("pipeline start")
    threads = [threading.Thread(target=generation), threading.Thread(target=detection)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log("ALL DONE")
