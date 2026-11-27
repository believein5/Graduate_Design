"""用 train_walkthrough.ipynb 中的训练函数重训缺陷先验模型（64），数据换成 defect_64_maxcov_trainA（2026-11-24）
python train_defect.py [--data defect_64_maxcov_trainA] [--run defect_64_maxcov_trainA]
从 notebook 取 cell 1（导入与路径）、cell 3（配置）、cell 5/7 中的函数定义、cell 13（采样与 train()），不运行演示部分；
训练设置与 2.5 节相同：batch 64、lr 1e-4、EMA 0.9999、200000 次迭代、混合精度 + torch.compile；可续训。
训练结束后写 <run>/done/ok，供 gpu_queue.py 中后续的权重选择任务判断
"""
import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "code" / "train_walkthrough.ipynb"


def nb_namespace(extra_def_cells=()):
    """执行 notebook 的定义部分，返回其全局命名空间；extra_def_cells 中的 cell 只取函数与类定义"""
    cells = json.load(open(NB))["cells"]
    src = lambda i: "".join(cells[i]["source"])
    ns = {"__name__": "nb"}
    exec(src(1).replace("%matplotlib inline", ""), ns)
    exec(src(3), ns)
    for i in (5, 7, *extra_def_cells):
        tree = ast.parse(src(i))
        tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(tree, NB.name, "exec"), ns)
    exec(src(13), ns)
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="defect_64_maxcov_trainA")
    ap.add_argument("--run", default="defect_64_maxcov_trainA")
    args = ap.parse_args()
    ns = nb_namespace()
    ns["RUNS"]["64"].update(data=args.data, per_class=None)
    run = ROOT / "runs" / args.run
    ns["run_dir"] = lambda kind: run                             # train() 通过全局 run_dir 取输出目录，避免写进已有的 runs/defect_64
    n = len(list((ROOT / "data" / "pcb_processed" / args.data).glob("*.jpg")))
    print(f"data {args.data}: {n} patches · run dir {run}", flush=True)
    ns["train"]("64")
    (run / "done").mkdir(exist_ok=True); (run / "done" / "ok").touch()


if __name__ == "__main__":
    main()
