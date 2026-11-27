"""检测器设置变体（detect.py --variant <名字>），2026-11-20 两天搜索用
每个变体是对 Ultralytics train() 参数的覆盖；"model" 指定权重文件，"imgsz" 默认 640，其余照原样传给 train()。
基线 base = 输入 640、其余为默认（与 --proto img640 相同）。
"""
VARIANTS = {
    "base": {},
    "flipud": {"flipud": 0.5},                                   # 电路板缺陷上下翻转不变（默认只左右翻转）
    "img800": {"imgsz": 800},
    "img960": {"imgsz": 960},
    "ep300": {"epochs": 300},
    "nomosaic": {"mosaic": 0.0},
    "close50": {"close_mosaic": 50},                             # 最后 50 个 epoch 关闭 mosaic（默认 10）
    "yolo11s": {"model": "yolo11s.pt"},                          # 更大的检测器（偏离论文的 YOLOv11n，单独报告）
    "dmgr": {"optimizer": "AdamW", "lr0": 0.01, "cos_lr": True},
    # 2026-11-24：框精度（mAP@50–95 的主要损失在高 IoU 阈值）
    "box10": {"box": 10.0},                                      # 框回归损失权重（默认 7.5）
    "box15": {"box": 15.0},
    "dfl3": {"dfl": 3.0},                                        # 分布式焦点损失权重（默认 1.5）
    "p2": {"model": "yolo11-p2.yaml"},                           # 加 stride 4 检测头，骨干等从 yolo11n.pt 迁移
    # 2026-11-25 P6：只改检测部分
    "rot90": {"flipud": 0.5, "transpose": 0.5},                  # 8 种 90° 旋转 / 镜像等概率（aug_rot90.py）
    "ft": {},                                                    # 两阶段第二阶段：设置同 base，起点为 --init 给的第一阶段权重
}
# 显存需求（MiB，batch 32），供 gpu_queue.py 使用；未列出的按 6000
MEM = {"img800": 9500, "img960": 13500, "yolo11s": 9000, "p2": 9000}


def combine(*names):
    out = {}
    for n in names:
        out.update(VARIANTS[n])
    return out


def get(name):
    """单个名字，或用 + 连接的组合（例如 flipud+img800）"""
    return combine(*name.split("+"))


def mem(name):
    return max([MEM.get(n, 6000) for n in name.split("+")])
