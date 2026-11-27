"""训练增强：随机转置（2026-11-25）
Ultralytics 只有左右、上下翻转；再加一次概率 p 的转置（交换 x、y），与两种翻转组合即得到 0°/90°/180°/270° 旋转及其镜像共 8 种，
p = 0.5 且两种翻转也为 0.5 时 8 种等概率。PCB 缺陷旋转 90° 后仍是同类缺陷，标注框旋转后仍是精确的矩形
（Ultralytics 的 degrees 任意角旋转会把框放大）。只在方形图像上转置（mosaic / letterbox 之后训练图像为 imgsz×imgsz）。
detect.py 在变体含 transpose 时于训练前调用 enable(p)；Linux 下 DataLoader 的 worker 由 fork 继承补丁。
"""
import random

import numpy as np
from ultralytics.data import augment as A


def transpose_labels(labels):
    """就地转置 labels 中的图像与实例（实例须为 xywh）"""
    labels["img"] = np.ascontiguousarray(np.swapaxes(labels["img"], 0, 1))
    ins = labels["instances"]
    b = ins.bboxes
    b[:] = b[:, [1, 0, 3, 2]]
    if ins.segments.size:
        ins.segments[...] = ins.segments[..., [1, 0]]
    if ins.keypoints is not None:
        ins.keypoints[..., :2] = ins.keypoints[..., [1, 0]]
    return labels


def enable(p):
    if getattr(A.RandomFlip, "_transpose_p", None) is not None:
        A.RandomFlip._transpose_p = p
        return
    A.RandomFlip._transpose_p = p
    get0, inst0 = A.RandomFlip.get_params, A.RandomFlip.apply_instances

    def get_params(self, labels):
        prm = get0(self, labels)
        h, w = labels["img"].shape[:2]
        prm["transpose"] = self.direction == "horizontal" and h == w and random.random() < A.RandomFlip._transpose_p
        return prm

    def apply_instances(self, labels, prm):
        labels = inst0(self, labels, prm)                      # 翻转之后实例为 xywh
        if prm.get("transpose"):
            labels = transpose_labels(labels)                  # 图像也在这里转置，保证两者同步
        return labels

    A.RandomFlip.get_params, A.RandomFlip.apply_instances = get_params, apply_instances
