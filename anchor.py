"""
============================================================
文件: anchors.py
      这套真实版代码是共用逻辑：
      8new_train.py、10new_eval.py、11new_detect.py 三个文件都需要导入 anchors.py 中 的AnchorGenerator。

      功能: RetinaFace Anchor 生成器（真实版新增）
      为 FPN 的 P2~P7 每层生成预设候选框（102375 个）
      这是真实 RetinaFace 的关键组件，原简化版省略了此部分
      原简化版直接把 GT（Ground Truth 该样本对应的真实目标框） 广播给所有 anchor，导致模型学乱；
      真实版通过 anchor 与 GT 的 IoU 匹配，只让"附近"的 anchor 负责预测。 IoU 是 Intersection over Union（交并比） 的缩写，用于衡量两个框（例如预测框与真实框）的重叠程度。如果某个 anchor 与某个 GT 框的 IoU > 0.5，就认为这个 anchor“正样本”，让它去学习预测该 GT 框。
============================================================
"""

import numpy as np
import torch


class AnchorGenerator:
    """
    RetinaFace 风格 Anchor 生成器
    为 P2~P7 每层生成 3 个不同尺度的 anchor，共 102375 个
    """

    def __init__(self, input_size=640):
        self.input_size = input_size
        # FPN 各层配置：stride=下采样倍数, size=特征图尺寸, base=基础 anchor 尺寸
        self.levels = [
            {'name': 'P2', 'stride': 4,   'size': 160, 'base': 16},
            {'name': 'P3', 'stride': 8,   'size': 80,  'base': 32},
            {'name': 'P4', 'stride': 16,  'size': 40,  'base': 64},
            {'name': 'P5', 'stride': 32,  'size': 20,  'base': 128},
            {'name': 'P6', 'stride': 64,  'size': 10,  'base': 256},
            {'name': 'P7', 'stride': 128, 'size': 5,   'base': 512},
        ]
        self.num_anchors_per_loc = 3

    def generate_base_anchors(self, base_size):
        """
        生成单个位置的基础 anchor（3 个尺度）
        尺寸: [base, base*1.5, base*2]，覆盖小/中/大目标
        返回: (3, 4) [x1, y1, x2, y2]
        """
        scales = [1.0, 1.5, 2.0]
        anchors = []
        for s in scales:
            w = h = base_size * s
            x1, y1, x2, y2 = -w / 2, -h / 2, w / 2, h / 2
            anchors.append([x1, y1, x2, y2])
        return np.array(anchors, dtype=np.float32)

    def generate_all_anchors(self):
        """
        生成全部 FPN 层的 anchor
        返回: (102375, 4) 的 numpy 数组 [x1,y1,x2,y2]，像素坐标
        """
        all_anchors = []
        for lvl in self.levels:
            stride = lvl['stride']
            fm_size = lvl['size']
            base = lvl['base']

            base_anchors = self.generate_base_anchors(base)  # (3, 4)

            # 特征图每个像素对应原图位置（中心对齐）
            shift_x = np.arange(fm_size) * stride + stride / 2
            shift_y = np.arange(fm_size) * stride + stride / 2
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)

            shifts = np.stack([
                shift_x.ravel(), shift_y.ravel(),
                shift_x.ravel(), shift_y.ravel()
            ], axis=1)  # (H*W, 4)

            num_loc = shifts.shape[0]
            shifts = shifts.reshape(num_loc, 1, 4)
            base_anchors = base_anchors.reshape(1, 3, 4)
            anchors = shifts + base_anchors  # (H*W, 3, 4)
            anchors = anchors.reshape(-1, 4)  # (H*W*3, 4)

            all_anchors.append(anchors)

        return np.vstack(all_anchors).astype(np.float32)

    def to_tensor(self, device='cpu'):
        """转为 PyTorch Tensor 并移到指定设备"""
        anchors = self.generate_all_anchors()
        return torch.from_numpy(anchors).to(device)