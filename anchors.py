"""
Anchor 生成器（组员原版，不裁剪）
"""
import numpy as np
import torch


class AnchorGenerator:
    def __init__(self, input_size=640):
        self.input_size = input_size
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
        scales = [1.0, 1.5, 2.0]
        anchors = []
        for s in scales:
            w = h = base_size * s
            x1, y1, x2, y2 = -w / 2, -h / 2, w / 2, h / 2
            anchors.append([x1, y1, x2, y2])
        return np.array(anchors, dtype=np.float32)

    def generate_all_anchors(self):
        all_anchors = []
        for lvl in self.levels:
            stride = lvl['stride']
            fm_size = lvl['size']
            base = lvl['base']

            base_anchors = self.generate_base_anchors(base)

            shift_x = np.arange(fm_size) * stride + stride / 2
            shift_y = np.arange(fm_size) * stride + stride / 2
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)

            shifts = np.stack([
                shift_x.ravel(), shift_y.ravel(),
                shift_x.ravel(), shift_y.ravel()
            ], axis=1)

            num_loc = shifts.shape[0]
            shifts = shifts.reshape(num_loc, 1, 4)
            base_anchors = base_anchors.reshape(1, 3, 4)
            anchors = shifts + base_anchors
            anchors = anchors.reshape(-1, 4)

            all_anchors.append(anchors)

        return np.vstack(all_anchors).astype(np.float32)

    def to_tensor(self, device='cpu'):
        anchors = self.generate_all_anchors()
        return torch.from_numpy(anchors).to(device)


if __name__ == '__main__':
    anchor_gen = AnchorGenerator(input_size=640)
    anchors = anchor_gen.to_tensor('cpu')
    print(f"Anchor 数量: {anchors.shape[0]}")
    print(f"Anchor X 范围: [{anchors[:,0].min():.0f}, {anchors[:,2].max():.0f}]")
    print(f"Anchor Y 范围: [{anchors[:,1].min():.0f}, {anchors[:,3].max():.0f}]")