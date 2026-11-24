"""
============================================================
文件编号: 5
所属周次: 第二周 第 7 天
实训步骤: 步骤 3 - RetinaFace 模型搭建之整体结构组装
功能作用: 组装 Backbone + FPN + 三个检测头，实现端到端人脸检测
          同时输出: 分类(前景/背景)、框回归、关键点回归
对应指导书: 实训四 3.③ RetinaFace整体模型搭建（含关键点检测分支）
依赖文件: 3_backbone.py, 4_fpn.py
输出结果: classifications(B,N,2), regressions(B,N,4), keypoints(B,N,10)
============================================================
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 动态导入 3_backbone.py 和 4_fpn.py
# 原因: 文件名带序号，无法直接用 from backbone import
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 加载 3_backbone.py
backbone_path = os.path.join(CURRENT_DIR, '3_backbone.py')
if not os.path.exists(backbone_path):
    raise FileNotFoundError(f"找不到 3_backbone.py: {backbone_path}")

import importlib.util

spec = importlib.util.spec_from_file_location("backbone_3", backbone_path)
backbone_module = importlib.util.module_from_spec(spec)
sys.modules["backbone_3"] = backbone_module
spec.loader.exec_module(backbone_module)
ResNetBackbone = backbone_module.ResNetBackbone
print(f"✓ 成功导入 ResNetBackbone (来自 3_backbone.py)")

# 加载 4_fpn.py
fpn_path = os.path.join(CURRENT_DIR, '4_fpn.py')
if not os.path.exists(fpn_path):
    raise FileNotFoundError(f"找不到 4_fpn.py: {fpn_path}")

spec = importlib.util.spec_from_file_location("fpn_4", fpn_path)
fpn_module = importlib.util.module_from_spec(spec)
sys.modules["fpn_4"] = fpn_module
spec.loader.exec_module(fpn_module)
FPN = fpn_module.FPN
print(f"✓ 成功导入 FPN (来自 4_fpn.py)")


# ============================================================
# RetinaFace 整体模型
# ============================================================

class RetinaFace(nn.Module):
    """
    RetinaFace 整体模型
    对应实训指导书: class RetinaFace(nn.Module)

    结构:
    1. Backbone: ResNet50 提取 C2~C5
    2. FPN: 融合为 P2~P7
    3. 三个并行 Head:
       - Classification Head: 判断 anchor 是否包含人脸
       - Regression Head: 精修人脸框坐标
       - Keypoint Head: 回归 5 个关键点坐标

    创新点: 检测与关键点一体化，单阶段完成（非两阶段 MTCNN）
    """

    def __init__(self, num_anchors=3, num_keypoints=5):
        """
        初始化 RetinaFace

        参数:
            num_anchors: 每个位置预设的 anchor 数量，默认 3
                        （不同长宽比: 1:1, 2:1, 1:2）
            num_keypoints: 人脸关键点数量，本课程固定 5
                          （左眼、右眼、鼻尖、左嘴角、右嘴角）
        """
        super().__init__()
        self.num_anchors = num_anchors
        self.num_keypoints = num_keypoints

        # 步骤 1: 骨干网络
        # 对应指导书: self.backbone = ResNet50()
        self.backbone = ResNetBackbone(pretrained=True)

        # 步骤 2: 特征金字塔
        # 对应指导书: self.fpn = FPN(in_channels_list, out_channels)
        # ResNet50 C2~C5 通道: [256, 512, 1024, 2048]
        self.fpn = FPN([256, 512, 1024, 2048], out_channels=256)

        # 步骤 3: 三个检测头（为每个 FPN 层配置）
        # 对应指导书: 为每个FPN输出层（P2-P7）配置三个分支

        # 分类头: 输出 num_anchors * 2（前景/背景二分类）
        self.classification_heads = nn.ModuleList([
            self._make_head(num_anchors * 2) for _ in range(6)
        ])

        # 回归头: 输出 num_anchors * 4（框坐标偏移 x,y,w,h）
        self.regression_heads = nn.ModuleList([
            self._make_head(num_anchors * 4) for _ in range(6)
        ])

        # 关键点头: 输出 num_anchors * 2 * num_keypoints
        # 5 个点 × 2 坐标 = 10
        self.keypoint_heads = nn.ModuleList([
            self._make_head(num_anchors * 2 * num_keypoints) for _ in range(6)
        ])

        # 初始化
        self._init_weights()

    def _make_head(self, out_channels):
        """
        构建检测头: 4 层 3x3 卷积 + ReLU，最后 1 层输出

        对应指导书: nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(), ...
        )
        """
        layers = []
        for _ in range(4):
            layers.extend([
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ])
        layers.append(nn.Conv2d(256, out_channels, kernel_size=3, padding=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        """初始化检测头权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播

        参数:
            x: (B, 3, 640, 640) 输入图像

        返回:
            classifications: (B, N_total, 2) 分类输出
            regressions:     (B, N_total, 4) 框回归输出
            keypoints:       (B, N_total, 10) 关键点输出
                             N_total = 所有 FPN 层的 anchor 总数
        """
        # 步骤 4: Backbone 提取特征
        # 对应指导书: backbone_features = self.backbone(x)
        c2, c3, c4, c5 = self.backbone(x)

        # 步骤 5: FPN 融合多尺度特征
        # 对应指导书: fpn_features = self.fpn(backbone_features)
        fpn_features = self.fpn((c2, c3, c4, c5))

        # 步骤 6: 对每个 FPN 层应用检测头
        # 对应指导书: 遍历每个FPN特征层，通过三个分支输出结果
        cls_outputs = []
        reg_outputs = []
        kpt_outputs = []

        for i, feature in enumerate(fpn_features):
            # 分类: (B, 256, H, W) → (B, num_anchors*2, H, W)
            cls = self.classification_heads[i](feature)
            # 回归: (B, 256, H, W) → (B, num_anchors*4, H, W)
            reg = self.regression_heads[i](feature)
            # 关键点: (B, 256, H, W) → (B, num_anchors*10, H, W)
            kpt = self.keypoint_heads[i](feature)

            # 步骤 7: 调整输出维度
            # 从 (B, C, H, W) → (B, H*W*num_anchors, K)
            B, _, H, W = cls.shape

            # 分类: (B, 2, H, W) → (B, H*W*A, 2)
            cls = cls.permute(0, 2, 3, 1).contiguous().view(B, -1, 2)
            # 回归: (B, 4, H, W) → (B, H*W*A, 4)
            reg = reg.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
            # 关键点: (B, 10, H, W) → (B, H*W*A, 10)
            kpt = kpt.permute(0, 2, 3, 1).contiguous().view(B, -1, 10)

            cls_outputs.append(cls)
            reg_outputs.append(reg)
            kpt_outputs.append(kpt)

        # 步骤 8: 拼接所有 FPN 层的输出
        # 对应指导书: classifications = torch.cat(classifications, dim=1)
        classifications = torch.cat(cls_outputs, dim=1)
        regressions = torch.cat(reg_outputs, dim=1)
        keypoints = torch.cat(kpt_outputs, dim=1)

        return classifications, regressions, keypoints


# ============================================================
# 测试代码
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("测试 RetinaFace 整体模型")
    print("=" * 60)

    # 步骤 9: 实例化模型
    model = RetinaFace(num_anchors=3, num_keypoints=5)

    # 步骤 10: 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量:")
    print(f"  总计: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"  可训练: {trainable_params:,}")

    # 步骤 11: 模拟输入
    x = torch.randn(2, 3, 640, 640)
    print(f"\n输入: {x.shape}")

    # 步骤 12: 前向传播
    cls, reg, kpt = model(x)

    # 步骤 13: 验证输出
    print(f"\n输出:")
    print(f"  分类: {cls.shape}  ← 每个 anchor 前景/背景概率")
    print(f"  回归: {reg.shape}  ← 每个 anchor 框偏移")
    print(f"  关键点: {kpt.shape}  ← 每个 anchor 5点坐标")

    print(f"\nAnchor 总数: {cls.shape[1]}")
    print(f"  = P2(160x160) + P3(80x80) + P4(40x40) + P5(20x20) + P6(10x10) + P7(5x5)")
    print(f"  × 3 anchors/位置")

    print("\n" + "=" * 60)
    print("✅ retinaface.py 测试通过")
    print("=" * 60)