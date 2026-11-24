"""
============================================================
文件编号: 4
所属周次: 第二周 第 7 天
实训步骤: 步骤 2 - RetinaFace 模型搭建之 FPN 特征金字塔
功能作用: 融合 ResNet50 输出的 C2~C5，生成 P2~P7 多尺度特征
          实现自顶向下上采样 + 横向连接，增强不同尺度人脸检测能力
对应指导书: 实训四 3.② 特征金字塔网络（FPN）搭建
依赖文件: 被 5_retinaface.py 导入
输出结果: [P2, P3, P4, P5, P6, P7]，均为 256 通道
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """
    特征金字塔网络 (Feature Pyramid Network)
    对应实训指导书: class FPN(nn.Module)

    核心思想:
    1. 自顶向下: 将深层语义强的特征上采样，传递到浅层
    2. 横向连接: 浅层细节特征与深层语义特征相加融合
    3. 额外层: P6/P7 通过下采样生成，用于检测极小目标

    输入: ResNet50 输出的 [C2, C3, C4, C5]
    输出: [P2, P3, P4, P5, P6, P7]，均为 256 通道
    """

    def __init__(self, in_channels_list, out_channels=256):
        """
        初始化 FPN

        参数:
            in_channels_list: [256, 512, 1024, 2048] 对应 C2~C5 的通道数
            out_channels: 统一输出通道数，默认 256（RetinaFace 标准配置）
        """
        super().__init__()

        # 步骤 1: 1x1 卷积层（横向连接）
        # 对应指导书: 1x1卷积，统一各特征层通道数
        # 将 C2(256)→256, C3(512)→256, C4(1024)→256, C5(2048)→256
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False)
            for in_ch in in_channels_list
        ])

        # 步骤 2: 3x3 卷积层（消除混叠）
        # 对应指导书: 3x3卷积，消除上采样带来的混叠效应
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            for _ in range(len(in_channels_list))
        ])

        # 步骤 3: 额外层 P6/P7（用于检测极小目标）
        # 对应指导书: 额外添加P6、P7特征层（用于检测小尺度人脸）
        # P6: 对 P5 下采样（stride=2）
        self.p6 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        # P7: 对 P6 下采样（stride=2），先 ReLU 再卷积
        self.p7 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """He 初始化，适配 ReLU 激活"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        """
        前向传播: 自顶向下融合

        参数:
            inputs: (C2, C3, C4, C5) 来自 ResNetBackbone

        返回:
            [P2, P3, P4, P5, P6, P7] 融合后的多尺度特征
        """
        c2, c3, c4, c5 = inputs

        # 步骤 4: 1x1 卷积统一通道数
        # 对应指导书: lateral_C5 = self.lateral_convs[3](C5)...
        p5 = self.lateral_convs[3](c5)  # (B, 256, 20, 20)
        p4 = self.lateral_convs[2](c4)  # (B, 256, 40, 40)
        p3 = self.lateral_convs[1](c3)  # (B, 256, 80, 80)
        p2 = self.lateral_convs[0](c2)  # (B, 256, 160, 160)

        # 步骤 5: 自顶向下上采样融合
        # 对应指导书: P4 = F.interpolate(P5, scale_factor=2) + lateral_C4
        # P5 → P4
        p4 = F.interpolate(p5, size=p4.shape[-2:], mode='nearest') + p4
        # P4 → P3
        p3 = F.interpolate(p4, size=p3.shape[-2:], mode='nearest') + p3
        # P3 → P2
        p2 = F.interpolate(p3, size=p2.shape[-2:], mode='nearest') + p2

        # 步骤 6: 3x3 卷积消除混叠
        # 对应指导书: P2 = self.fpn_convs[0](P2)...
        p2 = self.fpn_convs[0](p2)
        p3 = self.fpn_convs[1](p3)
        p4 = self.fpn_convs[2](p4)
        p5 = self.fpn_convs[3](p5)

        # 步骤 7: 生成 P6/P7
        # 对应指导书: P6 = self.p6(P5), P7 = self.p7(F.relu(P6))
        p6 = self.p6(p5)
        p7 = self.p7(F.relu(p6))

        return [p2, p3, p4, p5, p6, p7]


# ============================================================
# 测试代码
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("测试 FPN 特征金字塔网络")
    print("=" * 60)

    # 步骤 8: 模拟 ResNet50 输出
    batch_size = 2
    c2 = torch.randn(batch_size, 256, 160, 160)
    c3 = torch.randn(batch_size, 512, 80, 80)
    c4 = torch.randn(batch_size, 1024, 40, 40)
    c5 = torch.randn(batch_size, 2048, 20, 20)

    print(f"\n输入特征层:")
    print(f"  C2: {c2.shape}")
    print(f"  C3: {c3.shape}")
    print(f"  C4: {c4.shape}")
    print(f"  C5: {c5.shape}")

    # 步骤 9: 实例化 FPN
    fpn = FPN([256, 512, 1024, 2048], out_channels=256)

    # 步骤 10: 前向传播
    features = fpn((c2, c3, c4, c5))

    # 步骤 11: 验证输出
    print(f"\n输出特征层 (均为 256 通道):")
    for i, p in enumerate(features):
        print(f"  P{i + 2}: {p.shape}")

    print(f"\n特征层用途:")
    print(f"  P2(160x160): 检测小目标人脸（远处/小尺寸）")
    print(f"  P3(80x80):   检测中等目标")
    print(f"  P4(40x40):   检测大目标")
    print(f"  P5(20x20):   检测超大目标")
    print(f"  P6(10x10):   检测极小目标")
    print(f"  P7(5x5):     检测极极小目标")

    print("\n" + "=" * 60)
    print("✅ fpn.py 测试通过")
    print("=" * 60)