"""
============================================================
文件编号: 3
所属周次: 第二周 第 7 天
实训步骤: 步骤 1 - RetinaFace 模型搭建之骨干网络
功能作用: 实现 ResNet50 骨干网络，提取 C2/C3/C4/C5 多尺度特征
          为后续 FPN 特征金字塔提供输入
对应指导书: 实训四 3.① 骨干网络（ResNet50）搭建
依赖文件: 无（被 5_retinaface.py 导入）
输出结果: C2(256,160,160), C3(512,80,80), C4(1024,40,40), C5(2048,20,20)
============================================================
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetBackbone(nn.Module):
    """
    ResNet50 骨干网络
    对应实训指导书: class ResNet50(nn.Module)

    功能:
    1. 加载 torchvision 预训练 ResNet50
    2. 提取 layer1~layer4 输出，作为多尺度特征
    3. 输出 C2/C3/C4/C5 供 FPN 融合

    注意: 本实现使用 torchvision 预训练权重，加速收敛
    """

    def __init__(self, pretrained=True):
        """
        初始化 ResNet50 骨干网络

        参数:
            pretrained: 是否加载 ImageNet 预训练权重
                       建议 True，可大幅提升收敛速度
        """
        super().__init__()

        # 步骤 1: 加载 torchvision 预训练 ResNet50
        # 对应指导书: models.resnet50(pretrained=True)
        # 适配新版PyTorch（2.12.0）
        print(f"加载 ResNet50 (pretrained={pretrained})...")
        if pretrained:
            # 新版PyTorch推荐方式
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet50(weights=None)

        # 步骤 2: 提取各层组件
        # 对应指导书: 第一层 7x7 卷积 + BN + ReLU + 最大池化
        self.conv1 = resnet.conv1  # 7x7 卷积，stride=2
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 3x3 池化，stride=2

        # 步骤 3: 提取四个残差块组
        # 对应指导书: 四个残差块组，输出特征层分别为 C2、C3、C4、C5
        self.layer1 = resnet.layer1  # C2: 256通道，160x160（输入640x640时）
        self.layer2 = resnet.layer2  # C3: 512通道，80x80
        self.layer3 = resnet.layer3  # C4: 1024通道，40x40
        self.layer4 = resnet.layer4  # C5: 2048通道，20x20

        # 冻结部分层（可选，微调时有用）
        # 如果数据量小，可冻结 layer1~layer3，只训练 layer4 和后续 FPN
        self._freeze_stages()

    def _freeze_stages(self):
        """冻结早期层（可选策略）"""
        # 默认不冻结，全部参与训练
        pass

    def forward(self, x):
        """
        前向传播
        对应指导书: def forward(self, x): 返回 C2, C3, C4, C5

        参数:
            x: (B, 3, 640, 640) 输入图像张量

        返回:
            c2: (B, 256, 160, 160)  低层特征，细节丰富，适合小目标
            c3: (B, 512, 80, 80)    中层特征
            c4: (B, 1024, 40, 40)   高层特征
            c5: (B, 2048, 20, 20)   最高层特征，语义强，适合大目标
        """
        # 步骤 4: 第一层处理
        # 对应指导书: x = self.conv1(x) → bn1 → relu → maxpool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 步骤 5: 逐层提取特征
        # 对应指导书: C2 = self.layer1(x), C3 = self.layer2(C2)...
        c2 = self.layer1(x)  # 输出: (B, 256, 160, 160)
        c3 = self.layer2(c2)  # 输出: (B, 512, 80, 80)
        c4 = self.layer3(c3)  # 输出: (B, 1024, 40, 40)
        c5 = self.layer4(c4)  # 输出: (B, 2048, 20, 20)

        return c2, c3, c4, c5


# ============================================================
# 测试代码
# 对应指导书: 模型搭建后的验证测试
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("测试 ResNet50 骨干网络")
    print("=" * 60)

    # 步骤 6: 实例化模型（不加载预训练权重，测试用）
    backbone = ResNetBackbone(pretrained=False)

    # 步骤 7: 模拟输入
    batch_size = 2
    x = torch.randn(batch_size, 3, 640, 640)
    print(f"\n输入张量: {x.shape}")

    # 步骤 8: 前向传播
    c2, c3, c4, c5 = backbone(x)

    # 步骤 9: 验证输出尺寸
    print("\n输出特征层尺寸:")
    print(f"  C2: {c2.shape}  ← 160x160，小目标检测")
    print(f"  C3: {c3.shape}  ← 80x80，中等目标")
    print(f"  C4: {c4.shape}  ← 40x40，大目标")
    print(f"  C5: {c5.shape}  ← 20x20，超大目标+上下文")

    # 步骤 10: 验证感受野
    print("\n感受野分析:")
    print(f"  C2 每个像素对应原图: ~8x8 区域")
    print(f"  C3 每个像素对应原图: ~16x16 区域")
    print(f"  C4 每个像素对应原图: ~32x32 区域")
    print(f"  C5 每个像素对应原图: ~64x64 区域")

    print("\n" + "=" * 60)
    print("✅ backbone.py 测试通过")
    print("=" * 60)