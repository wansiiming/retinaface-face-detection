"""
RetinaFace 模型定义（完全匹配best_model.pth的结构）
- FPN lateral_convs/fpn_convs: 无 bias
- FPN p6/p7: 有 bias
- Head: 有 bias
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetBackbone(nn.Module):
    """ResNet50骨干网络"""
    def __init__(self, pretrained=False):
        super().__init__()
        if pretrained:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet50(weights=None)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


class FPN(nn.Module):
    """特征金字塔网络"""
    def __init__(self, in_channels_list=[256, 512, 1024, 2048], out_channels=256):
        super().__init__()
        # lateral_convs 和 fpn_convs: bias=False
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1, bias=False) for in_ch in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False) for _ in range(len(in_channels_list))
        ])
        # p6 和 p7: bias=True（匹配pth文件）
        self.p6 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=True)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=True)

    def forward(self, inputs):
        c2, c3, c4, c5 = inputs
        p5 = self.lateral_convs[3](c5)
        p4 = self.lateral_convs[2](c4)
        p3 = self.lateral_convs[1](c3)
        p2 = self.lateral_convs[0](c2)

        p4 = F.interpolate(p5, size=p4.shape[-2:], mode='nearest') + p4
        p3 = F.interpolate(p4, size=p3.shape[-2:], mode='nearest') + p3
        p2 = F.interpolate(p3, size=p2.shape[-2:], mode='nearest') + p2

        p2 = self.fpn_convs[0](p2)
        p3 = self.fpn_convs[1](p3)
        p4 = self.fpn_convs[2](p4)
        p5 = self.fpn_convs[3](p5)

        p6 = self.p6(p5)
        p7 = self.p7(F.relu(p6))

        return [p2, p3, p4, p5, p6, p7]


def make_head(in_channels=256, out_channels=6):
    """检测头（有 bias）"""
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True)
    )


class RetinaFace(nn.Module):
    """RetinaFace 完整模型"""
    def __init__(self, num_anchors=3, num_keypoints=5):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_keypoints = num_keypoints

        self.backbone = ResNetBackbone(pretrained=False)
        self.fpn = FPN()

        # 6个FPN层
        self.classification_heads = nn.ModuleList([
            make_head(256, num_anchors * 2) for _ in range(6)
        ])
        self.regression_heads = nn.ModuleList([
            make_head(256, num_anchors * 4) for _ in range(6)
        ])
        self.keypoint_heads = nn.ModuleList([
            make_head(256, num_anchors * 2 * num_keypoints) for _ in range(6)
        ])

    def forward(self, x):
        c2, c3, c4, c5 = self.backbone(x)
        features = self.fpn((c2, c3, c4, c5))

        cls_outputs = []
        reg_outputs = []
        kpt_outputs = []

        for i, feat in enumerate(features):
            cls = self.classification_heads[i](feat)
            reg = self.regression_heads[i](feat)
            kpt = self.keypoint_heads[i](feat)

            B, _, H, W = cls.shape
            cls = cls.permute(0, 2, 3, 1).reshape(B, -1, 2)
            reg = reg.permute(0, 2, 3, 1).reshape(B, -1, 4)
            kpt = kpt.permute(0, 2, 3, 1).reshape(B, -1, 10)

            cls_outputs.append(cls)
            reg_outputs.append(reg)
            kpt_outputs.append(kpt)

        return torch.cat(cls_outputs, 1), torch.cat(reg_outputs, 1), torch.cat(kpt_outputs, 1)


if __name__ == '__main__':
    print("=" * 50)
    print("测试 RetinaFace 模型定义")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = RetinaFace(num_anchors=3, num_keypoints=5)
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    x = torch.randn(1, 3, 640, 640).to(device)
    print(f"输入形状: {x.shape}")

    with torch.no_grad():
        cls, reg, kpt = model(x)

    print(f"\n输出形状:")
    print(f"  分类: {cls.shape}")
    print(f"  回归: {reg.shape}")
    print(f"  关键点: {kpt.shape}")

    print("\n✅ 模型定义正确！")