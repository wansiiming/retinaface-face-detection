"""
测试别人训练好的best_model.pth
"""
import torch
import cv2
import numpy as np
from PIL import Image
from retinaface_model import RetinaFace
from torchvision import transforms
import os

# 配置
MODEL_PATH = r"C:\Users\王子鸣\PycharmProjects\PythonProject\best_model.pth"
DATASET_ROOT = r"C:\Users\王子鸣\PycharmProjects\PythonProject\FaceDataset_organized"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 60)
print("加载 RetinaFace 模型")
print("=" * 60)
print(f"使用设备: {DEVICE}")

# 1. 创建模型
model = RetinaFace(num_anchors=3, num_keypoints=5)

# 2. 加载权重
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()
print("✅ 模型加载成功")

# 3. 测试图片路径
test_image = os.path.join(DATASET_ROOT, 'train', 'images', 'P0301_V01_FRONT_0001.jpg')

print(f"\n测试图片: {test_image}")

if not os.path.exists(test_image):
    print(f"❌ 测试图片不存在: {test_image}")
    exit()

# 4. 图像预处理（使用 PIL 读取，避免中文路径问题）
def preprocess(image_path):
    # 使用 PIL 读取图像
    img = Image.open(image_path).convert('RGB')
    original = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = original.shape[:2]

    # 预处理
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    return tensor, original, h, w

# 5. 推理
print("\n开始推理...")
tensor, original, h, w = preprocess(test_image)

with torch.no_grad():
    cls, reg, kpt = model(tensor)

print(f"\n输出形状:")
print(f"  分类: {cls.shape}")
print(f"  回归: {reg.shape}")
print(f"  关键点: {kpt.shape}")

# 6. 简单解码（取置信度最高的预测）
cls_prob = torch.softmax(cls[0], dim=1)[:, 1]  # 前景概率
best_idx = torch.argmax(cls_prob).item()
best_score = cls_prob[best_idx].item()

print(f"\n最佳检测:")
print(f"  置信度: {best_score:.4f}")
print(f"  Anchor索引: {best_idx}")

# 7. 保存结果
output_path = os.path.join(os.path.dirname(__file__), "test_input.jpg")
cv2.imwrite(output_path, original)
print(f"\n✅ 测试完成，输入图像已保存为 {output_path}")