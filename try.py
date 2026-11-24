"""
对比模型定义和pth文件的参数
"""
import torch
from retinaface_model import RetinaFace

MODEL_PATH = r"C:\Users\王子鸣\PycharmProjects\PythonProject\best_model.pth"

# 加载pth文件
state_dict = torch.load(MODEL_PATH, map_location='cpu')
print("pth文件中的参数（前20个）:")
for i, key in enumerate(list(state_dict.keys())[:20]):
    print(f"  {i+1}. {key}")

print("\n" + "=" * 50)

# 创建模型
model = RetinaFace()
model_dict = model.state_dict()
print("模型定义的参数（前20个）:")
for i, key in enumerate(list(model_dict.keys())[:20]):
    print(f"  {i+1}. {key}")

print("\n" + "=" * 50)

# 找出差异
pth_keys = set(state_dict.keys())
model_keys = set(model_dict.keys())

missing = model_keys - pth_keys
extra = pth_keys - model_keys

if missing:
    print(f"模型有但pth没有的参数: {list(missing)[:10]}")
if extra:
    print(f"pth有但模型没有的参数: {list(extra)[:10]}")
if not missing and not extra:
    print("✅ 参数完全匹配！可以加载模型")