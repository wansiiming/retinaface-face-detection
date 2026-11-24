"""
============================================================
文件编号: 11new
所属周次: 第二周 第 10 天（进阶版）
实训步骤: 步骤 1 - 推理演示之单张图像检测（真实版）
功能作用: 加载训练好的模型，对单张图像进行人脸检测
          【新增】包含真实后处理：Anchor 解码 + NMS + 置信度筛选
          不再硬取第 0 个 anchor，而是让模型自己挑"最像人脸"的位置
对应指导书: 实训四 6.② 训练结果可视化 + 推理代码（进阶）
依赖文件: 5_retinaface.py, 7_config.py, anchors.py
输出结果: xxx_result_new.jpg（带标注的结果图像）
============================================================
"""

import os
import sys
import importlib.util
import cv2
import torch
import numpy as np

# ============================================================
# 动态导入带序号的模块
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_module_from_file(module_name, file_name):
    """动态加载带序号的 .py 文件【已修复：全部从当前 new 文件夹加载】"""
    file_path = os.path.join(CURRENT_DIR, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到 {file_name}: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# 加载 5_retinaface.py
retinaface_module = load_module_from_file("retinaface_5", "5_retinaface.py")
RetinaFace = retinaface_module.RetinaFace

# 加载 7_config.py
cfg = load_module_from_file("config_7", "7_config.py")

# 【新增】加载 Anchor 生成器
anchors_module = load_module_from_file("anchors_gen", "anchors.py")
AnchorGenerator = anchors_module.AnchorGenerator

# 提取配置
NUM_ANCHORS = cfg.NUM_ANCHORS
NUM_KEYPOINTS = cfg.NUM_KEYPOINTS
DEVICE = cfg.DEVICE
DATASET_ROOT = cfg.DATASET_ROOT
INPUT_SIZE = 640

# ===================== 【已按你要求修改】 =====================
BEST_MODEL_NAME = 'best_model_new1.pth'
SAVE_DIR = os.path.join(CURRENT_DIR, "checkpoints")  # 模型在 new/checkpoints 下


# ============================================================
# 【新增】后处理工具函数：解码 + NMS
# ============================================================

def decode_predictions(pred_cls, pred_reg, pred_kpt, anchors, conf_thresh=0.5, nms_thresh=0.4):
    """
    解码模型输出：偏移量 → 真实坐标 + NMS 去重

    【新增详细说明】
    简化版的问题：直接取第 0 个 anchor 的预测，不管它是不是人脸。
    真实版的解决：
    1. 用 softmax 算出每个 anchor 包含人脸的置信度
    2. 只保留置信度 > conf_thresh（默认 0.5）的 anchor
    3. 把这些 anchor 预测的"偏移量"解码为真实像素坐标
    4. 用 NMS（非极大值抑制）去掉重叠的框，只保留最好的

    参数:
        pred_cls:     (1, N, 2) 分类输出（前景/背景 logits）
        pred_reg:     (1, N, 4) 框回归输出（偏移量 dx,dy,dw,dh）
        pred_kpt:     (1, N, 10) 关键点输出（偏移量）
        anchors:      (N, 4) 预设 anchor（像素坐标，范围 [0,640]）
        conf_thresh:  置信度阈值，低于此值的 anchor 被丢弃
        nms_thresh:   NMS IoU 阈值，重叠超过此值的框被去重

    返回:
        boxes:      (K, 4) 解码后的人脸框 [x1,y1,x2,y2]（像素坐标）
        scores:     (K,) 置信度
        keypoints:  (K, 5, 2) 解码后的 5 个关键点（像素坐标）
    """
    N = pred_cls.shape[1]

    # 1. 计算前景置信度（softmax 后取前景概率）
    scores = torch.softmax(pred_cls[0], dim=-1)[:, 1]  # (N,)

    # 2. 筛选高置信度 anchor
    keep = scores > conf_thresh
    if keep.sum() == 0:
        # 没有检测到人脸
        return torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0, 5, 2))

    scores = scores[keep]
    pred_reg = pred_reg[0][keep]  # (K, 4)
    pred_kpt = pred_kpt[0][keep]  # (K, 10)
    anchors_k = anchors[keep]  # (K, 4)

    # 3. 解码框坐标（反偏移量公式）
    # anchor 的宽高和中心
    a_w = anchors_k[:, 2] - anchors_k[:, 0]
    a_h = anchors_k[:, 3] - anchors_k[:, 1]
    a_cx = (anchors_k[:, 0] + anchors_k[:, 2]) / 2
    a_cy = (anchors_k[:, 1] + anchors_k[:, 3]) / 2

    # 预测的偏移量
    dx, dy, dw, dh = pred_reg[:, 0], pred_reg[:, 1], pred_reg[:, 2], pred_reg[:, 3]

    # 反算真实中心与宽高
    g_cx = dx * a_w + a_cx
    g_cy = dy * a_h + a_cy
    g_w = torch.exp(dw) * a_w
    g_h = torch.exp(dh) * a_h

    # 转回 x1,y1,x2,y2 格式
    x1 = g_cx - g_w / 2
    y1 = g_cy - g_h / 2
    x2 = g_cx + g_w / 2
    y2 = g_cy + g_h / 2
    boxes = torch.stack([x1, y1, x2, y2], dim=1)  # (K, 4)

    # 4. 解码关键点（同样反偏移量）
    kpt_x = pred_kpt[:, 0::2]  # (K, 5) 偶数索引为 x
    kpt_y = pred_kpt[:, 1::2]  # (K, 5) 奇数索引为 y
    # 反算真实坐标
    kpt_x = kpt_x * a_w.unsqueeze(1) + a_cx.unsqueeze(1)
    kpt_y = kpt_y * a_h.unsqueeze(1) + a_cy.unsqueeze(1)
    keypoints = torch.stack([kpt_x, kpt_y], dim=2)  # (K, 5, 2)

    # 5. NMS 去重（去掉重叠的人脸框）
    keep_idx = nms(boxes, scores, nms_thresh)
    boxes = boxes[keep_idx]
    scores = scores[keep_idx]
    keypoints = keypoints[keep_idx]

    return boxes, scores, keypoints


def nms(boxes, scores, threshold=0.5):
    """
    非极大值抑制（NMS）

    【新增详细说明】
    作用：一张图可能有很多 anchor 都说"我有人脸"，而且位置很接近，
    NMS 只保留置信度最高的那个，去掉与它重叠（IoU>threshold）的其他框。

    参数:
        boxes:     (N, 4) [x1,y1,x2,y2]
        scores:    (N,) 置信度
        threshold: IoU 阈值，超过此值视为重叠并去除

    返回:
        keep: 保留的索引（LongTensor）
    """
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)  # 每个框的面积

    # 按置信度从高到低排序
    _, order = scores.sort(descending=True)
    keep = []

    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order[0].item())
            break

        # 当前置信度最高的框
        i = order[0]
        keep.append(i.item())

        # 计算 i 与剩余所有框的 IoU
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        # 只保留 IoU 小于阈值的（不重叠的）
        mask = iou < threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


# ============================================================
# 图像预处理与可视化（与 11_detect.py 基本一致）
# ============================================================

def preprocess_image(image_path):
    """
    图像预处理
    步骤: 读取 → BGR2RGB → Resize → 归一化 → ImageNet 标准化 → 转张量
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    orig_h, orig_w = image.shape[:2]
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))
    img_norm = img_resized / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_std = (img_norm - mean) / std

    img_tensor = torch.from_numpy(img_std.transpose(2, 0, 1)).float()
    img_tensor = img_tensor.unsqueeze(0)  # (1, C, H, W)

    return img_tensor, image, orig_h, orig_w


def visualize_result(image, face_box, keypoints, score, save_path=None):
    """
    可视化检测结果（真实版）
    【新增】显示置信度 score，让学生看到"模型有多确定"
    """
    result = image.copy()
    x1, y1, x2, y2 = map(int, face_box)

    # 画人脸框
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 画关键点
    colors = [
        (0, 0, 255),  # 0: 左眼 - 红色
        (0, 255, 0),  # 1: 右眼 - 绿色
        (255, 0, 0),  # 2: 鼻尖 - 蓝色
        (255, 255, 0),  # 3: 左嘴角 - 黄色
        (255, 0, 255),  # 4: 右嘴角 - 紫色
    ]
    labels = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']

    for i, (x, y) in enumerate(keypoints):
        x, y = int(x), int(y)
        cv2.circle(result, (x, y), 5, colors[i], -1)
        cv2.putText(result, labels[i], (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)

    # 【新增】在左上角显示置信度
    info_text = f"Face: ({x1},{y1})-({x2},{y2})  Score:{score:.3f}"
    cv2.putText(result, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if save_path:
        cv2.imwrite(save_path, result)
        print(f"结果保存: {save_path}")

    return result


# ============================================================
# 检测主函数（真实版）
# ============================================================

def detect(image_path, model_path=None, save_result=True):
    """
    单张图像人脸检测主函数（真实版）

    【新增】与简化版的区别：
    - 简化版：直接取第 0 个 anchor，无置信度、无 NMS
    - 真实版：生成 anchor → 模型推理 → 置信度筛选 → 偏移量解码 → NMS 去重
    """
    print("=" * 60)
    print("RetinaFace 单张图像检测（真实版）")
    print("=" * 60)

    if model_path is None:
        model_path = os.path.join(SAVE_DIR, BEST_MODEL_NAME)

    print(f"图像: {image_path}")
    print(f"模型: {model_path}")

    # 加载模型
    print("\n加载模型...")
    model = RetinaFace(num_anchors=NUM_ANCHORS, num_keypoints=NUM_KEYPOINTS)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}\n"
                                f"请先运行 8new_train.py 训练真实版模型！")

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    # 【新增】生成 Anchor（像素坐标，用于解码）
    print("生成 Anchor...")
    anchor_gen = AnchorGenerator(input_size=INPUT_SIZE)
    anchors = anchor_gen.to_tensor(DEVICE)  # (102375, 4)，像素坐标 [0,640]
    print(f"  Anchor 总数: {anchors.shape[0]}")

    # 预处理
    print("预处理图像...")
    img_tensor, orig_image, orig_h, orig_w = preprocess_image(image_path)
    img_tensor = img_tensor.to(DEVICE)

    # 推理
    print("模型推理...")
    with torch.no_grad():
        pred_cls, pred_reg, pred_kpt = model(img_tensor)

    # 【新增】真实后处理：解码 + NMS（核心区别）
    print("后处理（解码 + NMS）...")
    boxes, scores, keypoints = decode_predictions(
        pred_cls, pred_reg, pred_kpt,
        anchors, conf_thresh=0.5, nms_thresh=0.4
    )

    if boxes.numel() == 0:
        print("\n⚠️  未检测到人脸（所有 anchor 置信度均低于 0.5）")
        # 没检测到也弹出图片
        cv2.imshow("检测结果", orig_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return orig_image

    # 取置信度最高的结果（NMS 后通常只剩 1 个）
    best_box = boxes[0].cpu().numpy()
    best_score = scores[0].item()
    best_kpt = keypoints[0].cpu().numpy()

    # 将坐标映射回原始图像尺寸（从 640x640 映射回原图）
    scale_x = orig_w / INPUT_SIZE
    scale_y = orig_h / INPUT_SIZE
    best_box[0] *= scale_x
    best_box[1] *= scale_y
    best_box[2] *= scale_x
    best_box[3] *= scale_y
    best_kpt[:, 0] *= scale_x
    best_kpt[:, 1] *= scale_y

    print(f"\n检测结果:")
    print(f"  人脸框: [{int(best_box[0])}, {int(best_box[1])}, {int(best_box[2])}, {int(best_box[3])}]")
    print(f"  置信度: {best_score:.3f}")
    print(f"  关键点:")
    labels = ['左眼', '右眼', '鼻尖', '左嘴角', '右嘴角']
    for i, (x, y) in enumerate(best_kpt):
        print(f"    {labels[i]}: ({int(x)}, {int(y)})")

    # 可视化
    if save_result:
        save_path = image_path.rsplit('.', 1)[0] + '_result_new.jpg'
    else:
        save_path = None

    result = visualize_result(orig_image, best_box, best_kpt, best_score, save_path)

    # ===================== 【自动弹出图片】 =====================
    cv2.imshow("检测结果", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n" * 1)
    print("=" * 60)
    print("检测完成")
    print("=" * 60)

    return result


if __name__ == '__main__':
    # 测试图像路径
    TEST_IMAGE = r"E:\FaceDataset\train\images\P0306_V01_FRONT_0024.jpg"
    # 如果默认路径不存在，自动找测试集第一张图
    if not os.path.exists(TEST_IMAGE):
        test_dir = os.path.join(DATASET_ROOT, 'test', 'images')
        if os.path.exists(test_dir):
            images = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])
            if images:
                TEST_IMAGE = os.path.join(test_dir, images[0])

    detect(TEST_IMAGE)