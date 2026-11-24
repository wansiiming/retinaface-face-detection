"""
RetinaFace 检测脚本（适配您的环境）
"""
import os
import sys
import importlib.util
import cv2
import torch
import numpy as np
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_module_from_file(module_name, file_name):
    """动态加载模块"""
    file_path = os.path.join(CURRENT_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到 {file_name}: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# 加载模块
retinaface_module = load_module_from_file("retinaface_5", "5_retinaface.py")
RetinaFace = retinaface_module.RetinaFace

cfg = load_module_from_file("config_7", "7_config.py")

anchors_module = load_module_from_file("anchors", "anchors.py")
AnchorGenerator = anchors_module.AnchorGenerator

# 配置
NUM_ANCHORS = cfg.NUM_ANCHORS
NUM_KEYPOINTS = cfg.NUM_KEYPOINTS
DEVICE = cfg.DEVICE
DATASET_ROOT = cfg.DATASET_ROOT
INPUT_SIZE = 640

# 模型路径（您的 best_model.pth 所在位置）
MODEL_PATH = r"C:\Users\王子鸣\PycharmProjects\PythonProject\best_model.pth"


def decode_predictions(pred_cls, pred_reg, pred_kpt, anchors, conf_thresh=0.01, nms_thresh=0.4):
    """解码模型输出"""
    scores = torch.softmax(pred_cls[0], dim=-1)[:, 1]

    keep = scores > conf_thresh
    if keep.sum() == 0:
        print(f"没有高于阈值 {conf_thresh} 的检测")
        return torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0, 5, 2))

    scores = scores[keep]
    pred_reg = pred_reg[0][keep]
    pred_kpt = pred_kpt[0][keep]
    anchors_k = anchors[keep]

    a_w = anchors_k[:, 2] - anchors_k[:, 0]
    a_h = anchors_k[:, 3] - anchors_k[:, 1]
    a_cx = (anchors_k[:, 0] + anchors_k[:, 2]) / 2
    a_cy = (anchors_k[:, 1] + anchors_k[:, 3]) / 2

    dx, dy, dw, dh = pred_reg[:, 0], pred_reg[:, 1], pred_reg[:, 2], pred_reg[:, 3]

    g_cx = dx * a_w + a_cx
    g_cy = dy * a_h + a_cy
    g_w = torch.exp(dw) * a_w
    g_h = torch.exp(dh) * a_h

    x1 = g_cx - g_w / 2
    y1 = g_cy - g_h / 2
    x2 = g_cx + g_w / 2
    y2 = g_cy + g_h / 2
    boxes = torch.stack([x1, y1, x2, y2], dim=1)

    kpt_x = pred_kpt[:, 0::2]
    kpt_y = pred_kpt[:, 1::2]
    kpt_x = kpt_x * a_w.unsqueeze(1) + a_cx.unsqueeze(1)
    kpt_y = kpt_y * a_h.unsqueeze(1) + a_cy.unsqueeze(1)
    keypoints = torch.stack([kpt_x, kpt_y], dim=2)

    keep_idx = nms(boxes, scores, nms_thresh)
    boxes = boxes[keep_idx]
    scores = scores[keep_idx]
    keypoints = keypoints[keep_idx]

    return boxes, scores, keypoints


def nms(boxes, scores, threshold=0.5):
    """非极大值抑制"""
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(descending=True)
    keep = []

    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order[0].item())
            break

        i = order[0]
        keep.append(i.item())

        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        mask = iou < threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def preprocess_image(image_path):
    """图像预处理"""
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size

    img_resized = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    img_norm = np.array(img_resized) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_std = (img_norm - mean) / std

    img_tensor = torch.from_numpy(img_std.transpose(2, 0, 1)).float()
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor, orig_h, orig_w


def visualize_result(image, face_box, keypoints, score, save_path=None):
    """可视化检测结果"""
    result = image.copy()
    x1, y1, x2, y2 = map(int, face_box)

    # 画人脸框
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 画关键点
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255)]
    labels = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']

    for i, (x, y) in enumerate(keypoints):
        x, y = int(x), int(y)
        cv2.circle(result, (x, y), 3, colors[i], -1)
        cv2.putText(result, labels[i], (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1)

    # 显示置信度
    cv2.putText(result, f'Score: {score:.3f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if save_path:
        # 使用 PIL 保存
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(result_rgb)
        pil_img.save(save_path)
        print(f"结果保存: {save_path}")

    return result


def detect(image_path):
    """单张图像人脸检测"""
    print("=" * 60)
    print("RetinaFace 人脸检测")
    print("=" * 60)

    print(f"图像: {image_path}")
    print(f"模型: {MODEL_PATH}")

    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"❌ 图片不存在: {image_path}")
        return

    # 检查模型是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型不存在: {MODEL_PATH}")
        return

    # 加载模型
    print("\n加载模型...")
    model = RetinaFace(num_anchors=NUM_ANCHORS, num_keypoints=NUM_KEYPOINTS)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    print("✅ 模型加载成功")

    # 生成 Anchor
    print("生成 Anchor...")
    anchor_gen = AnchorGenerator(input_size=INPUT_SIZE)
    anchors = anchor_gen.to_tensor(DEVICE)
    print(f"  Anchor 总数: {anchors.shape[0]}")

    # 预处理
    print("预处理图像...")
    img_tensor, orig_h, orig_w = preprocess_image(image_path)
    img_tensor = img_tensor.to(DEVICE)

    # 推理
    print("模型推理...")
    with torch.no_grad():
        pred_cls, pred_reg, pred_kpt = model(img_tensor)

    # 后处理
    print("后处理...")
    boxes, scores, keypoints = decode_predictions(
        pred_cls, pred_reg, pred_kpt, anchors,
        conf_thresh=0.01, nms_thresh=0.4
    )

    # 读取原图用于显示
    orig_image = cv2.imread(image_path)
    if orig_image is None:
        orig_image = np.array(Image.open(image_path))
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)

    if boxes.numel() == 0:
        print("\n⚠️ 未检测到人脸")
        # 保存原图
        save_path = os.path.join(CURRENT_DIR, "detect_result.jpg")
        cv2.imwrite(save_path, orig_image)
        print(f"已保存原图: {save_path}")
        return

    # 取最佳结果
    best_box = boxes[0].cpu().numpy()
    best_score = scores[0].item()
    best_kpt = keypoints[0].cpu().numpy()

    # 映射回原图尺寸
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

    # 保存结果
    save_path = os.path.join(CURRENT_DIR, "detect_result.jpg")
    result = visualize_result(orig_image, best_box, best_kpt, best_score, save_path)

    # 显示结果
    print(f"\n按任意键关闭图片窗口...")
    cv2.imshow("Detection Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n检测完成")


if __name__ == '__main__':
    # 测试图片路径
    TEST_IMAGE = r"C:\Users\王子鸣\PycharmProjects\PythonProject\FaceDataset_organized\train\images\P0301_V01_FRONT_0001.jpg"

    # 如果不存在，自动找一张
    if not os.path.exists(TEST_IMAGE):
        train_dir = os.path.join(DATASET_ROOT, 'train', 'images')
        if os.path.exists(train_dir):
            images = [f for f in os.listdir(train_dir) if f.endswith('.jpg')]
            if images:
                TEST_IMAGE = os.path.join(train_dir, images[0])
                print(f"使用图片: {TEST_IMAGE}")

    detect(TEST_IMAGE)