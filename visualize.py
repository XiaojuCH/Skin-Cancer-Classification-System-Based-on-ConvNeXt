# visualize.py
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings
from models import get_model
from torchvision import transforms

# 忽略FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 手动设置参数
IMAGE_PATH = "00_M.jpg"
CHECKPOINT_PATH = "output/checkpoints/global_best.pth"
MODEL_NAME = "convnext_small"
NUM_CLASSES = 3
CLASS_NAMES = ['Melanoma', 'Seborrheic Keratosis', 'Nevus']
#                黑素瘤          脂溢性角化病            痣


def load_model(model_name, checkpoint_path, num_classes=3):
    """加载训练好的模型"""
    model = get_model(model_name, num_classes=num_classes)
    print(">>> 🧠 正在加载模型...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['net'], strict=False)
    model.eval()
    print(f">>> ✅ 权重加载完成: {checkpoint_path}")
    return model


def preprocess_image(image_path, img_size=224):
    """图像预处理"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    scale = img_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (new_w, new_h))

    top = (img_size - new_h) // 2
    bottom = img_size - new_h - top
    left = (img_size - new_w) // 2
    right = img_size - new_w - left
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img_padded).unsqueeze(0)
    return tensor, img_padded


def generate_cam(model, input_tensor, original_img):
    """生成CAM热力图"""
    features = []

    def hook_fn(module, input, output):
        features.append(output.detach())

    # 挂钩最后一层Conv
    handle = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and 'stages.3' in name:
            handle = module.register_forward_hook(hook_fn)
            break

    if handle is None:
        raise RuntimeError("⚠️ 没有找到适合挂钩的卷积层，请检查模型结构。")

    with torch.no_grad():
        output = model(input_tensor)
    handle.remove()

    features = features[0].squeeze(0)
    weights = model.head.fc.weight if hasattr(model.head, 'fc') else model.classifier[-1].weight
    cam_weights = weights[output.argmax(dim=1).item()].detach()
    cam = (cam_weights.view(-1, 1, 1) * features).sum(dim=0)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-5)
    cam = cam.cpu().numpy()

    cam_resized = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(original_img, 0.7, heatmap, 0.3, 0)

    return overlayed, output


def visualize_prediction(image_path, model):
    """可视化预测结果"""
    tensor, original_img = preprocess_image(image_path)
    cam_img, output = generate_cam(model, tensor, original_img)
    probs = torch.softmax(output, dim=1).squeeze().detach().numpy()

    fig = plt.figure(figsize=(15, 6))
    fig.suptitle("皮肤病分类可视化结果", fontsize=16)

    # 原始图
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(original_img)
    ax1.set_title("原始图像")
    ax1.axis('off')

    # 热力图
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB))
    ax2.set_title("模型关注区域")
    ax2.axis('off')

    # 概率柱状图
    ax3 = fig.add_subplot(1, 3, 3)
    bars = ax3.barh(CLASS_NAMES, probs, color='skyblue')
    ax3.set_xlim(0, 1)
    ax3.set_title("诊断概率分布")
    ax3.invert_yaxis()
    for i, bar in enumerate(bars):
        ax3.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{probs[i] * 100:.1f}%', va='center')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    print(f">>> 📷 正在载入图像并进行分析: {IMAGE_PATH}")
    model = load_model(MODEL_NAME, CHECKPOINT_PATH, NUM_CLASSES)
    visualize_prediction(IMAGE_PATH, model)
