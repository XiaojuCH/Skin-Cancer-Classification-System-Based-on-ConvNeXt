# models.py
import timm
import torch.nn as nn
from safetensors.torch import load_file
import os

LOCAL_MODEL_DIR = 'D:/Paper_/resnet/pretrained'  # ← 你可以改成自己的路径


def load_model_weights_from_local(model, path):
    """
    从本地 safetensors 文件加载权重到模型中，跳过形状不匹配的层
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"预训练权重文件未找到: {path}")
    state_dict = load_file(path)

    # 获取当前模型的参数
    model_state_dict = model.state_dict()
    filtered_state_dict = {}

    # 过滤形状不匹配的层
    for k, v in state_dict.items():
        if k in model_state_dict and v.shape == model_state_dict[k].shape:
            filtered_state_dict[k] = v
        else:
            print(f"⚠️ 跳过权重 {k}")

    model.load_state_dict(filtered_state_dict, strict=False)
    print(f">>> ✅ 成功从本地加载权重: {path}")
    return model


def get_model(model_name, num_classes=3, pretrained=True):
    """
    根据模型名称创建网络，并选择是否加载本地预训练权重
    """
    supported_models = {
        'resnet18': 'resnet18.a1_in1k.safetensors',
        'resnet50': 'resnet50.a1_in1k.safetensors',
        'efficientnet_b0': 'efficientnet_b0.ra_in1k.safetensors',
        'efficientnet_b1': 'efficientnet_b1.ra_in1k.safetensors',
        'convnext_tiny': 'convnext_tiny.fb_in1k.safetensors',
        'convnext_small': 'convnext_small.fb_in1k.safetensors',
        'convnext_base': 'convnext_base.fb_in1k.safetensors',  # 新增！
        'densenet121': 'densenet121.a1_in1k.safetensors',
    }

    if model_name not in supported_models:
        raise ValueError(f"❌ 不支持的模型: {model_name}")

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    if pretrained:
        weight_file = supported_models[model_name]
        pretrained_path = os.path.join(LOCAL_MODEL_DIR, weight_file)
        model = load_model_weights_from_local(model, pretrained_path)

    return model
