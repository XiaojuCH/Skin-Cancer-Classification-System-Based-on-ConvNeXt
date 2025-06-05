import timm
import torch
import torch.nn as nn
from safetensors.torch import load_file
import os
import SCSA_Attention

LOCAL_MODEL_DIR = 'D:/Paper_/resnet/pretrained'  # 本地预训练权重路径


# class ECA(nn.Module):
#     def __init__(self, channels, k_size=3):
#         super().__init__()
#         self.avg = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         y = self.avg(x)  # B,C,1,1
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#         return x * self.sigmoid(y)
#
#
# class AttentionConvNeXt_ECA(nn.Module):
#     def __init__(self, backbone, num_classes):
#         super().__init__()
#         self.backbone = backbone
#         c = backbone.num_features
#         self.eca = ECA(c, k_size=3)
#         self.dropout = nn.Dropout(p=0.2)
#         self.head = nn.Linear(c, num_classes)
#
#     def forward(self, x):
#         x = self.backbone.forward_features(x)
#         x = self.eca(x)
#         x = x.mean([-2, -1])
#         return self.head(x)


def load_model_weights_from_local(model, path):
    """
    从本地 safetensors 文件加载权重到模型中，跳过形状不匹配的层
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"预训练权重文件未找到: {path}")
    state_dict = load_file(path)
    model_state_dict = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
    skipped = set(state_dict) - set(filtered)
    for k in skipped:
        print(f"⚠️ 跳过权重 {k}")
    model.load_state_dict(filtered, strict=False)
    print(f">>> ✅ 成功从本地加载权重: {path}")
    return model


class AttentionConvNeXt_SCSA(nn.Module):
    """ConvNeXt backbone + SCSA attention only + custom head"""

    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        channels = backbone.num_features
        # SCSA 注意力模块前置映射
        reduced = channels // 4
        self.reduce_conv = nn.Conv2d(channels, reduced, kernel_size=1)
        self.scsa = SCSA_Attention.SCSA(dim=reduced, head_num=8)
        self.expand_conv = nn.Conv2d(reduced, channels, kernel_size=1)
        # 分类头
        self.head = nn.Linear(channels, num_classes)

    def forward(self, x):
        # 提取特征
        x = self.backbone.forward_features(x)  # [B, C, H, W]
        # SCSA 注意力
        x = self.reduce_conv(x)
        x = self.scsa(x)
        x = self.expand_conv(x)
        # Global Pool + Head
        x = x.mean(dim=[2, 3])  # [B, C]
        logits = self.head(x)
        return logits


def get_model(model_name: str, num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """
    创建模型并加载预训练权重，仅保留 SCSA 注意力
    """
    supported = {
        'resnet18': 'resnet18.a1_in1k.safetensors',
        'resnet50': 'resnet50.a1_in1k.safetensors',
        'efficientnet_b0': 'efficientnet_b0.ra_in1k.safetensors',
        'efficientnet_b1': 'efficientnet_b1.ra_in1k.safetensors',
        'convnext_tiny': 'convnext_tiny.fb_in1k.safetensors',
        'convnext_small': 'convnext_small.fb_in1k.safetensors',
        'convnext_base': 'convnext_base.fb_in1k.safetensors',
        'densenet121': 'densenet121.a1_in1k.safetensors',
    }
    if model_name not in supported:
        raise ValueError(f"❌ 不支持的模型: {model_name}")

    # 创建 backbone，无头模型
    backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
    if pretrained:
        weight_path = os.path.join(LOCAL_MODEL_DIR, supported[model_name])
        load_model_weights_from_local(backbone, weight_path)

    if model_name.startswith('convnext_'):
        return AttentionConvNeXt_SCSA(backbone, num_classes)
    else:
        # 其他模型直接附加线性 head，无额外注意力
        model = backbone
        model.head = nn.Linear(model.num_features, num_classes)
        return model
