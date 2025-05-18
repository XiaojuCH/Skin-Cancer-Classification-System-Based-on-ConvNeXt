# -*- coding: utf-8 -*-
# @Author :Xiaoju
# Time : 2025/5/16 上午11:10
import os
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# （1）定义你的增强管线，和 train.py 中保持一致
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
])

# （2）加载一张原始图片
img_path = "01.jpg"   # 替换成你自己的图片路径
orig = Image.open(img_path).convert("RGB")

# （3）对同一张图片随机增强两次
aug1 = train_tf(orig)
aug2 = train_tf(orig)

# （4）把 tensor 转回 numpy，去掉 normalize，方便显示
def denormalize(tensor):
    t = tensor.clone()
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    t = t * std + mean
    return t.clamp(0,1).permute(1,2,0).numpy()

import torch
img1 = denormalize(aug1)
img2 = denormalize(aug2)

# （5）并排用 matplotlib 展示
fig, axes = plt.subplots(1, 2, figsize=(8,4))
axes[0].imshow(img1)
axes[0].set_title("Augmentation 1")
axes[0].axis("off")
axes[1].imshow(img2)
axes[1].set_title("Augmentation 2")
axes[1].axis("off")
plt.tight_layout()
plt.show()
