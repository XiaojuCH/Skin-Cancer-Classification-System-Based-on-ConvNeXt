# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms

class ISIC2017Dataset(Dataset):
    """ISIC-2017 皮肤病变分类数据集"""

    # 我们预期 CSV 至少包含这两列，第三类隐式计算
    POS_LABEL_COLS = ["melanoma", "seborrheic_keratosis"]

    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        """
        Args:
            root_dir (str): 数据集根目录，包含 train/val/test 子目录
            split (str): 'train' / 'val' / 'test'
            transform (callable): 图像预处理
            target_transform (callable): 标签预处理
        """
        self.root_dir = os.path.abspath(root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # CSV 文件就在 e.g. root_dir/train/train_groundtruth.csv
        csv_name = f"{split}_groundtruth.csv"
        csv_path = os.path.join(self.root_dir, split, csv_name)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        # 检查列
        if "image_id" not in self.df.columns or \
           not set(self.POS_LABEL_COLS).issubset(self.df.columns):
            raise ValueError(
                f"CSV 必须包含 'image_id' 和 {self.POS_LABEL_COLS} 三列，"
                f"当前列：{self.df.columns.tolist()}"
            )

        # 收集路径和标签
        self.image_paths = []
        self.labels = []
        for _, row in self.df.iterrows():
            image_id = row["image_id"]
            img_file = image_id if image_id.endswith('.jpg') else f"{image_id}.jpg"
            img_path = os.path.join(self.root_dir, split, "images", img_file)
            if not os.path.exists(img_path):
                print(f"Warning: 图像不存在 {img_path}")
                continue
            self.image_paths.append(img_path)

            m = row["melanoma"]
            s = row["seborrheic_keratosis"]
            b = 1.0 - m - s
            vec = np.array([m, s, b], dtype=np.float32)
            cls = int(vec.argmax())
            self.labels.append(cls)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        assert self.transform is None or callable(self.transform), f"transform={self.transform!r}"

        image = Image.open(img_path).convert('RGB')
        if self.transform:               # 只有当你传了 transform（如 train_tf）时才会执行
            image = self.transform(image)

        lbl = torch.tensor(label, dtype=torch.long)
        if self.target_transform:
            lbl = self.target_transform(lbl)

        return image, lbl

# 以下测试代码可删除
if __name__ == "__main__":
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    ds = ISIC2017Dataset(root_dir="D:/Paper_/resnet/ISIC-2017",
                         split="train",
                         transform=test_transform)
    print("样本数:", len(ds))
    im, lb = ds[0]
    print("图像尺寸:", im.shape, "标签:", lb)
