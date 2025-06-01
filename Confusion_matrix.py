# -*- coding: utf-8 -*-
# @Author : Xiaoju
# Time    : 2025/4/9 下午11:26

import os
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    PrecisionRecallDisplay
)
from torchvision.utils import save_image


def compute_best_thresholds(model, loader, device, num_classes=3):
    """
    在 loader（通常是验证集）上计算每个类别的最佳阈值（以 F1 最大化为准）。
    返回一个长度为 num_classes 的列表，里面是各类的阈值。
    """
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            probs = torch.softmax(model(imgs), dim=1).cpu().numpy()
            all_scores.append(probs)
            all_labels.append(labels.numpy())

    y_score = np.vstack(all_scores)  # shape (N, num_classes)
    y_true = np.hstack(all_labels)   # shape (N,)
    # one-hot 编码
    y_true_onehot = np.eye(num_classes)[y_true]  # shape (N, num_classes)

    best_thresh = []
    for i in range(num_classes):
        prec, recall, thresh = precision_recall_curve(y_true_onehot[:, i], y_score[:, i])
        # 计算 F1
        f1 = 2 * prec * recall / (prec + recall + 1e-8)
        # thresh 的长度比 prec/reall 写少 1 个，故取 f1[:-1]
        best_idx = np.argmax(f1[:-1])
        best_thresh.append(thresh[best_idx])
    return best_thresh


def threshold_predict(probs, best_thresh):
    """
    给定一个样本的预测概率向量 probs（长度 num_classes），
    以及各类的阈值 best_thresh，先看哪些 p_i > thresh_i，如果有多个，则取最大的概率对应的下标；
    如果 none (所有 p_i 都 ≤ thresh_i)，则直接取概率最大的类。
    """
    mask = [float(p) > float(t) for p, t in zip(probs, best_thresh)]
    if any(mask):
        # mask 中为 True 的候选类
        cands = [i for i, m in enumerate(mask) if m]
        # 在这些候选里选概率最大的索引
        return int(cands[np.argmax([probs[i] for i in cands])])
    else:
        return int(np.argmax(probs))


def plot_confusion(model, loader, device, best_thresh, out_path, title, class_names=None):
    """
    使用阈值 best_thresh 对 loader 中所有样本做预测，绘制混淆矩阵并保存。
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            probs = torch.softmax(model(imgs), dim=1).cpu().numpy()
            for p, l in zip(probs, labels.numpy()):
                pred = threshold_predict(p, best_thresh)
                all_preds.append(pred)
                all_labels.append(int(l))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=(class_names if class_names else None),
                yticklabels=(class_names if class_names else None))
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def save_classification_report(model, loader, device, out_txt_path, best_thresh, class_names=None):
    """
    在 loader 上做预测，先通过阈值 best_thresh 得到每个样本的预测标签，然后生成
    sklearn 的 classification_report，写到 out_txt_path。
    """
    model.eval()
    all_preds, all_labels = [], []
    all_probs = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    y_true = np.hstack(all_labels)       # shape (N,)
    y_score = np.vstack(all_probs)       # shape (N, num_classes)
    y_pred = []
    for p in y_score:
        y_pred.append(threshold_predict(p, best_thresh))
    y_pred = np.array(y_pred, dtype=int)

    report = classification_report(y_true, y_pred, target_names=(class_names if class_names else None), digits=4)
    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
    with open(out_txt_path, 'w', encoding='utf-8') as f:
        f.write(report)


def plot_multiclass_pr_curve(model, loader, device, out_path, num_classes=3, class_names=None):
    """
    在 loader（通常是验证集）上算出每个类别的 Precision-Recall 曲线，并全部画到一张图里。
    """
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            probs = torch.softmax(model(imgs), dim=1).cpu().numpy()
            all_scores.append(probs)
            all_labels.append(labels.numpy())

    y_score = np.vstack(all_scores)  # shape (N, num_classes)
    y_true = np.hstack(all_labels)   # shape (N,)
    y_true_onehot = np.eye(num_classes)[y_true]  # shape (N, num_classes)

    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        prec, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_score[:, i])
        label = class_names[i] if class_names else f"Class {i}"
        plt.plot(recall, prec, linewidth=2, label=f"{label}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Multi-class Precision-Recall Curve")
    plt.legend(loc="best")
    plt.grid(True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_loss_curve(train_losses, val_losses, out_path):
    """
    传入两个列表：train_losses 和 val_losses，它们长度应相同（等于训练的 epoch 数）。
    在一张图里绘制 “Train Loss / Val Loss 随 epoch 变化曲线”。
    """
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", marker='o')
    plt.plot(epochs, val_losses, label="Val Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs. Val Loss Curve")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def visualize_misclassified(model, loader, device, best_thresh, out_dir, class_names=None):
    """
    找到 loader（Val 或 Test）中所有误分类（p > thresh 但错分，或全部 p ≤ thresh 但错分）的样本，
    把它们对应的原始图像（经过 Normalize 的 Tensor）反归一化并保存到 out_dir 目录下，
    子目录分为 true_{label}（真实标签）、pred_{label}（预测标签）方式存放。
    文件名里可带索引以区分不同样本。
    """
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    # 为每个类别分别创建子文件夹： out_dir/true_X_pred_Y
    for true_cls in range(len(class_names) if class_names else 3):
        for pred_cls in range(len(class_names) if class_names else 3):
            if true_cls != pred_cls:
                sub = os.path.join(out_dir, f"true_{true_cls}_pred_{pred_cls}")
                os.makedirs(sub, exist_ok=True)

    idx_counter = {}
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs_device = imgs.to(device)
            logits = model(imgs_device)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            labels_np = labels.numpy()

            for i, (p, true_label) in enumerate(zip(probs, labels_np)):
                pred_label = threshold_predict(p, best_thresh)
                if pred_label != true_label:
                    folder = os.path.join(out_dir, f"true_{true_label}_pred_{pred_label}")
                    idx = idx_counter.get((true_label, pred_label), 0)
                    idx_counter[(true_label, pred_label)] = idx + 1

                    # 把 Normalize 后的 Tensor 反归一化到 [0,1] 再存为 png
                    img_tensor = imgs[i]  # 形状 [C,H,W]
                    # 反归一化（假设标准化用 [0.485,0.456,0.406],[0.229,0.224,0.225]）
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_unnorm = img_tensor * std + mean
                    img_unnorm = torch.clamp(img_unnorm, 0, 1)

                    save_path = os.path.join(folder, f"{batch_idx}_{i}.png")
                    save_image(img_unnorm, save_path)

