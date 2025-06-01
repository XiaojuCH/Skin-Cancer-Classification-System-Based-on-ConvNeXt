# -*- coding: utf-8 -*-
# @Author :Xiaoju
# Time : 2025/5/29 下午11:33

import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def compute_best_thresholds(model, loader, device, num_classes=3):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            probs = torch.softmax(model(imgs.to(device)), dim=1).cpu().numpy()
            all_scores.append(probs)
            all_labels.append(labels.numpy())
    y_score = np.vstack(all_scores)
    y_true = np.hstack(all_labels)
    y_true_onehot = np.eye(num_classes)[y_true]

    best_thresh = []
    for i in range(num_classes):
        prec, recall, thresh = precision_recall_curve(
            y_true_onehot[:, i], y_score[:, i]
        )
        f1 = 2 * prec * recall / (prec + recall + 1e-8)
        best_idx = np.argmax(f1[:-1])
        best_thresh.append(thresh[best_idx])
    return best_thresh

def threshold_predict(probs, best_thresh):
    mask = [p > t for p, t in zip(probs, best_thresh)]
    if any(mask):
        cands = [i for i, m in enumerate(mask) if m]
        return cands[np.argmax([probs[i] for i in cands])]
    else:
        return np.argmax(probs)

def plot_confusion(model, loader, device, best_thresh, out_path, title):
    # Collect preds & labels
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            probs = torch.softmax(model(imgs.to(device)), dim=1).cpu().numpy()
            for p, l in zip(probs, labels.numpy()):
                all_preds.append(threshold_predict(p, best_thresh))
                all_labels.append(l)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
