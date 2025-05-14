# -*- coding: utf-8 -*-
# @Author :Xiaoju
# Time : 2025/4/9 下午11:31

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


def getAUC(y_true, y_score, task):
    auc = 0.0
    if task == "multi-label":
        # 多标签任务
        for i in range(y_true.shape[1]):
            y_true_binary = y_true[:, i]
            y_score_binary = y_score[:, i]
            # 必须有正负类才能计算 AUC
            if len(np.unique(y_true_binary)) == 1:
                continue
            auc += roc_auc_score(y_true_binary, y_score_binary)
        auc /= y_true.shape[1]
    elif task == "multi-class":
        # 多分类任务（比如 pathmnist）
        auc = roc_auc_score(y_true, y_score, multi_class='ovr')
    else:
        # 二分类任务
        if len(np.unique(y_true)) == 1:
            return 0.0
        auc = roc_auc_score(y_true, y_score)
    return auc


def getACC(y_true, y_score, task, threshold=0.5):
    '''Accuracy metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_classes) for multi-label, and (n_samples,) for other tasks
    :param y_score: the predicted score of each class, shape: (n_samples, n_classes)
    :param task: the task of current dataset
    :param threshold: the threshold for multilabel and binary-class tasks
    '''
    if task == 'multi-label, binary-class':
        zero = np.zeros_like(y_score)
        one = np.ones_like(y_score)
        y_pre = np.where(y_score < threshold, zero, one)
        acc = 0
        for label in range(y_true.shape[1]):
            label_acc = accuracy_score(y_true[:, label], y_pre[:, label])
            acc += label_acc
        return acc / y_true.shape[1]
    elif task == 'binary-class':
        y_pre = np.zeros_like(y_true)
        for i in range(y_score.shape[0]):
            y_pre[i] = (y_score[i][-1] > threshold)
        return accuracy_score(y_true, y_pre)
    else:
        y_pre = np.zeros_like(y_true)
        for i in range(y_score.shape[0]):
            y_pre[i] = np.argmax(y_score[i])
        return accuracy_score(y_true, y_pre)


def save_results(y_true, y_score, outputpath):
    '''Save ground truth and scores
    :param y_true: np.ndarray, shape (n_samples,) 或 (n_samples, n_true_classes)
    :param y_score: np.ndarray, shape (n_samples, n_score_classes)
    '''
    # —— 1. 统一 true 标签的形状 —— #
    if y_true.ndim == 1:
        # 把 (n,) 变成 (n,1)
        y_true_mat = y_true.reshape(-1, 1)
    else:
        y_true_mat = y_true
    # y_score 本来就是 (n, n_classes)
    y_score_mat = y_score

    n_samples = y_score_mat.shape[0]
    n_true_cols = y_true_mat.shape[1]
    n_score_cols = y_score_mat.shape[1]

    # —— 2. 构造列名 —— #
    cols = ['id'] \
         + [f'true_{i}'  for i in range(n_true_cols)] \
         + [f'score_{i}' for i in range(n_score_cols)]

    # —— 3. 收集每一行数据 —— #
    data = []
    for idx in range(n_samples):
        row = [idx]
        # 真值
        row += y_true_mat[idx].tolist()
        # 预测分数
        row += y_score_mat[idx].tolist()
        data.append(row)

    # —— 4. 写入 DataFrame 并保存 —— #
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(outputpath, sep=',', index=False, encoding="utf_8_sig")
