# -*- coding: utf-8 -*-
# @Author : Xiaoju
# Time    : 2025/4/9 下午11:26

import warnings
import os
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import trange
from torch.amp import GradScaler, autocast

from Confusion_matrix import (
    compute_best_thresholds,
    plot_confusion,
    save_classification_report,
    plot_multiclass_pr_curve,
    plot_loss_curve,
    visualize_misclassified
)

# —— 过滤警告 ——#
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torchvision\\.models\\._utils"
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="You are using `torch.load` with `weights_only=False`"
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="kornia\\.feature\\.lightglue"
)

# 自定义模块
from models import get_model
from dataset import ISIC2017Dataset
from evaluator import getAUC, getACC, save_results


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-CE_loss)
        F_loss = (1 - pt) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


# === MixUp utility ===
def mixup_data(x, y, alpha=0.4):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================

def save_checkpoint(model, epoch, val_auc, save_dir, name, model_name):
    state = {'net': model.state_dict(), 'epoch': epoch,
             'val_auc': val_auc, 'model_name': model_name}
    path = os.path.join(save_dir, f'{name}.pth')
    torch.save(state, path)
    return path


# --------- Ensemble test ----------
def test_ensemble(models, loader, device, output_root, result_name):
    ys, yps = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        with autocast(device_type='cuda'):
            probs = sum(torch.softmax(m(imgs), dim=1) for m in models) / len(models)
        yps.append(probs.cpu().numpy())
        ys.append(labels.numpy())
    y_true = np.concatenate(ys, axis=0)
    y_score = np.concatenate(yps, axis=0)
    auc = getAUC(y_true, y_score, task="multi-class")
    acc = getACC(y_true, y_score, task="multi-class")
    print(f'[{result_name}] Ensemble Test AUC: {auc:.4f}  ACC: {acc:.4f}')
    os.makedirs(output_root, exist_ok=True)
    save_results(y_true, y_score, os.path.join(output_root, result_name))
    return auc, acc


def test(model, loader, device, output_root, result_name, tta_list=None):
    """
    model: 已经 model.to(device) 并 model.eval()。
    loader: DataLoader
    tta_list: 如果传入了多个 Tensor 级别的增强函数，就对同一 batch 做多次前向并求平均；
              如果为 None，就只做一次前向 softmax。
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            if tta_list:
                tta_probs_list = []
                for tf_fn in tta_list:
                    aug_imgs = tf_fn(imgs)
                    with autocast(device_type='cuda'):
                        logits = model(aug_imgs)
                    probs = torch.softmax(logits.float(), dim=1).cpu()
                    tta_probs_list.append(probs)
                stacked = torch.stack(tta_probs_list, dim=0)
                probs_mean = stacked.mean(dim=0)
                probs_np = probs_mean.numpy()
            else:
                with autocast(device_type='cuda'):
                    logits = model(imgs)
                probs_np = torch.softmax(logits.float(), dim=1).cpu().numpy()
            all_probs.append(probs_np)
            all_labels.append(labels.numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_score = np.concatenate(all_probs, axis=0)
    row_sums = y_score.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-4), \
        f"y_score 每行之和应为 1，但检测到前 5 行之和：{row_sums[:5]}"

    auc = getAUC(y_true, y_score, task="multi-class")
    acc = getACC(y_true, y_score, task="multi-class")
    print(f'[{result_name}] Test AUC: {auc:.4f}  ACC: {acc:.4f}')
    os.makedirs(output_root, exist_ok=True)
    save_results(y_true, y_score, os.path.join(output_root, result_name))
    return auc, acc


# EarlyStopping 监控 AUC
class EarlyStopping:
    """早停，监控验证 AUC，并保存最佳模型"""

    def __init__(self, patience=20, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = 0.0  # 最佳 AUC
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True


def main(input_root, output_root, num_epoch, model_name):
    # ================== 超参 ================== #
    batch_size = 64
    grad_accum_steps = 2  # 梯度累积步数
    base_lr = 2e-4
    wd = 3e-3

    # ================== 路径配置 ================== #
    ckpt_dir = os.path.join(output_root, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    global_best_path = os.path.join(ckpt_dir, 'global_best.pth')

    # ================== 检查点兼容性处理 ================== #
    global_best_auc = 0.0
    compatible_global_best = False
    if os.path.exists(global_best_path):
        try:
            gb = torch.load(global_best_path, map_location='cpu')
            if 'model_name' in gb and gb['model_name'] == model_name:
                global_best_auc = gb['val_auc']
                compatible_global_best = True
                print(f"Loaded compatible global best AUC = {global_best_auc:.4f}")
            else:
                print(f"⚠️ 发现不兼容检查点: {gb.get('model_name', '未知模型')} (当前使用 {model_name})")
        except Exception as e:
            print(f"⚠️ 加载检查点失败: {str(e)}")

    # ================== 数据增强 ================== #
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.15), ratio=(0.5, 2.0)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    # ================== 数据集 & Oversample ================== #
    train_ds = ISIC2017Dataset(input_root, 'train', train_tf)
    val_ds = ISIC2017Dataset(input_root, 'val', val_tf)
    test_ds = ISIC2017Dataset(input_root, 'test', val_tf)

    # —— 先统计 train_ds 中各类样本数 ——#
    labels = np.array(train_ds.labels)  # 每个样本的真实类别 0/1/2
    class_counts = np.bincount(labels)  # 例如 [374, 254, 1372]

    # —— 按 1/log(count+2) 计算每个类别的初始权重 ——#
    class_weights = 1.0 / np.log(class_counts + 2.0)  # shape=(3,)
    # —— 再把 Melanoma（类别 0）的权重额外 ×1.3 ——#
    class_weights[0] *= 2.0
    # —— 最后将全部类别权重归一化 ——#
    class_weights = class_weights / class_weights.sum()

    # —— 为每个样本赋予对应的采样权重 ——#
    sample_weights = np.array([class_weights[c] for c in labels], dtype=np.float32)

    # —— 用 WeightedRandomSampler 构造一个带替换的 sampler ——#
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,  # 与 class_weights 成比例
        num_samples=len(sample_weights),
        replacement=True
    )

    # —— 把 sampler 传给 DataLoader，不再使用 shuffle=True ——#
    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    val_loader = data.DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4
    )
    test_loader = data.DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4
    )

    # ================== 权重 & 损失 ================== #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_counts = torch.tensor(class_counts, dtype=torch.float32)
    w = 1.0 / torch.log(raw_counts + 2.0)
    w[0] *= 2.0  # Melanoma 权重再放大
    w = w / w.sum()
    w = w.to(device)

    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    # 如果后面想测 FocalLoss，就把上面一行改成：
    # criterion = FocalLoss(alpha=w, gamma=1.5)

    early_stop = EarlyStopping(patience=20, verbose=True)

    # ================== 模型 & 优化器 & Scheduler ================== #
    print("Using device:", device)
    model = get_model(model_name, num_classes=3).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=wd
    )

    # 使用带 warmup 的余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=15,  # 每15个epoch重启学习率
        T_mult=2,
        eta_min=1e-6  # 最小学习率
    )
    # # 先定义一个线性 warmup scheduler
    # warmup_epochs = 10
    #
    # def lr_lambda(epoch):
    #     if epoch < warmup_epochs:
    #         return (epoch + 1) / warmup_epochs
    #     else:
    #         return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (num_epoch - warmup_epochs)))
    #
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # total_steps = num_epoch * len(train_loader)
    # scheduler = optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=8e-4,
    #     total_steps=total_steps,
    #     pct_start=0.35,
    #     anneal_strategy='cos',
    #     div_factor=25,
    #     final_div_factor=1e-3
    # )
    scaler = GradScaler(device='cuda')

    # ========== 记录 Loss 曲线 ==========
    train_losses = []
    val_losses = []

    run_best_auc = 0.0

    # ================== 训练循环 ================== #
    for epoch in trange(1, num_epoch + 1, desc="Epochs"):
        model.train()
        optimizer.zero_grad()

        total_loss = 0.0
        for step, (imgs, labels) in enumerate(train_loader, 1):
            imgs, labels = imgs.to(device), labels.to(device)
            # MixUp：70% 概率，对整个 batch 做混合，alpha=0.4
            if epoch <= 40 and np.random.rand() < 0.7:
                imgs_m, y_a, y_b, lam = mixup_data(imgs, labels, alpha=0.4)
                with autocast(device_type='cuda'):
                    logits = model(imgs_m)
                    loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            else:
                with autocast(device_type='cuda'):
                    logits = model(imgs)
                    loss = criterion(logits, labels)

            loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
            total_loss += loss.item()

            if step % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        scheduler.step(epoch)
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ================== 验证阶段 ================== #
        model.eval()
        val_loss = 0.0
        ys_val, yps_val = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with autocast(device_type='cuda'):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                val_loss += loss.item()

                probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
                ys_val.append(labels.cpu().numpy())
                yps_val.append(probs)

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        y_true_val = np.concatenate(ys_val, axis=0)
        y_score_val = np.concatenate(yps_val, axis=0)
        val_auc = getAUC(y_true_val, y_score_val, task="multi-class")
        val_acc = getACC(y_true_val, y_score_val, task="multi-class")
        print(
            f"[Epoch {epoch}/{num_epoch}] Val AUC: {val_auc:.4f} Val ACC:{val_acc:.4f} Train Loss: {avg_train_loss:.10f}")

        # 保存 run-best
        if val_auc > run_best_auc + 1e-5:
            run_best_auc = val_auc
            run_best_path = save_checkpoint(model, epoch, val_auc, ckpt_dir, 'run_best', model_name)
            print(f">>> New run-best at epoch {epoch}, AUC={val_auc:.4f} ACC:{val_acc:.4f}")

        early_stop(val_auc, epoch)
        if early_stop.early_stop:
            print(f"\n⏹️ EarlyStopping triggered at epoch {early_stop.best_epoch}")
            break

    # ================== Testing FINAL ================== #
    print("\n===== Testing FINAL model (last epoch) =====")
    test(model, test_loader, device, output_root, 'test_final.csv', tta_list=None)

    # Testing RUN-BEST (with stronger TTA)
    print("\n===== Testing RUN-BEST model =====")
    ckpt = torch.load(os.path.join(ckpt_dir, 'run_best.pth'))
    model.load_state_dict(ckpt['net'])
    tta_list = [
        lambda x: x,
        lambda x: torch.flip(x, [-1]),
        lambda x: torch.flip(x, [-2]),
        lambda x: torch.flip(torch.flip(x, [-1]), [-2]),  # 同时水平+竖直
        lambda x: x.rot90(1, dims=[-2, -1]),
        lambda x: x.rot90(2, dims=[-2, -1]),
        lambda x: x.rot90(3, dims=[-2, -1]),
        lambda x: (x + torch.randn_like(x) * 0.01).clamp(0, 1),  # 加轻微噪声
    ]

    test(model, test_loader, device, output_root, 'test_run_best.csv', tta_list=tta_list)

    # 移除 Class-wise P/R/F1 打印（已由 save_classification_report 输出至文件）

    # 计算并打印各类最佳阈值
    best_thresh = compute_best_thresholds(model, val_loader, device, num_classes=3)
    # 手动将 Melanoma 阈值在原来基础上加 0.08（并确保不超过 1.0）
    best_thresh[0] = min(best_thresh[0] + 0.05, 1.0)
    print("Best thresholds per class:", best_thresh)

    plot_confusion(
        model,
        train_loader,
        device,
        best_thresh,
        out_path=os.path.join(output_root, "confusion_train_runbest.png"),
        title="Train Confusion",
        class_names=["Melanoma", "Nevus", "SeborrheicKeratosis"]
    )
    plot_confusion(
        model,
        val_loader,
        device,
        best_thresh,
        out_path=os.path.join(output_root, "confusion_val_runbest.png"),
        title="Val Confusion",
        class_names=["Melanoma", "Nevus", "SeborrheicKeratosis"]
    )
    plot_confusion(
        model,
        test_loader,
        device,
        best_thresh,
        out_path=os.path.join(output_root, "confusion_test_runbest.png"),
        title="Test Confusion",
        class_names=["Melanoma", "Nevus", "SeborrheicKeratosis"]
    )

    # 生成可视化报告
    print("\n===== 生成可视化报告 =====")
    class_names = ["Melanoma", "Nevus", "SeborrheicKeratosis"]
    best_model = get_model(model_name, num_classes=3).to(device)
    ckpt2 = torch.load(os.path.join(ckpt_dir, 'run_best.pth'), map_location=device)
    best_model.load_state_dict(ckpt2['net'])
    best_model.eval()

    save_classification_report(
        best_model, val_loader, device,
        out_txt_path=os.path.join(output_root, "report_val_runbest.txt"),
        best_thresh=best_thresh, class_names=class_names
    )
    save_classification_report(
        best_model, test_loader, device,
        out_txt_path=os.path.join(output_root, "report_test_runbest.txt"),
        best_thresh=best_thresh, class_names=class_names
    )

    plot_multiclass_pr_curve(
        best_model, val_loader, device,
        out_path=os.path.join(output_root, "pr_curve_val_runbest.png"),
        num_classes=3, class_names=class_names
    )

    plot_loss_curve(
        train_losses, val_losses,
        out_path=os.path.join(output_root, "loss_curve.png")
    )

    visualize_misclassified(
        best_model, val_loader, device, best_thresh,
        out_dir=os.path.join(output_root, "errors_val_runbest"),
        class_names=class_names
    )
    visualize_misclassified(
        best_model, test_loader, device, best_thresh,
        out_dir=os.path.join(output_root, "errors_test_runbest"),
        class_names=class_names
    )

    print("所有可视化已生成，保存在：", output_root)

    # 加载并测试 Global Best
    if compatible_global_best:
        try:
            gb_ckpt = torch.load(global_best_path)
            state_dict = gb_ckpt['net']
            filtered = {k.replace("model.", ""): v for k, v in state_dict.items()
                        if k.replace("model.", "") in model.state_dict()
                        and v.shape == model.state_dict()[k.replace("model.", "")].shape}
            match_rate = len(filtered) / len(model.state_dict())
            model.load_state_dict(filtered, strict=False)
            print(f"\n===== Testing GLOBAL‐BEST model (match={match_rate:.1%}) =====")
            test(model, test_loader, device, output_root, 'test_global_best.csv')
        except Exception as e:
            print(f"⚠️ 加载全局最佳失败: {str(e)}")
    else:
        print("\n⚠️ 无兼容的全局最佳检查点")

    ckpt = torch.load(os.path.join(output_root, "checkpoints/run_best.pth"))
    model.load_state_dict(ckpt['net'])
    model.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', required=True, help="数据集根目录")
    parser.add_argument('--output_root', default='./output', help="结果保存目录")
    parser.add_argument('--num_epoch', type=int, default=80, help="训练轮数")
    parser.add_argument('--model', default='convnext_small',
                        choices=['resnet18', 'resnet50', 'efficientnet_b0', 'efficientnet_b1',
                                 'convnext_tiny', 'convnext_small', 'convnext_base', 'densenet121'],
                        help="网络型号")
    args = parser.parse_args()
    main(args.input_root, args.output_root, args.num_epoch, args.model)
