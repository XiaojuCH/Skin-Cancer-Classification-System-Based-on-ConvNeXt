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
from torch.amp import GradScaler
from torch.amp import autocast

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


def save_checkpoint(model, epoch, val_auc, save_dir, name, model_name):
    """保存检查点并记录模型架构"""
    state = {
        'net': model.state_dict(),
        'epoch': epoch,
        'val_auc': val_auc,
        'model_name': model_name
    }
    path = os.path.join(save_dir, f'{name}.pth')
    torch.save(state, path)
    return path


def test(model, loader, device, output_root, result_name):
    model.eval()
    ys, yps = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            with autocast(device_type='cuda'):  # 混合精度推理
                probs = torch.softmax(model(imgs), dim=1).cpu().numpy()
            yps.append(probs)
            ys.append(labels.numpy())
    y_true = np.concatenate(ys, axis=0)
    y_score = np.concatenate(yps, axis=0)

    auc = getAUC(y_true, y_score, task="multi-class")
    acc = getACC(y_true, y_score, task="multi-class")
    print(f'[{result_name}] Test AUC: {auc:.4f}  ACC: {acc:.4f}')

    os.makedirs(output_root, exist_ok=True)
    save_results(y_true, y_score, os.path.join(output_root, result_name))
    return auc, acc


def main(input_root, output_root, num_epoch, model_name):
    # ================== 显存安全配置 ================== #
    # 根据显存容量动态调整 (RTX 4060 8GB 推荐 batch_size=48, 12GB 可调至64)
    batch_size = 48
    grad_accum_steps = 2  # 梯度累积步数
    base_lr = 1e-4
    wd = 1e-4

    # 自动检测显存容量
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"GPU 显存总量: {total_mem:.1f}GB")
        if total_mem > 10:  # 12GB 显存
            batch_size = 64
            grad_accum_steps = 1

    # ================== 路径配置 ================== #
    ckpt_dir = os.path.join(output_root, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    global_best_path = os.path.join(ckpt_dir, 'global_best.pth')

    # ================== 检查点兼容性处理 ================== #
    global_best_auc = 0.0
    compatible_global_best = False

    if os.path.exists(global_best_path):
        try:
            gb = torch.load(global_best_path)
            if 'model_name' in gb and gb['model_name'] == model_name:
                global_best_auc = gb['val_auc']
                compatible_global_best = True
                print(f"Loaded compatible global best AUC = {global_best_auc:.4f}")
            else:
                print(f"⚠️ 发现不兼容检查点: {gb.get('model_name', '未知模型')} (当前使用 {model_name})")
        except Exception as e:
            print(f"⚠️ 加载检查点失败: {str(e)}")

    # ================== 数据增强 (显存优化版) ================== #
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # 随机缩放裁剪
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

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ================== 数据集加载 ================== #
    train_ds = ISIC2017Dataset(input_root, 'train', train_tf)
    val_ds = ISIC2017Dataset(input_root, 'val', val_tf)
    test_ds = ISIC2017Dataset(input_root, 'test', val_tf)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True  # 加速数据加载
    )
    val_loader = data.DataLoader(
        val_ds,
        batch_size=batch_size * 2,  # 验证使用更大batch
        shuffle=False,
        num_workers=4
    )
    test_loader = data.DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4
    )

    # ================== 模型初始化 ================== #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = get_model(model_name, num_classes=3)
    model.to(device)

    # ================== 优化器配置修正 ================== #
    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=wd
    )

    # 使用带 warmup 的余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # 每10个epoch重启学习率
        T_mult=2,
        eta_min=1e-6  # 最小学习率
    )
    scaler = GradScaler(device='cuda')  # 混合精度梯度缩放

    # ================== 训练循环 (完整实现) ================== #
    run_best_auc = 0.0
    for epoch in trange(1, num_epoch + 1, desc="Epochs"):
        model.train()
        optimizer.zero_grad()

        total_loss = 0.0
        for step, (imgs, labels) in enumerate(train_loader, 1):
            imgs, labels = imgs.to(device), labels.to(device)

            with autocast(device_type='cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, labels) / grad_accum_steps

            scaler.scale(loss).backward()
            total_loss += loss.item()

            # 梯度累积更新
            if step % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # 学习率调度
        scheduler.step()

        # 打印训练损失
        # print(f"[Epoch {epoch}] Train Loss: {total_loss / len(train_loader):.10f}")

        # ================== 验证阶段 ================== #
        model.eval()
        ys_val, yps_val = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                with autocast(device_type='cuda'):
                    probs = torch.softmax(model(imgs), dim=1).cpu().numpy()
                yps_val.append(probs)
                ys_val.append(labels.numpy())

        y_true_val = np.concatenate(ys_val, axis=0)
        y_score_val = np.concatenate(yps_val, axis=0)
        val_auc = getAUC(y_true_val, y_score_val, task="multi-class")
        val_acc = getACC(y_true_val, y_score_val, task="multi-class")
        print(f"[Epoch {epoch}/{num_epoch}] Val AUC: {val_auc:.4f} Val ACC:{val_acc:.4f} Train Loss: {total_loss / len(train_loader):.10f}")

        # 保存当前最佳
        if val_auc > run_best_auc:
            run_best_auc = val_auc
            run_best_path = save_checkpoint(model, epoch, val_auc, ckpt_dir, 'run_best', model_name)
            print(f">>> New run-best at epoch {epoch}, AUC={val_auc:.4f} ACC:{val_acc:.4f}")

    # ================== 最终测试 ================== #
    print("\n===== Testing FINAL model (last epoch) =====")
    test(model, test_loader, device, output_root, 'test_final.csv')

    # 测试 run 最佳模型
    print("\n===== Testing RUN-BEST model =====")
    ckpt = torch.load(os.path.join(ckpt_dir, 'run_best.pth'))
    model.load_state_dict(ckpt['net'])
    test(model, test_loader, device, output_root, 'test_run_best.csv')

    # ================== 更新全局最佳 ================== #
    if run_best_auc > global_best_auc:
        shutil.copy(os.path.join(ckpt_dir, 'run_best.pth'), global_best_path)
        print(f"\n>>> Run-best AUC {run_best_auc:.4f} > global-best {global_best_auc:.4f}")
        print(f">>> Updated global_best.pth")
    else:
        print(f"\nGlobal best ({global_best_auc:.4f}) remains better than this run ({run_best_auc:.4f})")

    # ================== 安全加载全局最佳 ================== #
    if compatible_global_best:
        try:
            gb_ckpt = torch.load(global_best_path)
            state_dict = gb_ckpt['net']

            # 键名兼容处理
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

            # 严格参数过滤
            model_dict = model.state_dict()
            filtered = {k: v for k, v in state_dict.items()
                        if k in model_dict and v.shape == model_dict[k].shape}

            # 加载并报告匹配率
            match_rate = len(filtered) / len(model_dict)
            model.load_state_dict(filtered, strict=False)
            print(f"\n===== Testing GLOBAL-BEST model (参数匹配率 {match_rate:.1%}) =====")
            test(model, test_loader, device, output_root, 'test_global_best.csv')
        except Exception as e:
            print(f"\n⚠️ 加载全局最佳失败: {str(e)}")
    else:
        print("\n⚠️ 无兼容的全局最佳检查点")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', required=True, help="数据集根目录")
    parser.add_argument('--output_root', default='./output', help="结果保存目录")
    parser.add_argument('--num_epoch', type=int, default=64, help="训练轮数")
    parser.add_argument('--model', default='convnext_small',
                        choices=['resnet18', 'resnet50', 'efficientnet_b0', 'efficientnet_b1',
                                 'convnext_tiny', 'convnext_small', 'densenet121', 'convnext_base'],
                        help="网络型号")
    args = parser.parse_args()

    # 权重处理
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls_counts = torch.tensor([374, 254, 1372], dtype=torch.float32)
    cls_weights = 1.0 / cls_counts
    cls_weights = cls_weights / cls_weights.sum()
    cls_weights = cls_weights.to(device)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss(weight=cls_weights)
    main(args.input_root, args.output_root, args.num_epoch, args.model)
