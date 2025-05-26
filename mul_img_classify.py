import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.img_model import Net_R, config as img_config
import torch.nn.functional as F
from threeDataSet import get_dataloaders
import time
import json
import os
import torch.cuda.amp as amp

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def train_epoch(model, train_loader, optimizer, loss_criterion, scaler, config, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    data_bar = tqdm(train_loader)

    optimizer.zero_grad(set_to_none=True)

    for i, (data, _, target) in enumerate(data_bar):
        data = data.to(device)
        target = target.to(device).long()

        with amp.autocast():
            out = model(data)
            loss = loss_criterion(out, target)
            loss = loss / config['gradient_accumulation_steps']

        scaler.scale(loss).backward()

        if (i + 1) % config['gradient_accumulation_steps'] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_size = target.size(0)
        total_loss += loss.item() * batch_size * config['gradient_accumulation_steps']
        _, predicted = torch.max(out.data, 1)
        total_correct += (predicted == target).sum()
        total_samples += batch_size

        data_bar.set_description(
            'Train Loss: {:.4f} ACC: {:.2f}%'.format(
                total_loss / total_samples,
                100.0 * total_correct / total_samples
            )
        )

    return total_loss / total_samples, 100.0 * total_correct / total_samples

def main():
    # 配置参数
    config = {
        'model_name': img_config.model_name,
        'epochs': 200,
        'batch_size': 32,
        'learning_rate': img_config.learning_rate,
        'dropout': img_config.dropout,
        'num_classes': img_config.num_classes,
        'optimizer': img_config.optimizer,
        'scheduler': img_config.scheduler,
        'warmup_epochs': img_config.warmup_epochs,
        'weight_decay': img_config.weight_decay,
        'gradient_clip': img_config.gradient_clip,
        'gradient_accumulation_steps': img_config.gradient_accumulation_steps,
        'label_smoothing': img_config.label_smoothing,
        'val_frequency': img_config.val_frequency,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据加载
    data_loaders = get_dataloaders(
        "path/to/train_metadata.csv",
        "path/to/test_metadata.csv",
        config['batch_size']
    )
    train_loader = data_loaders['train']['image']
    test_loader = data_loaders['test']['image']

    # 模型初始化
    model = Net_R().to(device)
    model.apply(init_weights)

    # 优化器设置
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # 学习率调度器设置
    steps_per_epoch = len(train_loader) // config['gradient_accumulation_steps']
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        epochs=config['epochs'],
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy='cos'
    )

    # 损失函数设置
    loss_criterion = nn.CrossEntropyLoss(
        label_smoothing=config['label_smoothing']
    ).to(device)

    # 混合精度训练设置
    scaler = amp.GradScaler()
    best_acc = 0.0

    for epoch in range(1, config['epochs'] + 1):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, loss_criterion,
            scaler, config, device
        )

        # 更新学习率
        scheduler.step()

        # 保存最佳模型
        if train_acc > best_acc:
            best_acc = train_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': config
            }, 'model.pth')

if __name__ == "__main__":
    main()