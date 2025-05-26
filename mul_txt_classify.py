import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.txt_model import Transformer_CA, config as txt_config
from threeDataSet import get_dataloaders
import time
import json
import torch.cuda.amp as amp

def init_weights(m):
    """初始化模型权重"""
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)

def train_val(net, data_loader, train_optimizer, device, config, loss_criterion):
    """训练和验证函数"""
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    perfect_match = 0

    # 添加类别统计
    class_correct = torch.zeros(config['num_classes']).to(device)
    class_total = torch.zeros(config['num_classes']).to(device)

    data_bar = tqdm(data_loader)
    scaler = amp.GradScaler()

    with (torch.enable_grad() if is_train else torch.no_grad()):
        for _, data, target in data_bar:
            data = data.to(device)
            target = target.to(device).long()

            with amp.autocast():
                out = net(data)
                loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(train_optimizer)
                scaler.update()

            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            _, predicted = torch.max(out, 1)

            # 更新统计数据
            total_correct += (predicted == target).sum()
            total_samples += batch_size
            perfect_match += (predicted == target).all().sum()

            # 更新每个类别的统计
            for i in range(config['num_classes']):
                class_mask = (target == i)
                class_total[i] += class_mask.sum()
                class_correct[i] += ((predicted == target) & class_mask).sum()

            data_bar.set_description('{} Loss: {:.4f} ACC: {:.2f}%'
                                  .format('Train' if is_train else 'Test',
                                        total_loss / total_samples,
                                        100.0 * total_correct / total_samples))

    # 计算每个类别的准确率
    class_acc = (class_correct / class_total.clamp(min=1)).cpu().numpy()
    perfect_match_rate = 100.0 * perfect_match / total_samples

    return {
        'loss': total_loss / total_samples,
        'acc': 100.0 * total_correct / total_samples,
        'class_acc': class_acc,
        'perfect_match': perfect_match_rate
    }

def main():
    # 配置参数
    config = {
        'model_name': txt_config.model_name,
        'epochs': 200,
        'batch_size': 8,
        'learning_rate': txt_config.learning_rate,
        'dropout': txt_config.dropout,
        'num_classes': txt_config.num_classes,
        'embed': txt_config.embed,
        'dim_model': txt_config.dim_model,
        'hidden': txt_config.hidden,
        'num_head': txt_config.num_head,
        'num_encoder': txt_config.num_encoder,
        'optimizer': txt_config.optimizer
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    data_loaders = get_dataloaders(
        "path/to/train_metadata.csv",
        "path/to/test_metadata.csv",
        config['batch_size']
    )
    train_loader = data_loaders['train']['text']
    test_loader = data_loaders['test']['text']

    # 模型初始化
    model = Transformer_CA().to(device)
    model.apply(init_weights)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_criterion = nn.CrossEntropyLoss().to(device)
    best_acc = 0.0

    for epoch in range(1, config['epochs'] + 1):
        # 训练
        train_metrics = train_val(model, train_loader, optimizer, device, config, loss_criterion)

        # 保存最佳模型
        if train_metrics['acc'] > best_acc:
            best_acc = train_metrics['acc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': config,
                'class_acc': train_metrics['class_acc'].tolist()
            }, 'model.pth')

if __name__ == "__main__":
    main()