import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from model.multimodal_model import TransformerRS_200_b2ck01cos, config as mm_config
from model.txt_model import Transformer_CA
from model.img_model import Net_R
from threeDataSet import get_dataloaders
import time
import json

# 配置参数
config = {
    'model_name': mm_config.model_name,
    'epochs': 200,
    'batch_size': mm_config.batch_size,
    'learning_rate': mm_config.learning_rate,
    'dropout': mm_config.dropout,
    'num_classes': mm_config.num_classes,
    'embed': mm_config.embed,
    'dim_model': mm_config.dim_model,
    'hidden': mm_config.hidden,
    'num_head': mm_config.num_head,
    'num_encoder': mm_config.num_encoder,
    'optimizer': mm_config.optimizer,
    'temperature': mm_config.temperature,
    'alpha': mm_config.alpha
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    # 初始化统计
    clss_n = torch.zeros((config['num_classes'])).to(device)
    clss_m = torch.zeros((config['num_classes'])).to(device)
    total_loss = 0.0
    total_correct = 0
    total_num = 0
    data_bar = tqdm(data_loader)

    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data_i, data_t, target in data_bar:
            data_i, data_t, target = data_i.to(device), data_t.to(device), target.to(device).long()
            xr, xd, out = net(data_i, data_t)

            # 计算交叉熵损失
            ce_loss = loss_criterion(out, target)

            # 计算双向KL散度损失
            kl_loss_1 = F.kl_div(
                F.log_softmax(xr / config['temperature'], dim=1),
                F.softmax(xd / config['temperature'], dim=1)
            ) * (config['temperature'] ** 2)

            kl_loss_2 = F.kl_div(
                F.log_softmax(xd / config['temperature'], dim=1),
                F.softmax(xr / config['temperature'], dim=1)
            ) * (config['temperature'] ** 2)

            # 组合损失
            loss = (1 - config['alpha']) * kl_loss_1 + config['alpha'] * ce_loss
            loss += (1 - config['alpha']) * kl_loss_2 + config['alpha'] * ce_loss
            loss = 0.5 * loss

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            # 更新统计信息
            total_num += target.size(0)
            total_loss += loss.item() * data_i.size(0)

            # 计算预测结果
            _, predicted = torch.max(out, 1)
            total_correct += (predicted == target).sum().item()

            # 更新每个类别的统计
            for i in range(config['num_classes']):
                clss_n[i] += ((predicted == i) & (target == i)).sum().item()
                clss_m[i] += (target == i).sum().item()

            # 更新进度条
            data_bar.set_description(
                '{} Loss: {:.4f} ACC: {:.2f}%'
                .format('Train' if is_train else 'Test',
                        total_loss / total_num,
                        100.0 * total_correct / total_num))

    # 计算每个类别的准确率
    class_acc = (clss_n / clss_m).cpu().numpy()

    return {
        'loss': total_loss / total_num,
        'acc': 100.0 * total_correct / total_num,
        'class_acc': class_acc
    }

# 数据加载
data_loaders = get_dataloaders(
    "path/to/train_metadata.csv",
    "path/to/test_metadata.csv",
    config['batch_size']
)
train_loader = data_loaders['train']['multimodal']
test_loader = data_loaders['test']['multimodal']

# 加载预训练模型
model1 = Transformer_CA()
model1.load_state_dict(torch.load("path/to/text_model.pth")['model_state_dict'])
model2 = Net_R()
model2.load_state_dict(torch.load("path/to/image_model.pth")['model_state_dict'])

# 构建多模态模型
model3 = TransformerRS_200_b2ck01cos()
model3.features = model2.features
model3.postion_embedding = model1.postion_embedding
model3.encoder = model1.encoder
model3.encoders = model1.encoders
model3.conv = model1.conv
model3.avgpool_i = model2.avgpool
model3.avgpool_t = model1.avgpool

# 冻结预训练参数
def freeze_params(model, layers_to_freeze):
    for name, param in model.named_parameters():
        if any([layer_name in name for layer_name in layers_to_freeze]):
            param.requires_grad = False

freeze_params(model3, ['features', 'postion_embedding', 'encoder', 'encoders',
                       'conv', 'avgpool_i', 'avgpool_t'])

# 设置优化器和损失函数
optimizer = optim.Adam(model3.parameters(), lr=config['learning_rate'])
net = nn.DataParallel(model3)
net.to(device)

loss_criterion = nn.CrossEntropyLoss()
best_acc = 0.0

# 主训练循环
for epoch in range(1, config['epochs'] + 1):
    # 训练
    train_metrics = train_val(net, train_loader, optimizer)

    # 保存最佳模型
    if train_metrics['acc'] > best_acc:
        best_acc = train_metrics['acc']
        torch.save({
            'epoch': epoch,
            'model_state_dict': model3.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'config': config
        }, 'model_best.pth')

print(f"Training completed. Best accuracy: {best_acc:.2f}%")