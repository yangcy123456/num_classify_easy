# 🔥 修复OpenMP冲突（必加，解决运行崩溃）
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
# 仅新增绘图必需库，无sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader

# ===================== 固定随机种子（保证可复现） =====================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===================== 超参数 =====================
SEED = 42                # 模型训练随机种子
SPLIT_SEED = 123         # ✅ 数据划分独立种子（与噪声区分）
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 10
# ✅ 替换为【偏移+翻转】噪声参数（和你KNN完全一致）
SHIFT_MAX = 1            # 图像最大偏移像素
FLIP_PROB = 0.02         # 像素翻转概率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = list(range(10)) # MNIST标签

set_seed(SEED)  # 固定训练相关种子

# ===================== 🔥 噪声模块：偏移+像素翻转（替换原高斯噪声） =====================
def add_shift_flip_noise(image_batch):
    """批量添加 图像偏移 + 像素翻转 噪声"""
    noisy_batch = []
    for img in image_batch:
        # 重塑为28x28
        img = img.view(28, 28)
        # 随机偏移
        dx, dy = np.random.randint(-SHIFT_MAX, SHIFT_MAX + 1, 2)
        shifted = torch.zeros_like(img)
        
        # 偏移计算
        src_x1, src_x2 = max(0, -dx), 28 - max(0, dx)
        src_y1, src_y2 = max(0, -dy), 28 - max(0, dy)
        dst_x1, dst_x2 = max(0, dx), 28 + min(0, dx)
        dst_y1, dst_y2 = max(0, dy), 28 + min(0, dy)
        shifted[dst_x1:dst_x2, dst_y1:dst_y2] = img[src_x1:src_x2, src_y1:src_y2]
        
        # 像素随机翻转
        mask = torch.rand(28, 28, device=DEVICE) < FLIP_PROB
        noisy = torch.where(mask, 1 - shifted, shifted)
        
        noisy_batch.append(noisy.flatten())
    
    return torch.stack(noisy_batch)

# ===================== 数据加载（✅ 修改为读取 mnist.csv 并按 7:2:1 划分） =====================
def load_mnist_data():
    df = pd.read_csv("mnist.csv", header=None)
    images = df.iloc[:, 1:].values.astype(np.float32) / 255.0
    labels = df.iloc[:, 0].values.astype(np.int64)

    # ✅ 使用独立种子打乱全部数据
    rng = np.random.RandomState(SPLIT_SEED)
    indices = np.arange(len(images))
    rng.shuffle(indices)
    images, labels = images[indices], labels[indices]

    # 计算划分点
    total = len(images)
    train_end = int(total * 0.7)
    val_end = int(total * 0.9)

    train_imgs = images[:train_end]
    train_labs = labels[:train_end]
    val_imgs = images[train_end:val_end]
    val_labs = labels[train_end:val_end]
    test_imgs = images[val_end:]
    test_labs = labels[val_end:]

    print(f"数据划分完成（种子={SPLIT_SEED}）：训练集 {len(train_imgs)} | 验证集 {len(val_imgs)} | 测试集 {len(test_imgs)}")

    return (torch.tensor(train_imgs), torch.tensor(train_labs),
            torch.tensor(val_imgs),   torch.tensor(val_labs),
            torch.tensor(test_imgs),  torch.tensor(test_labs))

# ===================== 数据加载器（✅ 调整为三个集合） =====================
def create_dataloaders(train_imgs, train_labs, val_imgs, val_labs, test_imgs, test_labs):
    train_dataset = TensorDataset(train_imgs, train_labs)
    val_dataset = TensorDataset(val_imgs, val_labs)
    test_dataset = TensorDataset(test_imgs, test_labs)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader

# ===================== CNN模型 =====================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ===================== 训练 =====================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # ✅ 训练时添加【偏移+翻转】噪声
        images = add_shift_flip_noise(images)
        
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ===================== 评估（返回准确率+全部预测结果） =====================
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            all_preds.append(pred)
            all_labels.append(labels)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return 100 * correct / total, all_preds, all_labels

# ===================== 保存模型（按seed保存） =====================
def save_model(model, seed):
    save_dir = f"out/{SPLIT_SEED}_{seed}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "mnist_cnn.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n模型已保存到：{save_path}")

# ===================== 纯手写混淆矩阵（无sklearn） =====================
def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

# ===================== 计算分类指标 =====================
def calculate_class_metrics(cm):
    acc_list = []
    miss_list = []
    false_alarm_list = []
    for i in range(10):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP
        
        acc = TP / (TP + FN) if (TP + FN) != 0 else 0
        miss_rate = FN / (TP + FN) if (TP + FN) != 0 else 0
        false_alarm = FP / (TN + FP) if (TN + FP) != 0 else 0
        
        acc_list.append(acc)
        miss_list.append(miss_rate)
        false_alarm_list.append(false_alarm)
    return acc_list, miss_list, false_alarm_list

# ===================== 绘图代码（✅ 改为 3×2 子图：训练集、验证集、测试集） =====================
def plot_all_results(train_cm, train_acc, train_miss, train_false,
                     val_cm, val_acc, val_miss, val_false,
                     test_cm, test_acc, test_miss, test_false):
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 宋体
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(3, 2, figsize=(14, 20))  # 3行2列，高度加大

    datasets = [
        ("训练集", train_cm, train_acc, train_miss, train_false, axes[0]),
        ("验证集", val_cm,   val_acc,   val_miss,   val_false,   axes[1]),
        ("测试集", test_cm,  test_acc,  test_miss,  test_false,  axes[2])
    ]

    for title, cm, acc, miss, false_alarm, (ax_left, ax_right) in datasets:
        # 左图：混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes, ax=ax_left)
        ax_left.set_xlabel('预测标签', fontsize=12)
        ax_left.set_ylabel('真实标签', fontsize=12)
        ax_left.set_title(f'CNN（偏移+翻转噪声） - {title}', fontsize=14, pad=10)

        # 右图：指标表格
        table_data = [
            [f"{acc[i]:.2%}", f"{miss[i]:.2%}", f"{false_alarm[i]:.2%}"]
            for i in range(10)
        ]
        row_labels = [str(i) for i in classes]
        col_labels = ['准确率', '漏检率', '虚警率']

        ax_right.axis('off')
        table = ax_right.table(cellText=table_data,
                               rowLabels=row_labels,
                               colLabels=col_labels,
                               cellLoc='center',
                               loc='center',
                               colWidths=[0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        ax_right.set_title(f'{title}各类指标', fontsize=14, pad=10)

    plt.tight_layout()
    plt.show()

# ===================== 主函数 =====================
def main():
    # ✅ 加载并划分数据
    train_imgs, train_labs, val_imgs, val_labs, test_imgs, test_labs = load_mnist_data()
    train_loader, val_loader, test_loader = create_dataloaders(
        train_imgs, train_labs, val_imgs, val_labs, test_imgs, test_labs
    )

    model = CNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"设备：{DEVICE} | 训练种子：{SEED} | 数据划分种子：{SPLIT_SEED}")
    print(f"噪声：偏移±{SHIFT_MAX}像素 | 像素翻转概率{FLIP_PROB}\n")

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_acc, _, _ = evaluate(model, val_loader, DEVICE)
        test_acc, _, _ = evaluate(model, test_loader, DEVICE)
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {loss:.4f} | Val Acc: {val_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    save_model(model, SEED)

    # ✅ 绘制三合一混淆矩阵图表
    print("\n📊 生成训练集、验证集、测试集混淆矩阵与指标图表...")
    # 训练集评估
    train_acc, train_preds, train_labels = evaluate(model, train_loader, DEVICE)
    train_cm = compute_confusion_matrix(train_labels, train_preds)
    train_acc_list, train_miss_list, train_false_list = calculate_class_metrics(train_cm)

    # 验证集评估
    val_acc, val_preds, val_labels = evaluate(model, val_loader, DEVICE)
    val_cm = compute_confusion_matrix(val_labels, val_preds)
    val_acc_list, val_miss_list, val_false_list = calculate_class_metrics(val_cm)

    # 测试集评估
    test_acc, test_preds, test_labels = evaluate(model, test_loader, DEVICE)
    test_cm = compute_confusion_matrix(test_labels, test_preds)
    test_acc_list, test_miss_list, test_false_list = calculate_class_metrics(test_cm)

    plot_all_results(train_cm, train_acc_list, train_miss_list, train_false_list,
                     val_cm, val_acc_list, val_miss_list, val_false_list,
                     test_cm, test_acc_list, test_miss_list, test_false_list)
    print("✅ 全部运行完成！")

if __name__ == "__main__":
    main()