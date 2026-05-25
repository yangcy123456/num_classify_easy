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
SEED = 42                # 你可以随便改，保存路径会自动变
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 10
# ✅ 替换为【偏移+翻转】噪声参数（和你KNN完全一致）
SHIFT_MAX = 1            # 图像最大偏移像素
FLIP_PROB = 0.02         # 像素翻转概率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = list(range(10)) # MNIST标签

set_seed(SEED)  # 固定种子

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

# ===================== 数据加载 =====================
def load_mnist_data():
    train_df = pd.read_csv("mnist_train.csv")
    test_df = pd.read_csv("mnist_test.csv")

    train_images = train_df.iloc[:, 1:].values.astype(np.float32) / 255.0
    train_labels = train_df.iloc[:, 0].values.astype(np.int64)
    test_images = test_df.iloc[:, 1:].values.astype(np.float32) / 255.0
    test_labels = test_df.iloc[:, 0].values.astype(np.int64)

    train_images = torch.tensor(train_images)
    train_labels = torch.tensor(train_labels)
    test_images = torch.tensor(test_images)
    test_labels = torch.tensor(test_labels)

    return train_images, train_labels, test_images, test_labels

# ===================== 数据加载器 =====================
def create_dataloaders(train_images, train_labels, test_images, test_labels):
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader

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
    save_dir = f"out/{seed}"
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

# ===================== 绘图代码（完全保留） =====================
def plot_result(cm, acc_list, miss_list, false_alarm_list):
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 宋体
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_xlabel('预测标签', fontsize=12)
    ax1.set_ylabel('真实标签', fontsize=12)
    ax1.set_title('CNN（偏移+翻转噪声）', fontsize=14, pad=10)

    # 右图：指标表格
    table_data = [
        [f"{acc_list[i]:.2%}", f"{miss_list[i]:.2%}", f"{false_alarm_list[i]:.2%}"]
        for i in range(10)
    ]
    row_labels = [str(i) for i in classes]
    col_labels = ['准确率', '漏检率', '虚警率']

    ax2.axis('off')
    table = ax2.table(cellText=table_data,
                      rowLabels=row_labels,
                      colLabels=col_labels,
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    ax2.set_title('各类识别指标', fontsize=14, pad=10)

    plt.tight_layout()
    plt.show()

# ===================== 主函数 =====================
def main():
    train_images, train_labels, test_images, test_labels = load_mnist_data()
    train_loader, test_loader = create_dataloaders(train_images, train_labels, test_images, test_labels)

    model = CNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"设备：{DEVICE} | 随机种子：{SEED}")
    print(f"噪声：偏移±{SHIFT_MAX}像素 | 像素翻转概率{FLIP_PROB}\n")

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        acc, _, _ = evaluate(model, test_loader, DEVICE)
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {loss:.4f} | Acc: {acc:.2f}%")

    save_model(model, SEED)

    # 自动绘制混淆矩阵+指标表
    print("\n📊 生成测试集混淆矩阵与指标图表...")
    _, all_preds, all_labels = evaluate(model, test_loader, DEVICE)
    cm = compute_confusion_matrix(all_labels, all_preds)
    acc_list, miss_list, false_alarm_list = calculate_class_metrics(cm)
    plot_result(cm, acc_list, miss_list, false_alarm_list)
    print("✅ 全部运行完成！")

if __name__ == "__main__":
    main()