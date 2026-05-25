import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
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
NOISE_STD = 0.1          # 噪声强度（高斯噪声）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(SEED)  # 固定种子

# ===================== 噪声模块 =====================
def add_gaussian_noise(images, mean=0., std=0.1):
    """给图像加高斯噪声"""
    noise = torch.randn_like(images) * std + mean
    noisy_images = images + noise
    noisy_images = torch.clamp(noisy_images, 0., 1.)  # 保持在0~1之间
    return noisy_images

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
        
        # ✅ 训练时自动加噪声
        images = add_gaussian_noise(images, std=NOISE_STD)
        
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ===================== 评估 =====================
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return 100 * correct / total

# ===================== 保存模型（按seed保存） =====================
def save_model(model, seed):
    save_dir = f"out/{seed}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "mnist_cnn.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n模型已保存到：{save_path}")

# ===================== 主函数 =====================
def main():
    train_images, train_labels, test_images, test_labels = load_mnist_data()
    train_loader, test_loader = create_dataloaders(train_images, train_labels, test_images, test_labels)

    model = CNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"设备：{DEVICE} | 随机种子：{SEED} | 噪声强度：{NOISE_STD}\n")

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        acc = evaluate(model, test_loader, DEVICE)
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {loss:.4f} | Acc: {acc:.2f}%")

    save_model(model, SEED)

if __name__ == "__main__":
    main()