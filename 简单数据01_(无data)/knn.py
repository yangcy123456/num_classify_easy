import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ====================== 1. 加载所有32x32数字样本 ======================
def load_digits(folder="data_noice/seed6454"):
    samples = []
    labels = []
    for fname in os.listdir(folder):
        if not fname.endswith(".txt"):
            continue
        try:
            lbl = int(fname.split("_")[0])
        except:
            continue
        
        path = os.path.join(folder, fname)
        mat = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                row = [int(c) for c in line.strip() if c in "01"]
                if len(row) == 32:
                    mat.append(row)
        
        mat = np.array(mat)
        if mat.shape == (32, 32):
            samples.append(mat)
            labels.append(lbl)
    return np.array(samples), np.array(labels)

# ====================== 2. KNN 核心算法 ======================
def knn_classify(sample, data, labels, k=3):
    """
    输入一个样本，返回KNN预测的数字
    """
    # 展平成一维向量
    sample_flat = sample.flatten()
    data_flat = data.reshape(len(data), -1)
    
    # 计算欧氏距离
    distances = np.sqrt(np.sum((data_flat - sample_flat) ** 2, axis=1))
    
    # 取距离最近的 k 个
    top_k_idx = distances.argsort()[:k]
    top_k_labels = labels[top_k_idx]
    
    # 投票：出现最多的就是结果
    pred = np.bincount(top_k_labels).argmax()
    return pred

# ====================== 3. 主程序 ======================
if __name__ == "__main__":
    # 加载数据
    X, y_true = load_digits()
    print(f"加载完成：共 {len(X)} 个样本")

    # KNN 预测（k=3/5/7 效果最好）
    y_pred = []
    k_value = 1
    for s in X:
        res = knn_classify(s, X, y_true, k=k_value)
        y_pred.append(res)
    y_pred = np.array(y_pred)

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    digits = list(range(10))

    # 计算指标：准确率、漏检率、虚警率
    accs, misses, false_alarms = [], [], []
    total = cm.sum()
    for i in range(10):
        tp = cm[i, i]
        total_i = cm[i, :].sum()
        acc = tp / total_i if total_i != 0 else 0
        miss = (total_i - tp) / total_i if total_i != 0 else 0

        fp = cm[:, i].sum() - tp
        non_i = total - total_i
        fa = fp / non_i if non_i != 0 else 0

        accs.append(f"{acc:.2%}")
        misses.append(f"{miss:.2%}")
        false_alarms.append(f"{fa:.2%}")

    # ====================== 绘图：宋体 + 混淆矩阵+指标表 ======================
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左：混淆矩阵热力图
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=digits, yticklabels=digits, ax=ax1)
    ax1.set_xlabel("预测标签", fontsize=12)
    ax1.set_ylabel("真实标签", fontsize=12)
    ax1.set_title(f"KNN 混淆矩阵 (k={k_value})", fontsize=14)

    # 右：指标表
    table_data = [[accs[i], misses[i], false_alarms[i]] for i in range(10)]
    ax2.axis('off')
    table = ax2.table(cellText=table_data,
                      rowLabels=[str(i) for i in digits],
                      colLabels=["准确率", "漏检率", "虚警率"],
                      cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.5, 2.5)
    ax2.set_title("各类识别指标", fontsize=14)

    plt.tight_layout()
    plt.show()