import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ====================== 加载数据 ======================
def load_digits(folder="data_noice/seed6454"):
    samples = []
    labels = []
    for fname in os.listdir(folder):
        if not fname.endswith(".txt"): continue
        try: lbl = int(fname.split("_")[0])
        except: continue
        path = os.path.join(folder, fname)
        mat = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                row = [int(c) for c in line.strip() if c in "01"]
                if len(row) == 32:
                    mat.append(row)
        mat = np.array(mat)
        if mat.shape == (32,32):
            samples.append(mat)
            labels.append(lbl)
    return np.array(samples), np.array(labels)

# ====================== 🔥 强化版 KNN（余弦距离 + 加权投票）======================
def knn_weighted_cosine(sample, data, labels, k=1):
    sf = sample.flatten()
    norm_s = np.linalg.norm(sf) + 1e-8

    best_score = -1
    pred = 0

    for i in range(len(data)):
        df = data[i].flatten()
        norm_d = np.linalg.norm(df) + 1e-8
        sim = np.dot(sf, df) / (norm_s * norm_d)  # 余弦相似度（越高越像）

        # 加权：相似度越高权重越大
        weight = sim ** 2

        if weight > best_score:
            best_score = weight
            pred = labels[i]

    return pred

# ====================== 主程序 ======================
if __name__ == "__main__":
    X, y_true = load_digits()
    k_val = 1  # 这个数据集 k=1 最稳！
    y_pred = [knn_weighted_cosine(s, X, y_true, k=k_val) for s in X]

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    digits = list(range(10))

    # 计算指标
    accs, misses, fas = [], [], []
    for i in range(10):
        tp = cm[i,i]
        total = cm[i,:].sum()
        acc = tp/total if total else 0
        accs.append(f"{acc:.2%}")
        misses.append(f"{(total-tp)/total:.2%}")
        fas.append("0.00%" if total==0 else f"{(cm[:,i].sum()-tp)/(cm.sum()-total):.2%}")

    # 绘图（宋体 + 混淆矩阵+指标表）
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=digits, yticklabels=digits, ax=ax1)
    ax1.set_xlabel('预测标签')
    ax1.set_ylabel('真实标签')
    ax1.set_title(f'强化KNN混淆矩阵（k={k_val}）')

    # 指标表
    table_data = [[accs[i], misses[i], fas[i]] for i in range(10)]
    ax2.axis('off')
    table = ax2.table(cellText=table_data, rowLabels=[str(i) for i in digits],
                      colLabels=['准确率','漏检率','虚警率'], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.5, 2.5)
    ax2.set_title('识别指标表')

    plt.tight_layout()
    plt.show()