import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ====================== 加载数据 ======================
def load_all_samples(folder_path="data/testDigits"):
    samples = []
    true_labels = []
    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue
        try:
            label = int(filename.split('_')[0])
        except:
            continue
        file_path = os.path.join(folder_path, filename)
        mat = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                row = [int(c) for c in line.strip() if c in '01']
                if len(row) == 32:
                    mat.append(row)
        mat = np.array(mat)
        if mat.shape == (32, 32):
            samples.append(mat)
            true_labels.append(label)
    return np.array(samples), np.array(true_labels)

# ====================== 计算类别均值 ======================
def compute_class_means(X, y):
    mean_mat = np.zeros((10, 32, 32), dtype=np.float32)
    cnt = np.zeros(10, dtype=int)
    for s, l in zip(X, y):
        mean_mat[l] += s
        cnt[l] += 1
    for i in range(10):
        if cnt[i] > 0:
            mean_mat[i] /= cnt[i]
    return mean_mat, cnt

# ====================== 欧氏距离分类 ======================
def predict_euclid(sample, means):
    dists = [np.linalg.norm(sample - means[i]) for i in range(10)]
    return np.argmin(dists)

# ====================== 主程序 ======================
if __name__ == "__main__":

    # 1. 加载数据
    X, y_true = load_all_samples()

    # 2. 计算均值模板
    class_means, count = compute_class_means(X, y_true)

    # 3. 预测
    y_pred = [predict_euclid(s, class_means) for s in X]

    # 4. 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    classes = list(range(10))

    # 5. 计算指标
    acc_list = []
    miss_list = []
    false_alarm_list = []

    total = cm.sum()

    for i in range(10):
        tp = cm[i, i]
        total_i = cm[i, :].sum()
        acc = tp / total_i if total_i != 0 else 0.0

        fn = total_i - tp
        miss_rate = fn / total_i if total_i != 0 else 0.0

        fp = cm[:, i].sum() - tp
        non_i = total - total_i
        fa_rate = fp / non_i if non_i != 0 else 0.0

        acc_list.append(acc)
        miss_list.append(miss_rate)
        false_alarm_list.append(fa_rate)

    # ====================== 绘图：左混淆矩阵 + 右指标表 ======================
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 宋体
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_xlabel('预测标签', fontsize=12)
    ax1.set_ylabel('真实标签', fontsize=12)
    ax1.set_title('混淆矩阵（欧式距离均值匹配）', fontsize=14, pad=10)

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