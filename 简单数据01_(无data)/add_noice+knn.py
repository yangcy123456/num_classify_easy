import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.ndimage import shift

# ====================== 1. 加载原始数据 ======================
def load_digits(folder="data/testDigits"):
    samples = []
    labels = []
    filenames = []
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
            filenames.append(fname)
    return np.array(samples), np.array(labels), filenames

# ====================== 2. 生成噪声：位置偏移 + 随机加1 ======================
def add_noise(img, shift_max=1, flip_prob=0.02):
    """
    shift_max: 最大偏移像素（1=上下左右各动1格）
    flip_prob: 随机翻转像素概率（模拟加1噪声）
    """
    # 1. 随机上下左右偏移
    dx = np.random.randint(-shift_max, shift_max+1)
    dy = np.random.randint(-shift_max, shift_max+1)
    img_noise = shift(img, (dx, dy), cval=0)  # 偏移，空的地方填0
    
    # 2. 随机加1（翻转像素）
    noise = np.random.rand(32,32) < flip_prob
    img_noise[noise] = 1 - img_noise[noise]  # 0变1，1变0
    return img_noise

# ====================== 3. 保存噪声数据到 data_noice/testDigits ======================
def save_noisy_data(samples, labels, filenames, out_folder="data_noice/testDigits"):
    os.makedirs(out_folder, exist_ok=True)
    for i in range(len(samples)):
        fname = filenames[i]
        img = samples[i]
        out_path = os.path.join(out_folder, fname)
        with open(out_path, 'w', encoding='utf-8') as f:
            for row in img:
                f.write(''.join([str(int(x)) for x in row]) + '\n')

# ====================== 4. KNN（k=1 余弦距离）======================
def knn_cosine(sample, data, labels):
    sf = sample.flatten()
    norm_s = np.linalg.norm(sf) + 1e-8
    max_sim = -1
    pred = 0
    for i in range(len(data)):
        df = data[i].flatten()
        norm_d = np.linalg.norm(df) + 1e-8
        sim = np.dot(sf, df) / (norm_s * norm_d)
        if sim > max_sim:
            max_sim = sim
            pred = labels[i]
    return pred

# ====================== 主程序 ======================
if __name__ == "__main__":
    # 1. 加载原始数据
    X, y, fnames = load_digits()
    
    # 2. 生成噪声
    X_noise = [add_noise(img) for img in X]
    
    # 3. 保存到 data_noice/testDigits
    save_noisy_data(X_noise, y, fnames)
    print("✅ 噪声数据集已生成：data_noice/testDigits")

    # 4. KNN在噪声数据上测试
    y_pred = [knn_cosine(s, X_noise, y) for s in X_noise]
    cm = confusion_matrix(y, y_pred)
    digits = list(range(10))

    # 计算指标
    accs, misses, fas = [], [], []
    for i in range(10):
        total = cm[i].sum()
        tp = cm[i,i]
        acc = tp/total if total else 0
        accs.append(f"{acc:.2%}")
        misses.append(f"{(total-tp)/total:.2%}")
        fp = cm[:,i].sum() - tp
        fa = fp/(cm.sum()-total) if (cm.sum()-total)!=0 else 0
        fas.append(f"{fa:.2%}")

    # 绘图
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=digits, yticklabels=digits, ax=ax1)
    ax1.set_xlabel('预测标签')
    ax1.set_ylabel('真实标签')
    ax1.set_title('噪声数据 KNN 混淆矩阵')

    # 指标表
    table_data = [[accs[i], misses[i], fas[i]] for i in range(10)]
    ax2.axis('off')
    table = ax2.table(cellText=table_data, rowLabels=[str(i) for i in digits],
                      colLabels=['准确率','漏检率','虚警率'], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.5,2.5)
    ax2.set_title('识别指标')
    
    plt.tight_layout()
    plt.show()