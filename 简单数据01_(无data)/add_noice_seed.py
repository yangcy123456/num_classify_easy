import os
import numpy as np
from scipy.ndimage import shift

# ====================== 核心：加噪声（位置偏移 + 随机像素翻转） ======================
def add_noise(img, shift_max=1, flip_prob=0.02):
    # 位置偏移（上下左右 ±shift_max 像素）
    dx = np.random.randint(-shift_max, shift_max + 1)
    dy = np.random.randint(-shift_max, shift_max + 1)
    img_noisy = shift(img, (dx, dy), cval=0)

    # 随机加噪声（0变1 / 1变0）
    noise_mask = np.random.rand(32, 32) < flip_prob
    img_noisy[noise_mask] = 1 - img_noisy[noise_mask]
    return img_noisy

# ====================== 读取原始 32x32 数字 ======================
def load_single_digit(file_path):
    mat = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            row = [int(c) for c in line.strip() if c in '01']
            if len(row) == 32:
                mat.append(row)
    return np.array(mat)

# ====================== 保存噪声数据到 data_noice/seed**** ======================
def save_noisy_dataset(src_folder="data/testDigits", shift_max=1, flip_prob=0.02):
    # 随机种子（4位数字）
    seed = np.random.randint(1000, 9999)
    np.random.seed(seed)
    
    # 输出目录：data_noice/seed****
    save_folder = f"data_noice/seed{seed}"
    os.makedirs(save_folder, exist_ok=True)

    # 遍历所有原始文件
    for fname in os.listdir(src_folder):
        if not fname.endswith(".txt"):
            continue
        
        file_path = os.path.join(src_folder, fname)
        img = load_single_digit(file_path)
        if img.shape != (32, 32):
            continue

        # 加噪声
        img_noisy = add_noise(img, shift_max, flip_prob)

        # 保存
        save_path = os.path.join(save_folder, fname)
        with open(save_path, 'w', encoding='utf-8') as f:
            for row in img_noisy:
                f.write(''.join(map(str, row.astype(int))) + '\n')

    print(f"✅ 噪声数据集已生成：{save_folder}")
    return save_folder

# ====================== 一键运行 ======================
if __name__ == "__main__":
    # 生成：偏移1像素 + 2%噪声
    save_noisy_dataset(shift_max=1, flip_prob=0.02)