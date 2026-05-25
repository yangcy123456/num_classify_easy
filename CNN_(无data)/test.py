# import os
# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch.utils.data import TensorDataset, DataLoader
print("good")
# #超参数
# lr=0.001
# batch_size=64
# epochs=10

# data_train=pd.read_csv('mnist_train.csv')
# data_test=pd.read_csv('mnist_test.csv')

# train_images=data_train.iloc[:,1:].values.astype('float32')/255.0
# train_labels=data_train.iloc[:,0].values.astype('int64')

# test_images=data_test.iloc[:,1:].values.astype('float32')/255.0
# test_labels=data_test.iloc[:,0].values.astype('int64')

# train_images = torch.tensor(train_images)
# train_labels = torch.tensor(train_labels)
# test_images = torch.tensor(test_images) 
# test_labels = torch.tensor(test_labels)

# train_dataset = TensorDataset(train_images, train_labels)
# test_dataset = TensorDataset(test_images, test_labels)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
import pandas as pd

for fname in ['mnist_train.csv', 'mnist_test.csv', 'mnist.csv']:
    try:
        df = pd.read_csv(fname, nrows=0)  # 只读表头
        print(f"{fname}: 列数 = {df.shape[1]}")
    except Exception as e:
        print(f"{fname}: 读取失败 - {e}")

# 再读一行查看内容
for fname in ['mnist_train.csv', 'mnist_test.csv']:
    df = pd.read_csv(fname, nrows=2)
    print(f"\n{fname} 前两行:")
    print(df.head(2))