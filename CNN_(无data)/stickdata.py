import pandas as pd

train = pd.read_csv('mnist_train.csv', header=None)
test = pd.read_csv('mnist_test.csv', header=None)

print("训练集形状:", train.shape)  # 应为 (60000, 785)
print("测试集形状:", test.shape)  # 应为 (10000, 785)

mnist = pd.concat([train, test], ignore_index=True)
mnist.to_csv('mnist.csv', index=False, header=False)
print(f"合并完成，总行数：{len(mnist)}，列数：{mnist.shape[1]}")