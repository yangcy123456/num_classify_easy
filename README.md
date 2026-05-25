# README.md
本项目自己简单搭建了一个带有噪声的CNN，以mnist为数据，先通过0，1的数据自学，通过工程化与随机数实现具体的落地。
同时增加表格对比各个模型之间的差异，最终得出CNN的效果优于mnist最好的结论。
注：我将代码中的data删去单独放源码。

.
├── 简单数据 01_(无 data)
│   ├── 欧式距离.py
│   ├── 余弦距离.py
│   ├── 总结.txt          （对01模型的总结）
│   ├── add_noice_seed.py
│   ├── add_noice+knn.py
│   ├── knn.py
│   └── knnplus.py
└── CNN_(无 data)
    ├── out/
    ├── model1.py          （CNN 自己写的）
    ├── model001 (1).py    （增加了健壮性）
    ├── model002 (1).py    （增加了绘图功能）
    ├── model003 (1).py    （找出了库中的问题）（突然报错，因为pytorch版本自己更新了环境不对）
    ├── model004 (1).py    （增加了随机数切分功能）
    └── stickdata.py       （之前的.csv没有放在一起）
