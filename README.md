# TemporalDIC

基于深度学习的时序数字图像相关项目，使用TemporalDIC网络进行多帧光流估计。

## 安装

1. 安装Python环境和依赖：
```bash
pip install torch torchvision numpy pandas einops tensorboard
```

2. 准备数据集：
```
data/
├── img/           # 图像文件
├── dis/           # 位移文件
└── Annotations.csv # 注释文件
```

## 运行

### 训练
```bash
python train.py --exp_name your_experiment_name
```

### 测试
```bash
python test.py
```
