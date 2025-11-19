import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
import math


# sin activation
class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


# 定义神经网络层
class Layer(nn.Module):
    def __init__(self, in_features, out_features, activation):
        super(Layer, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)  # 创建线性层
        # 根据激活函数类型设置Xavier初始化的增益值
        gain = 5 / 3 if isinstance(activation, nn.Tanh) else 1
        nn.init.xavier_normal_(self.linear.weight, gain=gain)  # 使用Xavier正态分布初始化权重
        nn.init.zeros_(self.linear.bias)  # 将偏置初始化为0
        self.linear = self.linear

    def forward(self, x):
        return self.linear(x)  # 执行线性变换


# 定义物理信息神经网络(PINN)
class PINN(nn.Module):
    def __init__(self, sizes, mean=None, std=None, activation=nn.Tanh()):
        super(PINN, self).__init__()

        self.mean = mean  # 输入数据的均值，用于标准化
        self.std = std  # 输入数据的标准差，用于标准化

        # 根据字符串选择激活函数
        if activation == 'sin':
            activation = Sine()
        elif activation == 'relu':
            activation = torch.nn.ReLU()

        # 构建神经网络层
        layer = []
        # 构建隐藏层
        for i in range(len(sizes) - 2):
            linear = Layer(sizes[i], sizes[i + 1], activation)  # 创建线性层
            layer += [linear, activation]  # 将线性层和激活函数添加到层列表中
        # 添加输出层（无激活函数）
        layer += [Layer(sizes[-2], sizes[-1], activation)]
        # 将层序列组合成神经网络
        self.net = nn.Sequential(*layer)

    # 前向传播函数
    def forward(self, x):
        # 如果提供了均值和标准差，则对输入进行标准化
        if self.mean is None:
            X = x  # 不使用标准化
        else:
            X = (x - self.mean) / self.std  # 标准化处理

        return self.net(X)  # 通过神经网络前向传播

