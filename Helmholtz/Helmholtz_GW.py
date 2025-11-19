#!/usr/bin/env python
# coding: utf-8

# 导入必要的库
import os  # 操作系统接口
import time  # 时间相关功能
import numpy as np  # 数值计算库
import sys  # 系统相关参数和函数

sys.path.append("..")  # 添加上级目录到Python路径
from pinn import *  # 导入PINN相关代码
from grad_stats import *  # 导入梯度统计相关代码
import math  # 数学函数
import matplotlib.pyplot as plt  # 绘图库

import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
from torch.autograd import grad  # 自动求导
from torch.optim import Adam  # Adam优化器
from torch.optim.lr_scheduler import StepLR, ExponentialLR, MultiStepLR  # 学习率调度器

# 设置设备：优先使用GPU，如果没有则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实验设置 - 定义空间域边界
lx = ly = -1  # 左边界和下边界
rx = ry = 1  # 右边界和上边界

# 定义亥姆霍兹方程的参数
a_1 = 1  # x方向的波数系数
a_2 = 4  # y方向的波数系数
k = 1  # 波数
lam = k ** 2  # lambda参数，等于k的平方


# 生成精确解（真实解）
def generate_u(x):
    """生成精确解：u(x,y) = sin(πx) * sin(4πy)"""
    return np.sin(a_1 * np.pi * x[:, 0:1]) * np.sin(a_2 * np.pi * x[:, 1:2])


# 计算精确解对x的二阶导数
def u_xx(x, a_1, a_2):
    return - (a_1 * np.pi) ** 2 * np.sin(a_1 * np.pi * x[:, 0:1]) * np.sin(a_2 * np.pi * x[:, 1:2])


# 计算精确解对y的二阶导数
def u_yy(x, a_1, a_2):
    return - (a_2 * np.pi) ** 2 * np.sin(a_1 * np.pi * x[:, 0:1]) * np.sin(a_2 * np.pi * x[:, 1:2])


# 计算强迫项（源项）
def f(x, a_1, a_2, lam):
    """根据精确解计算对应的强迫项f(x,y)"""
    return u_xx(x, a_1, a_2) + u_yy(x, a_1, a_2) + lam * generate_u(x, a_1, a_2)


# 生成测试网格和精确解
x = np.linspace(lx, rx, 1001)[:, None]  # 在x方向生成1001个点
y = np.linspace(ly, ry, 1001)[:, None]  # 在y方向生成1001个点
xx, yy = np.meshgrid(x, y)  # 创建网格
X = np.hstack((xx.flatten()[:, None], yy.flatten()[:, None]))  # 将网格点展平并组合
u_sol = generate_u(X)  # 计算精确解


# 采样函数：生成训练数据
def sampler(num_r, num_b, u_func, lx, rx, ly, ry, delta_N=1001):
    # 生成训练数据
    x = np.linspace(lx, rx, delta_N)  # x方向离散点
    y = np.linspace(ly, ry, delta_N)  # y方向离散点
    xb = np.linspace(lx, rx, num_b)  # 边界上的x点
    yb = np.linspace(ly, ry, num_b)  # 边界上的y点

    # 创建内部点网格
    xx, yy = np.meshgrid(x, y)
    X = np.vstack([xx.ravel(), yy.ravel()]).T  # 所有内部点

    # X方向的边界（左边界和右边界）
    lb = lx * np.ones((yb.shape))  # 左边界x坐标
    rb = rx * np.ones((yb.shape))  # 右边界x坐标
    Xlb = np.vstack((lb, yb)).T  # 左边界点
    Xrb = np.vstack((rb, yb)).T  # 右边界点
    UXlb = u_func(Xlb)  # 左边界真值
    UXrb = u_func(Xrb)  # 右边界真值

    # Y方向的边界（下边界和上边界）
    lb = ly * np.ones((xb.shape))  # 下边界y坐标
    rb = ry * np.ones((xb.shape))  # 上边界y坐标
    Ylb = np.vstack((xb, lb)).T  # 下边界点
    Yrb = np.vstack((xb, rb)).T  # 上边界点
    UYlb = u_func(Ylb)  # 下边界真值
    UYrb = u_func(Yrb)  # 上边界真值

    # 转换为PyTorch张量
    idxs = np.random.choice(xx.size, num_r, replace=False)  # 随机选择残差点
    X_train = torch.tensor(X[idxs], dtype=torch.float32, requires_grad=True, device=device)  # 训练点（需要梯度）
    X_rb = torch.tensor(Xrb, dtype=torch.float32, device=device)  # 右边界点
    X_lb = torch.tensor(Xlb, dtype=torch.float32, device=device)  # 左边界点
    Y_rb = torch.tensor(Yrb, dtype=torch.float32, device=device)  # 上边界点
    Y_lb = torch.tensor(Ylb, dtype=torch.float32, device=device)  # 下边界点

    # 计算训练数据的均值和标准差（用于归一化）
    X_mean = torch.tensor(np.mean(np.concatenate([X[idxs], Xrb, Xlb, Yrb, Ylb], 0), axis=0, keepdims=True),
                          dtype=torch.float32, device=device)
    X_std = torch.tensor(np.std(np.concatenate([X[idxs], Xrb, Xlb, Yrb, Ylb], 0), axis=0, keepdims=True),
                         dtype=torch.float32, device=device)

    # 边界条件的真值
    U_X_rb = torch.tensor(UXrb, dtype=torch.float32, device=device).reshape(num_b, 1)
    U_X_lb = torch.tensor(UXlb, dtype=torch.float32, device=device).reshape(num_b, 1)
    U_Y_rb = torch.tensor(UYrb, dtype=torch.float32, device=device).reshape(num_b, 1)
    U_Y_lb = torch.tensor(UYlb, dtype=torch.float32, device=device).reshape(num_b, 1)

    return X_train, X_lb, X_rb, Y_lb, Y_rb, U_X_lb, U_X_rb, U_Y_lb, U_Y_rb, X_mean, X_std


# 计算PDE残差
def Helmholtz_res(uhat, data):
    """计算亥姆霍兹方程的残差"""
    x = data[:, 0:1]  # x坐标
    y = data[:, 1:2]  # y坐标

    # 计算一阶导数
    du = grad(outputs=uhat, inputs=data, grad_outputs=torch.ones_like(uhat), create_graph=True)[0]
    dudx = du[:, 0:1]  # 对x的一阶导
    dudy = du[:, 1:2]  # 对y的一阶导

    # 计算二阶导数
    dudxx = grad(outputs=dudx, inputs=data, grad_outputs=torch.ones_like(uhat), create_graph=True)[0][:, 0:1]  # 对x的二阶导
    dudyy = grad(outputs=dudy, inputs=data, grad_outputs=torch.ones_like(uhat), create_graph=True)[0][:, 1:2]  # 对y的二阶导

    # 计算源项（强迫项）
    source = - (a_1 * math.pi) ** 2 * torch.sin(a_1 * math.pi * x) * torch.sin(a_2 * math.pi * y) - \
             (a_2 * math.pi) ** 2 * torch.sin(a_1 * math.pi * x) * torch.sin(a_2 * math.pi * y) + \
             lam * torch.sin(a_1 * math.pi * x) * torch.sin(a_2 * math.pi * y)

    # 计算残差：u_xx + u_yy + λu - source
    residual = dudxx + dudyy + lam * uhat - source
    return residual


# 训练参数设置
lr = 1e-3  # 学习率
mm = 10  # 权重更新频率
alpha_ann = 0.5  # 权重平滑参数
Adam_n_epochs = 30000  # 训练轮数
i_print = 100  # 打印频率

# 数据点数量设置
N_r = 20000  # 残差点数量
N_bc = 100  # 每个边界的点数
layer_sizes = [2, 50, 50, 50, 50, 1]  # 神经网络结构 [输入层, 隐藏层..., 输出层]

# 学习率调度设置
guding_lr = False  # 是否使用固定学习率
lr_gamma = 0.1  # 学习率衰减系数

# 设置结果保存路径
if guding_lr:
    path_loc = './results/guding_lr_%s_A-%s' % (lr, Adam_n_epochs)
else:
    path_loc = './results/step_lr_%s_gamma_%s_A-%s' % (lr, lr_gamma, Adam_n_epochs)

print('guding_lr, lr: ', guding_lr, lr)

# 方法列表：DB-PINN的三种变体
method_list = ['DB_PINN_mean', 'DB_PINN_std', 'DB_PINN_kurt']

# 主训练循环
for i in range(3):  # 遍历三种方法
    method = method_list[i]  # 当前方法
    for j in range(1):  # 每种方法运行1次
        print('i,j, method: ', i, j, method)
        save_loc = path_loc + '/method_' + str(method) + '/run_' + str(j)  # 保存路径
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)  # 创建保存目录

        # 采样训练数据
        X_train, X_lb, X_rb, Y_lb, Y_rb, U_X_lb, U_X_rb, U_Y_lb, U_Y_rb, X_mean, X_std = sampler(num_r=N_r, num_b=N_bc,
                                                                                                 u_func=generate_u,
                                                                                                 lx=lx, rx=rx, ly=ly,
                                                                                                 ry=ry)

        # 初始化PINN网络
        net = PINN(sizes=layer_sizes, mean=X_mean, std=X_std, activation=torch.nn.Tanh()).to(device)

        # 初始化损失权重
        lambd_0 = torch.ones(1, device=device)  # 第一组边界权重
        lambds_0 = []  # 记录权重历史
        lambd_1 = torch.ones(1, device=device)  # 第二组边界权重
        lambds_1 = []  # 记录权重历史

        # 记录各种损失
        losses = []  # 总损失
        losses_boundary_0 = []  # 第一组边界损失
        losses_boundary_1 = []  # 第二组边界损失
        losses_residual = []  # 残差损失
        l2_error = []  # L2误差

        N_l = 0  # 权重更新计数器

        # 设置优化器参数
        params = [{'params': net.parameters(), 'lr': lr}]
        milestones = [[10000, 20000, 30000]]  # 学习率调整节点

        # 设置优化器和学习率调度器
        if guding_lr:
            optimizer = Adam(params)  # 固定学习率
        else:
            optimizer = Adam(params)
            scheduler = MultiStepLR(optimizer, milestones[0], gamma=lr_gamma)  # 多步长学习率衰减

        print("训练残差点的形状", X_train.size())
        print("训练边界点的形状 (*4)", X_lb.size())
        start_time = time.time()  # 开始计时

        # 训练循环
        for epoch in range(Adam_n_epochs):
            # 前向传播计算各损失项
            uhat = net(X_train)  # 网络预测
            res = Helmholtz_res(uhat, X_train)  # 计算残差
            l_reg = torch.mean((res) ** 2)  # 残差损失

            # 计算X方向边界损失（左右边界）
            predl = net(X_lb)  # 左边界预测
            predr = net(X_rb)  # 右边界预测
            l_bc_0 = torch.mean((predl - U_X_lb) ** 2)  # 左边界损失
            l_bc_0 += torch.mean((predr - U_X_rb) ** 2)  # 右边界损失

            # 计算Y方向边界损失（上下边界）
            predl = net(Y_lb)  # 下边界预测
            predr = net(Y_rb)  # 上边界预测
            l_bc_1 = torch.mean((predl - U_Y_lb) ** 2)  # 下边界损失
            l_bc_1 += torch.mean((predr - U_Y_rb) ** 2)  # 上边界损失

            # 堆叠所有损失
            L_t = torch.stack((l_reg, l_bc_0, l_bc_1))

            # 权重更新（每mm个epoch更新一次）
            with torch.no_grad():
                if epoch % mm == 0:
                    N_l += 1  # 更新计数器

                    # 计算各损失的梯度统计量
                    stdr, kurtr = loss_grad_stats(l_reg, net)  # 残差梯度统计
                    stdb0, kurtb0 = loss_grad_stats(l_bc_0, net)  # 第一组边界梯度统计
                    stdb1, kurtb1 = loss_grad_stats(l_bc_1, net)  # 第二组边界梯度统计

                    # 计算梯度的最大值和均值
                    maxr, meanr = loss_grad_max_mean(l_reg, net)
                    maxb0, meanb0 = loss_grad_max_mean(l_bc_0, net, lambg=lambd_0)
                    maxb1, meanb1 = loss_grad_max_mean(l_bc_1, net, lambg=lambd_1)

                    # 初始化运行平均值
                    if epoch == 0:
                        lam_avg_bc_0 = torch.zeros(1, device=device)  # 第一组边界权重运行平均
                        lam_avg_bc_1 = torch.zeros(1, device=device)  # 第二组边界权重运行平均
                        running_mean_L = torch.zeros(1, device=device)  # 损失运行平均

                    # DB-PINN均值变体
                    if method == 'DB_PINN_mean':
                        hat_all = maxr / meanb0 + maxr / meanb1  # 总权重预算

                        # 更新损失运行平均
                        mean_param = (1. - 1 / N_l)
                        running_mean_L = mean_param * running_mean_L + (1 - mean_param) * L_t.detach()

                        # 计算难度指数
                        l_t_vector = L_t / running_mean_L

                        # 按难度分配权重
                        hat_bc_0 = hat_all * l_t_vector[1] / torch.sum(l_t_vector[1:])
                        hat_bc_1 = hat_all * l_t_vector[2] / torch.sum(l_t_vector[1:])

                        # Welford算法更新权重
                        lambd_0 = lam_avg_bc_0 + 1 / N_l * (hat_bc_0 - lam_avg_bc_0)
                        lambd_1 = lam_avg_bc_1 + 1 / N_l * (hat_bc_1 - lam_avg_bc_1)
                        lam_avg_bc_0 = lambd_0
                        lam_avg_bc_1 = lambd_1

                    # DB-PINN标准差变体
                    elif method == 'DB_PINN_std':
                        hat_all = stdr / stdb0 + stdr / stdb1  # 总权重预算

                        mean_param = (1. - 1 / N_l)
                        running_mean_L = mean_param * running_mean_L + (1 - mean_param) * L_t.detach()
                        l_t_vector = L_t / running_mean_L  # 难度指数
                        hat_bc_0 = hat_all * l_t_vector[1] / torch.sum(l_t_vector[1:])
                        hat_bc_1 = hat_all * l_t_vector[2] / torch.sum(l_t_vector[1:])

                        lambd_0 = lam_avg_bc_0 + 1 / N_l * (hat_bc_0 - lam_avg_bc_0)
                        lambd_1 = lam_avg_bc_1 + 1 / N_l * (hat_bc_1 - lam_avg_bc_1)
                        lam_avg_bc_0 = lambd_0
                        lam_avg_bc_1 = lambd_1

                    # DB-PINN峰度变体
                    elif method == 'DB_PINN_kurt':
                        # 计算变异系数
                        covr = stdr / kurtr  # 残差变异系数
                        covb0 = stdb0 / kurtb0  # 第一组边界变异系数
                        covb1 = stdb1 / kurtb1  # 第二组边界变异系数
                        hat_all = covr / covb0 + covr / covb1  # 总权重预算

                        mean_param = (1. - 1 / N_l)
                        running_mean_L = mean_param * running_mean_L + (1 - mean_param) * L_t.detach()
                        l_t_vector = L_t / running_mean_L  # 难度指数
                        hat_bc_0 = hat_all * l_t_vector[1] / torch.sum(l_t_vector[1:])
                        hat_bc_1 = hat_all * l_t_vector[2] / torch.sum(l_t_vector[1:])

                        lambd_0 = lam_avg_bc_0 + 1 / N_l * (hat_bc_0 - lam_avg_bc_0)
                        lambd_1 = lam_avg_bc_1 + 1 / N_l * (hat_bc_1 - lam_avg_bc_1)
                        lam_avg_bc_0 = lambd_0
                        lam_avg_bc_1 = lambd_1

            # 计算加权总损失
            loss = l_reg + lambd_0 * l_bc_0 + lambd_1 * l_bc_1

            # 反向传播和优化
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            if guding_lr:
                optimizer.step()  # 更新参数（固定学习率）
            else:
                optimizer.step()  # 更新参数
                scheduler.step()  # 更新学习率

            # 定期打印和记录
            if epoch % i_print == 0:
                # 计算测试误差
                inp = torch.tensor(X, dtype=torch.float32, device=device)
                out = net(inp).cpu().data.numpy().reshape(u_sol.shape)
                tmp = np.linalg.norm(out.reshape(-1) - u_sol.reshape(-1)) / np.linalg.norm(out.reshape(-1))  # 相对L2误差

                # 记录各种指标
                l2_error.append(tmp)
                losses_boundary_0.append(l_bc_0.item())
                losses_boundary_1.append(l_bc_1.item())
                losses_residual.append(l_reg.item())
                lambds_0.append(lambd_0.item())
                lambds_1.append(lambd_1.item())
                losses.append(loss.item())

                # 打印训练信息
                print(
                    "Adam优化:   轮次 {}/{}, 总损失={:.6f}, 边界损失0={:.6f}, 边界损失1={:.6f}, 残差损失={:.6f}, 权重0={:.4f}, 权重1={:.4f}, 学习率={:,.7f}, L2误差 (%)={:.6f}".format(
                        epoch + 1, Adam_n_epochs, loss.item(), l_bc_0.item(), l_bc_1.item(), l_reg.item(),
                        lambd_0.item(), lambd_1.item(), optimizer.param_groups[0]['lr'], tmp * 100))

                # 训练结束
        elapsed_time = time.time() - start_time
        print('Adam训练时间 = ', elapsed_time)

        # 最终预测
        inp = torch.tensor(X, dtype=torch.float32, device=device)
        out = net(inp).cpu().data.numpy().reshape(u_sol.shape)

        # 打印最终结果
        print("\n...Adam训练...\n")
        print("方法: , 运行次数: ", method, j)
        print("预测相对L2误差 = {:e}\n".format(
            np.linalg.norm(out.reshape(-1) - u_sol.reshape(-1)) / np.linalg.norm(u_sol.reshape(-1))))
        print("预测绝对误差 = {:e}\n".format(np.mean(np.abs(out.reshape(-1) - u_sol.reshape(-1)))))
        print("\n.....\n")

        # 准备绘图数据
        U_star = u_sol.reshape(xx.shape)  # 精确解
        U_pred = out.reshape(xx.shape)  # 预测解

        ########### 绘制结果
        # 图1: 精确解、预测解和绝对误差
        fig = plt.figure(1, figsize=(18, 5))
        fig_1 = plt.subplot(1, 3, 1)
        plt.pcolor(xx, yy, U_star, cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title('精确解 $u(x)$')
        fig_2 = plt.subplot(1, 3, 2)
        plt.pcolor(xx, yy, U_pred, cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title('预测解 $u(x)$')
        fig_3 = plt.subplot(1, 3, 3)
        plt.pcolor(xx, yy, np.abs(U_star - U_pred), cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title('绝对误差')
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc, '1.predictions.png'))
        plt.show()
        plt.close()

        # 图2: 损失曲线
        fig_2 = plt.figure(2)
        ax = fig_2.add_subplot(1, 1, 1)
        ax.plot(losses_residual, label='$\mathcal{L}_{r}$')  # 残差损失
        ax.plot(losses_boundary_0, label='$\mathcal{L}_{u_{b0}}$')  # 第一组边界损失
        ax.plot(losses_boundary_1, label='$\mathcal{L}_{u_{b1}}$')  # 第二组边界损失
        ax.set_yscale('log')  # 对数坐标
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('损失')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc, '2.loss.png'))
        plt.show()
        plt.close()

        # 图3: 边界权重变化
        fig_3 = plt.figure(3)
        ax = fig_3.add_subplot(1, 1, 1)
        ax.plot(lambds_0, label='$\lambda_{u_{b0}}$')  # 第一组边界权重
        ax.plot(lambds_1, label='$\lambda_{u_{b1}}$')  # 第二组边界权重
        ax.set_xlabel('迭代次数')
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc, '3.bc_weights.png'))
        plt.show()
        plt.close()

        # 图4: L2误差
        fig_4 = plt.figure(4)
        ax = fig_4.add_subplot(1, 1, 1)
        ax.plot(l2_error)  # L2误差曲线
        ax.set_xlabel('迭代次数')
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc, '4.L2_error.png'))
        plt.show()
        plt.close()