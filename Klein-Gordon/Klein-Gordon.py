#!/usr/bin/env python
# coding: utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import time
import numpy as np
import sys

sys.path.append("..")
from pinn import *
from grad_stats import *
import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import grad
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ExponentialLR, MultiStepLR

# 设置设备：如果有GPU则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实验设置 - 定义空间和时间的边界
lx = 0
lt = 0
rx = 1
rt = 1


# 定义Klein-Gordon方程的解函数
def kg_equation(x, y):  # (x,t)
    return x * np.cos(5 * np.pi * y) + (x * y) ** 3


# 解函数（向量化版本）
def u_func(x):  # x = (x, t)
    return x[:, 0:1] * np.cos(5 * np.pi * x[:, 1:2]) + (x[:, 1:2] * x[:, 0:1]) ** 3


# 计算解函数对时间的二阶导数
def u_tt(x):  # x = (x, t)
    return - 25 * np.pi ** 2 * x[:, 0:1] * np.cos(5 * np.pi * x[:, 1:2]) + 6 * x[:, 1:2] * x[:, 0:1] ** 3


# 计算解函数对空间的二阶导数
def u_xx(x):  # x = (x, t)
    return np.zeros((x.shape[0], 1)) + 6 * x[:, 0:1] * x[:, 1:2] ** 3


# 定义Klein-Gordon方程的右端项（源项）
def f_func(x, alpha=-1.0, beta=0.0, gamma=1.0, k=3.0):
    return u_tt(x) + alpha * u_xx(x) + beta * u_func(x) + gamma * u_func(x) ** k


# 采样函数：生成训练数据
def sampler(num_r, num_b, num_0, lx, rx, lt, rt, delta_N=1001):
    # 生成训练数据
    x = np.linspace(lx, rx, delta_N)  # 空间离散点
    t = np.linspace(lt, rt, delta_N)  # 时间离散点

    # 创建网格
    xx, tt = np.meshgrid(x, t)
    X = np.vstack([xx.ravel(), tt.ravel()]).T

    # 边界点采样
    tb = np.linspace(lt, rt, num_b)
    # X边界（左边界和右边界）
    lb = lx * np.ones((tb.shape))  # 左边界x坐标
    rb = rx * np.ones((tb.shape))  # 右边界x坐标
    Xlb = np.vstack((lb, tb)).T  # 左边界点
    Xrb = np.vstack((rb, tb)).T  # 右边界点
    UXlb = kg_equation(Xlb[:, 0:1], Xlb[:, 1:2])  # 左边界真值
    UXrb = kg_equation(Xrb[:, 0:1], Xrb[:, 1:2])  # 右边界真值

    # 初始条件采样
    xb = np.linspace(lx, rx, num_0)
    # 时间边界（初始时刻）
    tlb = lt * np.ones((xb.shape))  # 初始时刻
    Xic = np.vstack((xb, tlb)).T  # 初始条件点
    Uic = kg_equation(Xic[:, 0:1], Xic[:, 1:2])  # 初始条件真值

    # 转换为训练张量
    idxs = np.random.choice(xx.size, num_r, replace=False)  # 随机选择残差点
    X_train = torch.tensor(X[idxs], dtype=torch.float32, requires_grad=True, device=device)

    X_lb = torch.tensor(Xlb, dtype=torch.float32, device=device)  # 左边界
    X_rb = torch.tensor(Xrb, dtype=torch.float32, device=device)  # 右边界
    X_ic = torch.tensor(Xic, dtype=torch.float32, requires_grad=True, device=device)  # 初始条件

    # 计算训练数据的均值和标准差（用于归一化）
    X_mean = torch.tensor(np.mean(np.concatenate([X[idxs], Xlb, Xrb, Xic], 0), axis=0, keepdims=True),
                          dtype=torch.float32, device=device)
    X_std = torch.tensor(np.std(np.concatenate([X[idxs], Xlb, Xrb, Xic], 0), axis=0, keepdims=True),
                         dtype=torch.float32, device=device)

    # 残差项的真值
    U_Train = torch.tensor(f_func(X[idxs]), dtype=torch.float32, requires_grad=True, device=device)

    # 边界条件和初始条件的真值
    U_X_lb = torch.tensor(UXlb, dtype=torch.float32, device=device).reshape(num_b, 1)
    U_X_rb = torch.tensor(UXrb, dtype=torch.float32, device=device).reshape(num_b, 1)
    U_ic = torch.tensor(Uic, dtype=torch.float32, requires_grad=True, device=device).reshape(num_0, 1)

    return X_train, X_lb, X_rb, X_ic, U_Train, U_X_lb, U_X_rb, U_ic, X_mean, X_std


# 计算PDE残差
def KG_res(uhat, data):
    x = data[:, 0:1]  # 空间坐标
    t = data[:, 1:2]  # 时间坐标

    poly = torch.ones_like(uhat)  # 用于保持形状

    # 计算一阶导数
    du = grad(outputs=uhat, inputs=data,
              grad_outputs=torch.ones_like(uhat), create_graph=True)[0]

    dudx = du[:, 0:1]  # 对x的一阶导
    dudt = du[:, 1:2]  # 对t的一阶导

    # 计算二阶导数
    dudxx = grad(outputs=dudx, inputs=data,
                 grad_outputs=torch.ones_like(uhat), create_graph=True)[0][:, 0:1]  # 对x的二阶导
    dudtt = grad(outputs=dudt, inputs=data,
                 grad_outputs=torch.ones_like(uhat), create_graph=True)[0][:, 1:2]  # 对t的二阶导

    # Klein-Gordon方程残差: u_tt - u_xx + u^3 = 0
    residual = dudtt - dudxx + uhat ** 3

    return residual


# 计算初始条件中对时间的导数
def KG_res_u_t(uhat, data):  # data=(x,t)
    poly = torch.ones_like(uhat)

    # 计算一阶导数
    du = grad(outputs=uhat, inputs=data,
              grad_outputs=torch.ones_like(uhat), create_graph=True)[0]

    dudt = du[:, 1:2]  # 对时间的导数
    return dudt


# 存储所有损失和L2误差
all_losses = []
list_of_l2_Errors = []

# 训练参数设置
lr = 1e-3  # 学习率
mm = 100  # 权重更新频率
i_print = 100  # 打印频率

alpha_ann = 0.5  # 权重平滑参数
n_epochs = 20000  # 训练轮数
num_r = 10000  # 残差点数量
num_b = 1000  # 边界点数量
num_0 = 1000  # 初始点数量
layer_sizes = [2, 50, 50, 50, 1]  # 网络层结构 [输入, 隐藏层..., 输出]

# 学习率调度设置
guding_lr = False
if guding_lr:
    path_loc = './results/guding_lr_%s_rb0_%s_%s_%s_iter_%s_%s' % (lr, num_r, num_b, num_0, n_epochs, layer_sizes)
else:
    path_loc = './results/step_lr_%s_rb0_%s_%s_%s_iter_%s_%s' % (lr, num_r, num_b, num_0, n_epochs, layer_sizes)

print('guding_lr, lr: ', guding_lr, lr)
print('num_r, num_b, num_0: ', num_r, num_b, num_0)
print('layer_sizes: ', layer_sizes)

# 创建保存结果的目录
if not os.path.exists(path_loc):
    os.makedirs(path_loc)

# 生成精确解用于误差计算
x = np.linspace(lx, rx, 1001)
t = np.linspace(lt, rt, 1001)
xx, tt = np.meshgrid(x, t)
u_sol = kg_equation(xx, tt)  # 精确解
X = np.vstack([xx.ravel(), tt.ravel()]).T  # 测试点

# 方法列表：0-等权重, 1-均值法, 2-标准差法, 3-峰度法, DB_PINN的三种变体
method_list = [0, 1, 2, 3, 'DB_PINN_mean', 'DB_PINN_std', 'DB_PINN_kurt']

# 主训练循环
for i in range(7):  # 遍历所有7种方法
    method = method_list[i]
    for j in range(1):  # 每种方法运行1次

        print('i, j, method: ', i, j, method)
        save_loc = path_loc + '/method_' + str(method) + '/run_' + str(j)
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)

        extras = str(num_r) + "+" + str(num_b) + "+" + str(num_0)
        print("#######使用以下配置进行训练#####\n", extras)

        # 采样训练数据
        X_train, X_lb, X_rb, X_ic, U_Train, U_X_lb, U_X_rb, U_ic, X_mean, X_std = sampler(num_r, num_b, num_0, lx, rx,
                                                                                          lt, rt)
        # 初始化PINN网络
        net = PINN(sizes=layer_sizes, mean=X_mean, std=X_std, activation=torch.nn.Tanh()).to(device)

        # 初始化损失权重
        lambd_r = torch.ones(1, device=device)  # 残差权重
        lambd_bc = torch.ones(1, device=device)  # 边界条件权重
        lambd_ic = torch.ones(1, device=device)  # 初始条件权重
        lambd_r_all = [];  # 记录权重历史
        lambd_bc_all = [];
        lambd_ic_all = [];

        # 记录各种损失和误差
        losses = []
        losses_initial = [];
        losses_boundary = [];
        losses_residual = [];
        l2_error = []

        # 训练计数器
        N_l = 0
        # 优化器参数
        params = [{'params': net.parameters(), 'lr': lr}]
        milestones = [[10000, 20000, 30000]]  # 学习率调整节点

        # 设置优化器和学习率调度器
        if guding_lr:
            optimizer = Adam(params)
        else:
            optimizer = Adam(params)
            scheduler = MultiStepLR(optimizer, milestones[0], gamma=0.1)  # 多步长学习率衰减

        print("残差点的形状: ", X_train.size())
        print("边界点的形状 (*2): ", X_lb.size())
        print("初始点的形状: ", X_ic.size())

        start_time = time.time()  # 开始计时

        # 训练循环
        for epoch in range(n_epochs):
            # 前向传播
            uhat = net(X_train)  # 网络预测
            res = KG_res(uhat, X_train)  # 计算残差
            l_reg = torch.mean((res - U_Train) ** 2)  # 残差损失

            # 边界条件预测和损失
            predl = net(X_lb)  # 左边界预测
            predr = net(X_rb)  # 右边界预测
            l_bc = torch.mean((predl - U_X_lb) ** 2)  # 左边界损失
            l_bc += torch.mean((predr - U_X_rb) ** 2)  # 右边界损失

            # 初始条件预测和损失
            pred_ic = net(X_ic)  # 初始条件预测
            l_ic = torch.mean((pred_ic - U_ic) ** 2)  # 初始值损失
            gpreds = KG_res_u_t(pred_ic, X_ic)  # 初始时刻时间导数
            l_ic += torch.mean((gpreds) ** 2)  # 初始导数损失

            # 所有损失的堆叠
            L_t = torch.stack((l_reg, l_bc, l_ic))

            # 权重更新（每mm个epoch更新一次）
            with torch.no_grad():
                if epoch % mm == 0:

                    N_l += 1  # 更新计数器
                    # 计算各损失的梯度统计量
                    stdr, kurtr = loss_grad_stats(l_reg, net)  # 残差梯度统计
                    stdb, kurtb = loss_grad_stats(l_bc, net)  # 边界梯度统计
                    stdi, kurti = loss_grad_stats(l_ic, net)  # 初始梯度统计

                    maxr, meanr = loss_grad_max_mean(l_reg, net)  # 残差梯度最大/均值
                    maxb, meanb = loss_grad_max_mean(l_bc, net, lambg=lambd_bc)  # 边界梯度最大/均值
                    maxi, meani = loss_grad_max_mean(l_ic, net, lambg=lambd_ic)  # 初始梯度最大/均值

                    # 初始化运行平均值
                    if epoch == 0:
                        lam_avg_bc = torch.zeros(1, device=device)  # 边界权重运行平均
                        lam_avg_ic = torch.zeros(1, device=device)  # 初始权重运行平均
                        running_mean_L = torch.zeros(1, device=device)  # 损失运行平均

                    # 方法1: 基于最大值/均值的权重（max/avg）
                    if method == 1:
                        lamb_hat = maxr / meanb  # 边界权重估计
                        lambd_bc = (1 - alpha_ann) * lambd_bc + alpha_ann * lamb_hat  # 平滑更新
                        lamb_hat = maxr / meani  # 初始权重估计
                        lambd_ic = (1 - alpha_ann) * lambd_ic + alpha_ann * lamb_hat

                    # 方法2: 基于标准差的权重（逆狄利克雷）
                    elif method == 2:
                        lamb_hat = stdr / stdb  # 边界权重估计
                        lambd_bc = (1 - alpha_ann) * lambd_bc + alpha_ann * lamb_hat
                        lamb_hat = stdr / stdi  # 初始权重估计
                        lambd_ic = (1 - alpha_ann) * lambd_ic + alpha_ann * lamb_hat

                    # 方法3: 基于峰度的权重
                    elif method == 3:
                        covr = stdr / kurtr  # 残差变异系数
                        covb = stdb / kurtb  # 边界变异系数
                        covi = stdi / kurti  # 初始变异系数
                        lamb_hat = covr / covb  # 边界权重估计
                        lambd_bc = (1 - alpha_ann) * lambd_bc + alpha_ann * lamb_hat
                        lamb_hat = covr / covi  # 初始权重估计
                        lambd_ic = (1 - alpha_ann) * lambd_ic + alpha_ann * lamb_hat

                    # DB-PINN均值变体
                    elif method == 'DB_PINN_mean':
                        hat_all = maxr / meanb + maxr / meani  # 总权重预算
                        mean_param = (1. - 1 / N_l)  # 运行平均参数
                        running_mean_L = mean_param * running_mean_L + (1 - mean_param) * L_t.detach()  # 更新损失运行平均
                        l_t_vector = L_t / running_mean_L  # 难度指数（当前损失/历史平均）
                        hat_bc = hat_all * l_t_vector[1] / (l_t_vector[1] + l_t_vector[2])  # 按难度分配边界权重
                        hat_ic = hat_all * l_t_vector[2] / (l_t_vector[1] + l_t_vector[2])  # 按难度分配初始权重
                        # Welford算法更新权重
                        lambd_bc = lam_avg_bc + 1 / N_l * (hat_bc - lam_avg_bc)
                        lambd_ic = lam_avg_ic + 1 / N_l * (hat_ic - lam_avg_ic)
                        lam_avg_bc = lambd_bc  # 更新运行平均
                        lam_avg_ic = lambd_ic

                    # DB-PINN标准差变体
                    elif method == 'DB_PINN_std':
                        hat_all = stdr / stdb + stdr / stdi  # 总权重预算
                        mean_param = (1. - 1 / N_l)
                        running_mean_L = mean_param * running_mean_L + (1 - mean_param) * L_t.detach()
                        l_t_vector = L_t / running_mean_L  # 难度指数
                        hat_bc = hat_all * l_t_vector[1] / (l_t_vector[1] + l_t_vector[2])
                        hat_ic = hat_all * l_t_vector[2] / (l_t_vector[1] + l_t_vector[2])
                        lambd_bc = lam_avg_bc + 1 / N_l * (hat_bc - lam_avg_bc)
                        lambd_ic = lam_avg_ic + 1 / N_l * (hat_ic - lam_avg_ic)
                        lam_avg_bc = lambd_bc
                        lam_avg_ic = lambd_ic

                    # DB-PINN峰度变体
                    elif method == 'DB_PINN_kurt':
                        covr = stdr / kurtr  # 残差变异系数
                        covb = stdb / kurtb  # 边界变异系数
                        covi = stdi / kurti  # 初始变异系数
                        hat_all = covr / covb + covr / covi  # 总权重预算
                        mean_param = (1. - 1 / N_l)
                        running_mean_L = mean_param * running_mean_L + (1 - mean_param) * L_t.detach()
                        l_t_vector = L_t / running_mean_L  # 难度指数
                        hat_bc = hat_all * l_t_vector[1] / (l_t_vector[1] + l_t_vector[2])
                        hat_ic = hat_all * l_t_vector[2] / (l_t_vector[1] + l_t_vector[2])
                        lambd_bc = lam_avg_bc + 1 / N_l * (hat_bc - lam_avg_bc)
                        lambd_ic = lam_avg_ic + 1 / N_l * (hat_ic - lam_avg_ic)
                        lam_avg_bc = lambd_bc
                        lam_avg_ic = lambd_ic

                    else:
                        # 方法0: 等权重（基准方法）
                        lambd_bc = torch.ones(1, device=device)
                        lambd_ic = torch.ones(1, device=device)

            # 计算加权总损失
            loss = l_reg + lambd_bc.item() * l_bc + lambd_ic.item() * l_ic

            # 定期打印和记录
            if epoch % i_print == 0:
                # 计算测试误差
                inp = torch.tensor(X, dtype=torch.float32, device=device, requires_grad=True)
                out = net(inp).cpu().data.numpy().reshape(u_sol.shape)
                tmp = np.linalg.norm(out.reshape(-1) - u_sol.reshape(-1)) / np.linalg.norm(out.reshape(-1))  # 相对L2误差

                # 记录各种指标
                l2_error.append(tmp)
                list_of_l2_Errors.append(tmp)
                all_losses.append(loss.item())
                losses_initial.append(l_ic.item())
                losses_boundary.append(l_bc.item())
                losses_residual.append(l_reg.item())
                lambd_r_all.append(lambd_r.item())
                lambd_bc_all.append(lambd_bc.item())
                lambd_ic_all.append(lambd_ic.item())

                # 打印训练信息
                print(
                    "轮次 {}/{}, 总损失={:.4f}, 残差损失={:.6f}, 边界损失={:.6f}, 初始损失={:.6f}, 残差权重={:.4f}, 边界权重={:.4f}, 初始权重={:.4f}, 学习率={:.5f}, L2误差(%)={:.3f}".format(
                        epoch + 1, n_epochs, loss.item(), l_reg.item(), l_bc.item(), l_ic.item(), lambd_r.item(),
                        lambd_bc.item(), lambd_ic.item(), optimizer.param_groups[0]['lr'], tmp * 100))

            # 反向传播和优化
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            if guding_lr:
                optimizer.step()  # 更新参数
            else:
                optimizer.step()
                scheduler.step()  # 更新学习率

        # 训练结束，计算总时间
        elapsed_time = time.time() - start_time

        # 最终预测
        inp = torch.tensor(X, dtype=torch.float32, device=device, requires_grad=True)
        out = net(inp).cpu().data.numpy().reshape(u_sol.shape)

        # 打印最终结果
        print("\n.....\n")
        print("方法: , 运行次数: ", method, j)
        print("相对L2误差 = {:e}\n".format(
            np.linalg.norm(out.reshape(-1) - u_sol.reshape(-1)) / np.linalg.norm(u_sol.reshape(-1))))
        print("平均绝对误差 = {:e}\n".format(np.mean(np.abs(out.reshape(-1) - u_sol.reshape(-1)))))
        print("\n.....\n")

        # 保存模型
        torch.save(net.state_dict(), os.path.join(save_loc, 'model.pth'))

        # 准备绘图数据
        U_star = u_sol.reshape(xx.shape)  # 精确解
        U_pred = out.reshape(xx.shape)  # 预测解

        ########### 绘制结果
        # 图1: 精确解、预测解和绝对误差
        fig = plt.figure(1, figsize=(18, 5))
        fig_1 = plt.subplot(1, 3, 1)
        plt.pcolor(xx, tt, U_star, cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$t$')
        plt.title('精确解 $u(x)$')
        fig_2 = plt.subplot(1, 3, 2)
        plt.pcolor(xx, tt, U_pred, cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$t$')
        plt.title('预测解 $u(x)$')
        fig_3 = plt.subplot(1, 3, 3)
        plt.pcolor(xx, tt, np.abs(U_star - U_pred), cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$t$')
        plt.title('绝对误差')
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc, '1.predictions.png'))
        plt.show()
        plt.close()

        # 图2: 损失曲线
        fig_2 = plt.figure(2)
        ax = fig_2.add_subplot(1, 1, 1)
        ax.plot(losses_residual, label='$\mathcal{L}_{r}$')  # 残差损失
        ax.plot(losses_boundary, label='$\mathcal{L}_{bc}$')  # 边界损失
        ax.plot(losses_initial, label='$\mathcal{L}_{ic}$')  # 初始损失
        ax.set_yscale('log')  # 对数坐标
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('损失')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc, '2.loss.png'))
        plt.show()
        plt.close()

        # 图3: 学习到的权重
        fig_3 = plt.figure(3)
        ax = fig_3.add_subplot(1, 1, 1)
        ax.plot(lambd_bc_all, label='$\lambda_{bc}$')  # 边界权重
        ax.plot(lambd_ic_all, label='$\lambda_{ic}$')  # 初始权重
        ax.set_xlabel('迭代次数')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc, '3.learned_weights.png'))
        plt.show()
        plt.close()

        # 图4: L2误差
        fig_4 = plt.figure(4)
        ax = fig_4.add_subplot(1, 1, 1)
        ax.plot(l2_error)  # L2误差
        ax.set_xlabel('迭代次数')
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc, '4.L2_error.png'))
        plt.show()
        plt.close()