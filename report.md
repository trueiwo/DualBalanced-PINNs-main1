1.  **引言**

**研究问题（problem）**

物理信息神经网络（PINN）已经成为一种新的学习范式，通过将物理方程约束、边界条件约束、初始条件约束加入到损失函数当中进而求解偏微分方程（PDEs）。尽管这是成功的，但PINN由于难以处理多目标优化任务（PDE残差损失与边界和初始条件损失不在一个量级或者差异较大，训练难度不同），其仍然具有准确性差、收敛速度慢的问题。

**解决方案（Innovation）**

为解决上述问题，由香港浸会大学和新加坡科技研究局高性能计算研究院合作共同提出了一种新型双平衡PINN（BP-PINN），通过集成外部平衡和内部平衡来动态调整损失权值，以缓解PINN中的两个不平衡的问题，进而提升PINN的多目标任务优化能力。

创新点

传统的损失函数权值分配方法：

- 手动分配 ：效率低，需要大量人工试错改进
- （学习方法）将权重作为超参数在梯度下降中更新：随参数增加，优化复杂度不断增加，训练困难
- （计算方法）基于梯度统计的加权方法：只考虑PDE残差与每个初始条件与边界条件的关系，而没有考虑初始和边界条件彼此之间的内部关系（梯度不同造成的训练难度差异）

DB-PINN——将PINN中的多优化目标考虑为两个主要目标

- 平衡PDE残差损失和内在条件损失
- 平衡内部具有不同拟合难度的条件损失
- 为了平滑的权重更新和稳定的训练提出一种健壮的权重更新策略

1.  **论文**

**2.1输入数据**

对于求解Klein–Gordon 方程：![alt text](image-62.png)

- 空间坐标x
- 时间坐标t

Wave方程：![alt text](image-61.png)

- 一维空间x
- 时间坐标t

Helmholtz方程：![alt text](image-60.png)

- 二维空间（x , y）

Burgers方程：![alt text](image-59.png)

- 一维空间x
- 时间坐标t

Navier-Stokes方程：（方程形式取决于具体问题）

- 二维不可压缩N-S：二维空间+时间
- 三维不可压缩N-S：三维空间+时间

Schrödinger方程：

- 一维：（x ，t）
- 二维：（x ， y , t）

Heat方程：

- 输入取决于维度——一维、二维、三维+时间

**2.2相关性计算**

（1）PDE残差损失和条件拟合损失之间的相关性：

条件损失总权重：![alt text](image-58.png)

∇<sub>θ</sub>表示损失相对于网络参数θ的梯度向量 L<sup>r</sup>表示PDE残差损失

λiLi表示第i个条件损失

（2）条件损失中不同损失的训练难度系数：![alt text](image-57.png)

Lt表示t时刻的某一观测条件损失，μLt表示截止t时刻前所有该观测损失的平均值

**2.3针对DB-PINNs提出的权重优化策略**

由于梯度下降更新具有随机性，瞬时权重值表现出较大的方差。现有的基于梯度的权重方法通常采用指数移动平均（EMA）策略来更新权重：λⁱ = (1−α)λⁱ + αˆλⁱ，其中 α 是一个超参数。然而，作者观察到该更新策略仍无法有效处理较大方差，导致瞬时权重值频繁出现突然尖峰，甚至发生算术溢出——当使用标准差或峰度作为统计度量时，此现象尤为明显。为解决这一问题，本文提出了一种鲁棒的免调超参数损失权重更新策略，这种策略使用Welford算法，一种用来跟踪观测条件拟合的均值的在线估计方法。使用以下更新规则的平均观测损失向量：![alt text](image-56.png)

权重更新同理：![alt text](image-55.png)

**2.4训练与损失函数**

损失函数由数据拟合损失、边界条件损失和初始条件损失构成：![alt text](image-54.png)

参数θ的更新迭代：![alt text](image-53.png)

 

**2.5论文中的逻辑框架图**
![alt text](image-52.png)
**2.6论文方法流程图**

![alt text](image-51.png)

**3.项目代码**

源码链接 ：https://github.com/chenhong-zhou/DualBalanced-PINNs

**3.1创建环境（安装说明, 数据集准备，依赖说明，运行配置命令行等）**

conda create -n PI python=3.10.19 -y

conda activate PI

nvcc --version #检查cuda版本

conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=11.8 -c pytorch -c nvidia #在官网找到合适的版本指令并下载

（其余所需包自行通过conda install或者pip install自行下载）

**3.2测试结果**

由于原文作者的代码中采用七种不同的损失权重分配方法，每一种都有四张输出图片，共28张图片，如果外加七个公式一共将近两百张图片，故这里着重分析介绍文中提到的Klein-Gordon方程，其中用于对比的损失权重分配方法分别是：0-等权重, 1-均值法, 2-标准差法, 3-峰度法, DB_PINN的三种变体（'DB_PINN_mean', 'DB_PINN_std', 'DB_PINN_kurt'），每种方法的四张输出图像依次为 1.精确解、预测解和绝对误差 2. 损失曲线 3. 学习到的权重 4，L2误差

方法0
![alt text](image-47.png)
![alt text](image-48.png)
![alt text](image-49.png)
![alt text](image-50.png)
方法1
![alt text](image-43.png)
![alt text](image-44.png)
![alt text](image-45.png)
![alt text](image-46.png)
方法2
![alt text](image-39.png)
![alt text](image-40.png)
![alt text](image-41.png)
![alt text](image-42.png)
方法3
![alt text](image-35.png)
![alt text](image-36.png)
![alt text](image-37.png)
![alt text](image-38.png)
方法4
![alt text](image-31.png)
![alt text](image-32.png)
![alt text](image-33.png)
![alt text](image-34.png)
方法5
![alt text](image-27.png)
![alt text](image-28.png)
![alt text](image-29.png)
![alt text](image-30.png)
方法6
![alt text](image-23.png)
![alt text](image-24.png)
![alt text](image-25.png)
![alt text](image-26.png)
由上述运行结果可知（彩色图片是将前两幅图进行对比，看精确解和预测解的差别大小，图片颜色越接近越准确，第三张彩色图则是看二者的绝对误差，颜色越蓝误差越小，越红误差越大），综合来看，DB_PINN_mean（也就是文中理论部分提出的方法）最优

**3.3论文公式对应代码**

注：由于作者通过七个不同的方程进行实验，并在文中只采用三个方程进行实验结果验证（其余为补充实验放在代码里，但总体网络架构基本类似），故以下每个方程的具体代码实现只列举文中多次提到的Klein-Gordon方程作为代表

![alt text](image-12.png)(1)

在DualBalanced-PINNs-main/Klein-Gordon.py文件中的241-304行进行了前置运算，并分别在308、322、338进行求和得到总权重（文中公式只提到了公式一这一种方法，另两个方法的理论描述由实验部分引出）
![alt text](image-13.png)
![alt text](image-14.png)
![alt text](image-15.png)

![alt text](image-11.png)(2)

在DualBalanced-PINNs-main/Klein-Gordon.py文件中的第311、325以及341行对难度指数进行了计算
![alt text](image-16.png)

![alt text](image-10.png)(3)

在DualBalanced-PINNs-main/Klein-Gordon.py文件中的第312、313，326、327以及342、343行体现
![alt text](image-17.png)

![alt text](image-9.png)(4)

在DualBalanced-PINNs-main/Klein-Gordon.py文件中的第310、324、340行分别体现（如上图）

![alt text](image-8.png)(5)

在DualBalanced-PINNs-main/Klein-Gordon.py文件中的第315、316，328、329以及344、345行分别体现（如上图）

![alt text](image-7.png)(6)

Klein-Gordon方程如公式(6)所示，在DualBalanced-PINNs-main/Klein-Gordon.py文件中的第109-129行代码计算得到
![alt text](image-18.png)

 ![alt text](image-6.png)(7)

Wave方程如图公式(7)，在DualBalanced-PINNs-main/Wave.py文件中的第115-124行代码计算得来
![alt text](image-19.png)

 ![alt text](image-5.png)(8)

公式(8)为Wave方程精确的解析解，在DualBalanced-PINNs-main/Wave.py文件中的第30-39行定义
![alt text](image-20.png)

 ![alt text](image-4.png)(9)

Helmholtz方程如图公式(9)所示，在DualBalanced-PINNs-main/Helmholtz.py文件的第110-122行中定义成残差形式
![alt text](image-21.png)

 ![alt text](image-3.png)(10)

方程(10)为源项，为使Helmholtz方程有简单的解析解形式，在DualBalanced-PINNs-main/Helmholtz.py文件的第118-139行体现
![alt text](image-22.png)

 ![alt text](image-2.png)(11)

Helmholtz方程精确解如公式(11)所示，在DualBalanced-PINNs-main/Helmholtz.py文件的第37-39行定义
![alt text](image-1.png)