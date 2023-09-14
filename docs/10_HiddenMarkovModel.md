---
title: 这是一个机器学习笔记
author: 程俊彦
date: 2022-10-29
---
[TOC]
## HMM
### 1、定义及三要素
隐马尔可夫模型是关于时序的概率模型，描述由一个隐藏的马尔可夫链随机生成不可观测的状态随机序列 再由各个状态生成一个观测从而产生观测随机序列的过程。隐藏的马尔可夫链随机生成的状态的序列，称为状态序列( state sequence )Q ;每个状态生成一个观测，而由此产生的观测的随机序列，称为观测序列( observation sequence )V。序列的每个位置又可以看作是一个时刻。

模型的三要素记为$\lambda(A,B,\pi)$,其中A是状态转移矩阵（N*N）表示从某一个状态转移到另外一个状态的概率。B是观测概率矩阵（N*M）表示在第i个状态下，出现第j个观测值的概率，$\pi$表示初始状态分布。

隐马尔可夫有两个假设条件：1）齐次马尔可夫性，2）任意时候的观测只依赖于状态，与其他因素无关，观测的独立性。

模型主要处理三种问题：
1. 概率计算 从$\lambda(A,B,\pi)$出发计算观测序列O出现的概率$P(O|\lambda)$.
2. 学习模型，从观测序列O出发，计算模型的参数$\lambda(A,B,\pi)$，刚好与问题1相反。
3. 预测问题（解码），已知模型$\lambda(A,B,\pi)$和观测序列O，求对给定观测序列条件概率$P(I|\lambda)$最大的状态序列I。即给定观测序列，求最有可能的对应的状态序列。

### 2、概率计算算法(计算$P(O|\lambda)$)
#### 直接计算法
1. 从初始概率出发，计算各个状态的概率分布，使用A矩阵
2. 然后从各个状态的概率分布出发，计算观测概率，使用B矩阵

#### 前向计算法
1. 首先定义前向概率:给定隐马尔可夫模型 ，定义到时刻t部分观测序列为
$o_1,o_2,...0_t$且状态为$q_i$的概率为前向概率，记作:
$$\alpha_i(t)=P(o_1,o_2,...o_t, i_t=q_i|\lambda)$$

2. 算法
   输入：$\lambda$,O
   输出：观测序列的概率$P(O|\lambda)$
   1. 初始化：
   $$\alpha_1(i)=\pi_ib_i(o_1)$$
   表示从1时刻，从第i个初始状态出发得到观测值$o_1$的概率
   2. 递推
   $$\alpha_{(t+1)}(i)=\left[\sum\limits_{j=1}\limits^N\alpha_t(j)\alpha_{ji}\right]b_i(o_{t+1})$$
   表示在t+1时刻得到$o_{t+1}$的概率为，t时刻位于j状态，然后j转移到i状态的概率乘以在i状态得到观测值$o_{t+1}$的概率。
   3. 终止
   $$P(O|\lambda)=\sum\limits_{i=1}\limits^N\alpha_T(i)$$
   表示在T时刻，从所有状态出发，得到最有一个观测值的概率。

#### 后向计算法
1. 首先定义后向概率：给定隐马尔可夫模型 ，定义在时刻t状态为 $qi$ 的条件
下，从 t+1 的部分观测序列为$o_{t+1},o_{t+2},...o_T$的概率为后向概率，记作:
$$\beta_t(i)=P(o_{t+1},o_{t+2},...,o_T|i_t = q_i,\lambda)$$

2. 算法
   输入：$\lambda$,O
   输出：观测序列的概率$P(O|\lambda)$
   1. 初始
   $$\beta_T(i)=1$$
   2. 递推
   $$\beta_t(i)=\sum\limits_{j=1}\limits^Na_{ij}b_j(o_{t+1})\beta_{t+1}(j)$$ 
   3. 终点
   $$P(O|\lambda)=\sum\limits_{i=1}\limits^N\pi_ib_i(o_1)\beta_1(i)$$

### 3、其他概率计算
1. 给定模型$\lambda$和观测O，计算在t时刻处于状态$q_i$的概率，记为：
$$\gamma_t(i)=P(i_t=q_i|O,\lambda)$$
可以通过前向后向概率计算$$\alpha_t(i)\beta_t(i)=P(i_t=q_i,O|\lambda)$$于是推出：
$$\begin{aligned}
\gamma_t(i)&=P(i_t=q_i|O,\lambda)\\
&=\frac{P(i_t=q_i,O|\lambda)}{P(O|\lambda)}\\
&=\frac{\alpha_t(i)\beta_t(i)}{P(O|\lambda)}\\
&=\frac{\alpha_t(i)\beta_t(i)}{\sum\limits_{j=1}\limits^N\alpha_t(i)\beta_t(i)}
\end{aligned}$$

2. 给定模型$\lambda$和观测O，计算在t时刻处于状态$q_i$,在t+1时刻处于状态$q_j$的概率，记为：
$$\xi_t(i,j)=P(i_t=q_i,i_{t+1}=q_j|O,\lambda)$$

$$
\xi_t(i, j)=\frac{P\left(i_t=q_i, i_{t+1}=q_j, O \mid \lambda\right)}{P(O \mid \lambda)}=\frac{P\left(i_t=q_i, i_{t+1}=q_j, O \mid \lambda\right)}{\sum_{i=1}^N \sum_{j=1}^N P\left(i_t=q_i, i_{t+1}=q_j, O \mid \lambda\right)}
$$
而
$$
P\left(i_t=q_i, i_{t+1}=q_j, O \mid \lambda\right)=\alpha_t(i) a_{i j} b_j\left(o_{t+1}\right) \beta_{t+1}(j)
$$

$$
\xi_t(i, j)=\frac{\alpha_t(i) a_{i j} b_j\left(o_{t+1}\right) \beta_{t+1}(j)}{\sum_{i=1}^N \sum_{j=1}^N \alpha_t(i) a_{i j} b_j\left(o_{t+1}\right) \beta_{t+1}(j)}
$$

3. 将 $\gamma_t(i)$ 和 $\xi_t(i, j)$ 对各个时刻 $t$ 求和, 可以得到一些有用的期望值。
(1) 在观测 $O$ 下状态 $i$ 出现的期望值:
$$
\sum_{t=1}^T \gamma_t(i)
$$
(2) 在观测 $O$ 下由状态 $i$ 转移的期望值:
$$
\sum_{t=1}^{T-1} \gamma_t(i)
$$
(2) 在观测 $O$ 下由状态 $i$ 转移到状态$j$的期望值:
$$\sum\limits_{t=1}\limits^{T-1}\xi_t(i,j)$$

### 4、学习算法