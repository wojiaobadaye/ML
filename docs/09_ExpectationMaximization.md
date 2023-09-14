---
title: 这是一个机器学习笔记
author: 程俊彦
date: 2022-10-29
---
[TOC]
## EM算法
### 1、定义与例子
EM 算法是一种选代算法， 1977 年由 Dempster 等人总结提出，用于含有隐变量（hidden variable） 的概率模型参数的极大似然估计，或极大后验概率估计。 EM 算法的每次选代由两步组成 :E 步，求期望 expectation）; 步，求极大 maximization）所以这一算法称为期望极大算法 Cexpectation maximization algorithm），简称 EM法。本章首先叙述 EM 算法，然后讨论 EM 算法的收敛性:作为 EM 算法的应用，介绍高斯混合模型的学习:最后叙述 EM 算法的推广----GEM 算法。

EM 算法就是含有隐变量的概率模型参数的极大似然估计法，或极大后验概率估计法。我们仅讨论极大似然估计，极大后验概率估计与其类似。

介绍三硬币模型，有三枚硬币ABC，正面朝上的概率分别为$\pi,p,q$。进行如下掷硬币试验 先掷硬币 ，根据其结果选出硬币B或硬币C ，正面选硬币B ，反面边硬币 C; 然后掷选出的硬币，掷硬币的结果，出现正面记作1，出现反面记作 0; 独立地重复 次试验(这里， η= 10) ，观测结果如下:
$$1, 1,0, 1,0, 0, 1,0, 1, 1$$
假设只能观测到掷硬币的结果，不能观测掷硬币的过程。问如何估计三硬币正面出现的概率，即三硬币模型的参数。

三硬币模型可以写作：
$$
\begin{aligned}
P(y \mid \theta) & =\sum_z P(y, z \mid \theta)=\sum_z P(z \mid \theta) P(y \mid z, \theta) \\
& =\pi p^y(1-p)^{1-y}+(1-\pi) q^y(1-q)^{1-y}
\end{aligned}
$$
这里, 随机变量 $y$ 是观测变量, 表示一次试验观测的结果是 1 或 0 ; 随机变量 $z$ 是隐变量, 表示末观测到的掷硬币 $\mathrm{A}$ 的结果; $\theta=(\pi, p, q)$ 是模型参数。这一模型是以上数据的生成模型。注意, 随机变量 $y$ 的数据可以观测, 随机变量 $z$ 的数据不可观测。将观测数据表示为 $Y=\left(Y_1, Y_2, \cdots, Y_n\right)^{\mathrm{T}}$, 末观测数据表示为 $Z=\left(Z_1, Z_2, \cdots, Z_n\right)^{\mathrm{T}}$ 则观测数据的似然函数为
$$
P(Y \mid \theta)=\sum_Z P(Z \mid \theta) P(Y \mid Z, \theta)
$$
即
$$
P(Y \mid \theta)=\prod_{j=1}^n\left[\pi p^{y_j}(1-p)^{1-y_j}+(1-\pi) q^{y_j}(1-q)^{1-y_j}\right]
$$
考虑求模型参数 $\theta=(\pi, p, q)$ 的极大似然估计, 即
$$
\hat{\theta}=\arg \max _\theta \log P(Y \mid \theta)
$$
这个问题没有解析解, 只有通过迭代的方法求解。EM算法就是可以用于求解这 个问题的一种迭代算法。

### 2、算法
输入：观测数据Y，隐变量数据Z，联合分布$P(Y，Z|\theta)$,条件分布$P(Z|Y,\theta)$
输出：模型参数$\theta$
1. 选择参数的初值$\theta^{(0)}$,开始迭代
2. E步：计算
$$\begin{aligned}
Q(\theta,\theta^{(i)})&=E_Z[logP(Y,Z|\theta)Y,\theta^{(i)}]\\
&=\sum\limits_ZlogP(Y,Z|\theta)P(Z|Y,\theta^{(i)})
\end{aligned}$$
3. M步：求使得Q极大化的$\theta$，确定第i+1次迭代的参数估计值$\theta^{i+1}$
4. 重复2-3步，直到收敛