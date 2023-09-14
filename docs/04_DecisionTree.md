---
title: 这是一个机器学习笔记
author: 程俊彦
date: 2022-10-29
---
[TOC]


## 决策树
### 1、生成
1. Kraft不等式定理，最优码
2. 信息熵；对样本集S中的随即成员进行最优（最短码长）的编码时所需要的比特位数 $H(p)=-\sum p_ilog_2(p_i)$
3. 经验熵，经验条件熵
   设训练数据集为D，|D|表示其样本容量，即样本个数.设有k个类 $C_k$,k = 
1,2,.. . ,K , $|C_k|$ 为属于类K的样本个数。设特征A有n个不同的取值$(\alpha_1,\alpha_2,\cdots,\alpha_n)$,根据特征A的取值将D划分为n个子集$D_i$为样本个数。记子集 $D_i$ 中属于类 $C_k$ 的样本的集合为 $D_{ik}$.经验熵和经验条件熵：
$$\begin{aligned}
H(D)&=-\sum\limits_{k=1}\limits^K\frac{|C_k|}{|D|}log_2\frac{|C_k|}{|D|}\\
H(D\mid A)&=-\sum\limits_{i=1}\limits^n\frac{|D_i|}{|D|}H(D_i)\\
&=-\sum\limits_{i=1}\limits^n\frac{|D_i|}{|D|}\sum\limits_{k=1}\limits^K\frac{|D_{ik}|}{|D_i|}log_2\frac{|D_{ik}|}{|D_i|}
\end{aligned}$$

4. 信息增益(互信息)：$Gain(D,A)=H(D)-H(D|A)$
5. 信息增益比：$Gain_{Ratio}=\frac{Gain(D,A)}{H_A(D)}$,其中$H_A(D)=-\sum\limits_{i=1}\limits^n\frac{|D_i|}{|D|}log_2\frac{|D_i|}{|D|}$,n是特征A取值的个数。
6. 生成算法：
   - 输入：数据集D，特征集A，阈值$\epsilon$
   - 输出：决策树T
   1. 若D中所有实例属于同一类$C_k$,则T为单节点树，将类$C_k$作为该结点类标记，返回$T_i$。
   2. 若$A=\varnothing$,则T为单节点树，并将D中实例数最大的类$C_k$作为该结点类标记，返回$T_i$。
   3. 否则，按照信息增益或信息增益比算法计算A中各特征对D的信息增益，选择信息增益最大的特征$A_g$.
   4. 如果$A_g$的信心增益比小于阈值$\epsilon$，则设置T为单节点树，并将D中实例数最大的类$C_k$作为该结点类标记，返回$T_i$。
   5. 否则，对$A_g$的每一可能取值$a_i$，依$A_g=a_i$将D分割为若干非空子集$D_i$，将$D_i$中实例数最大的类$C_k$作为标记，构建子结点，由结点及其子结点构成树T，返回$T_i$。
   6. 对第$i$个子结点，以$D_i$为训练集，以$A-A_g$为特征集，递归地调用步骤1~5，得到子树$T_i$，返回$T_i$。
7. ID3,C4.5算法分别使用信息增益和信息增益比进行计算。

### 2、剪枝（预剪枝和后剪枝）