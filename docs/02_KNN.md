---
title: 这是一个机器学习笔记
author: 程俊彦
date: 2022-10-29
---
[TOC]

## K近邻(K-nearest neighbor)

### 1、基本定义
- 是一种基本的分类方法
- 分类时，对新的实例，根据其K个最近邻的训练实例的类别，通过多数表决等方法进行预测
- 没有显式的学习过程，实际是对特征空间的划分
- 三要素（**都是超参**）：
  - K值
  - 距离度量
  - 分类决策规则

### 2、K近邻算法
- 假设有n个样本，p个变量，k个类别
$$
\left\{\begin{array}{c}
T=\{(x_1,y_1),(x_2,y_2),\cdots ,(x_n,y_n)\} \\
x_i\in \chi \subseteq \mathbb{R}^n \\
y_i\in \gamma =\{c_1,c_2,\cdots ,c_k\}
\end{array}\right.
$$
1. 根据给定的度量距离，在$T$中寻找与$x$ 最邻近的$k$个点，涵盖这$k$个点的领域记为$N_k(x)$
2. 在$N_k(x)$中根据分类决策规则确定$x$的的类别$y$:
$$y=\mathop{\arg\max}\limits_{c_j}\sum\limits_{x_i\in N_k(x)}I(y_i=c_j),\\ i = 1,2,\cdots ,n;j=1,2,\cdots ,k$$
3. $k=1$时是将最邻近的点的类作为$x$的类。

### 2、K近邻模型要素
#### 2.1 定义
- **单元(cell)**：特征空间中，对每个训练实例$x_i$，距离该点比其他点更近的所有点组成的一个区域。
- **划分**：每个实例点拥有一个单元，所有训练实例点的单元构成对特征空间的一个划分

#### 2.2 度量距离的选择
- $Lp$距离：假设$x_i,x_j\in \chi$，$x_i=\{x^{(1)}_i\,x^{(2)}_i\,\cdots ,x^{(n)}_i\}^T$,$x_j=\{x^{(1)}_j\,x^{(2)}_j\,\cdots ,x^{(n)}_j\}^T$,$x_i,x_j$的$Lp$距离定义为：
$$
L_p(x_i,x_j)=\left(\sum\limits^{n}\limits_{l=1}\vert x^{(l)}_i-x^{(l)}_j\vert ^p\right)^{\frac{1}{p}}, p\geqslant 1
$$
- $p=1$时，称为曼哈顿距离：
$$
L_p(x_i,x_j)=\sum\limits^{n}\limits_{l=1}\vert x^{(l)}_i-x^{(l)}_j\vert
$$
- $p=1$时，称为欧氏距离
$$
L_p(x_i,x_j)=\left(\sum\limits^{n}\limits_{l=1}\vert x^{(l)}_i-x^{(l)}_j\vert ^2\right)^{\frac{1}{2}}
$$
- $p=\infty$时，他是各个坐标距离的最大值：
$$L_p(x_i,x_j)=\mathop{\max}\limits_{l}\vert x^{(l)}_i-x^{(l)}_j\vert $$

#### 2.3 $k$值的选择
- $k$选的过小，如$k=1$,意味着用较小邻域中的训练实例进行预测，“学习”的近似误差会比较小，但是“学习”的估计误差会很大，因为会“学习”到噪声，模型变复杂，容易过拟合。
- $k$选的过大，如$k=n$那么不管输入什么$x$，都会预测他为训练实例中最多的类，模型毫无意义。
- 一般$k$取的较小，并且通过交叉验证法来选取最优的$k$值。

#### 2.4 分类决策规则
- 分类时：
  - 多数投票，选择这$k$个样本中出现最多的类别标记作为预测结果。
  - 多数投票法等价于经验风险最小化。(李航P53证明我看不懂)
- 回归时：
  - 平均法，即将这$k$个样本的实值输出标记的平均值作为预测结果。
  - 距离加权法，可基于距离远近进行加权平均或加权投票，距离越近的样本权重越大.

### 3、K近邻算法
- 假设有n个样本，p个变量，k个类别
$$
\left\{\begin{array}{c}
T=\{(x_1,y_1),(x_2,y_2),\cdots ,(x_n,y_n)\} \\
x_i\in \chi \subseteq \mathbb{R}^n \\
y_i\in \gamma =\{c_1,c_2,\cdots ,c_k\}
\end{array}\right.
$$
1. 根据给定的度量距离，在$T$中寻找与$x$ 最邻近的$k$个点，涵盖这$k$个点的领域记为$N_k(x)$
2. 在$N_k(x)$中根据分类决策规则确定$x$的的类别$y$:
$$y=\mathop{\arg\max}\limits_{c_j}\sum\limits_{x_i\in N_k(x)}I(y_i=c_j),\\ i = 1,2,\cdots ,n;j=1,2,\cdots ,k$$
3. $k=1$时是将最邻近的点的类作为$x$的类。

### 4、$kd$树（K-Dimension Tree）
- 为了提高搜索效率，使用特殊结构存储训练数据。
- kd树的算法复杂度是$O(p\cdot ln(n))$,小于穷算$O(p\cdot n)$

#### 4.1 构造平衡$kd$树
1. 构造根节点，使根节点对应于p维空间中包含的所有实例点的超矩形区域。
2. 选择$x^{i}$作为坐标轴，以T中所有实例的$x^{i}$坐标的中位数为切分点，将根节点对应的超矩形区域切分为两个子区域。
3. 由根节点生成深度为1的左右子节点，左侧代表对应坐标小于$x^{i}$切分点的子区域，右边代表对应坐标大于$x^{i}$切分点的子区域。
4. 重复2~3步骤，对于深度为j的节点，选择$x^{l}$作为切分坐标轴，$l=(j\mathop{\mod p})+1$,生成深度为j+1的左右子节点。
5. 所有落在超平面上的点保存在节点中。**（落在超平面上的点是不是中位数？）**
6. 重复直到两个子区域不存在任何实例点。

#### 4.2 搜索平衡$kd$树（最邻近法）
1. 给定一个实例点$x$,首先从根节点出发，按照二分类顺序递归的向下访问$kd$树，直到叶节点。
2. 此叶节点为“当前最近点”。
3. 递归的向上回退，在每个节点进行以下操作：
   1. 如果该节点保存的实例点比当前保存的实例点距离$x$更近，则更新“当前最近点”。
   2. 当前最近点一定存在于该节点一个子节点对应的区域，检查该子节点的父节点的另一子结点对应的区域是否有更近的点。（也就是检查当前最近点距离分割超平面的距离，也就是轴与球心的半径）。
      1. 如果该距离小于$x$到节点的距离，说明需要检查该节点对应的子节点，
      2. 如果该距离大于$x$到节点的距离，说明对应的子节点中不会存在更近的点，继续向上回退。
4. 当退回根节点时，游戏结束。

#### 4.3 其他$kd$树
- 1975年Jon Louis Bentley的论文《Multidimensional Binary Search Trees Used for Associative Searching》中，是循环切割的，而在王永明的《图像局部不变性特征与描述》一书中，是根据方差最大的一维切割的

### 5、代码实现
#### 5.1 我是调包侠


### 参考文献
> https://zhuanlan.zhihu.com/p/25994179
> https://www.joinquant.com/view/community/detail/c2c41c79657cebf8cd871b44ce4f5d97