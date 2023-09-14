---
title: 这是一个机器学习笔记
author: 程俊彦
date: 2022-10-29
---
[TOC]
## 四、朴素贝叶斯（Naive Bayes）

### 1、定义与基本学习方法
#### 1.1 定义
- 是基于贝叶斯定理和特征条件独立性假设的分类方法
- 从训练数据集中学习到联合概率分布$P(X,Y)$,然后根据给定的测试集$X$，选择后验概率最大的$P(Y|X)$
- **朴素**来自于特征条件独立性假设,假设n个观测，p个变量：
$$
\begin{aligned}
P(X=x|Y=c_k)&=P(X^{(1)}=x^{(1)},X^{(2)}=x^{(2)},\cdots ,X^{(n)}=x^{(n)}|Y=c_k)\\&=\prod\limits_{j=1}\limits^p P(X^{(j)}=x^{(j)}|Y=c_k)
\end{aligned}
$$

#### 1.2 基本学习方法
1. 首先学习分类概率$P(Y=c_k)$，即所有训练实例中各个类的概率。
2. 其次学习$P(X=x|Y=c_k)$,在每个类中，找出$X=x$的概率，X是p维特征向量的时候，参数个数就变为$K\cdot \prod\limits_{j=1}\limits^pS_j$,其中$S_j$是$x^{(j)}$的可取值个数。举个例子，假设Y=[A,B]，X=[大小{大/小}，颜色{白/黑}}，标记{有/无}]
那么对于Y=A时，我要求{大小，颜色，标记}的所有组合个数$2^3=8$个联合概率分布，对于Y=B也是一样。
3. 这时引入**朴素**的概念：我**大不大**和我**白不白**和我**有没有标记**无关，这样我的联合概率就等于边缘概率相乘，这样在Y=A时我又大又白的概率就是在Y=A时我大的概率乘以我白的概率，我只需要算出$K\cdot \sum\limits_{j=1}\limits^pS_j$个概率。
4. 总结一下上面的内容，其实我们从训练实例中学习到的是如下的一个概率矩阵（也许该叫列表？）,其中$k=1,2,\cdots ,k ; j=1,2,\cdots ,p ; l=1,2,\cdots ,S_j$

|类别$C_k$|变量$x^{(j)}$|变量$x^{(j)}=a_{lj}$|
|---|---|---|
|c1|x1|$a_{11}$|
|c1|x1|$a_{21}$|
|c1|x2|$a_{12}$|
|c1|x2|$a_{22}$|
|c1|x2|$a_{32}$|
|$\cdots$|$\cdots$|$\cdots$|
|c_k|x_p|$a_{lp}$|
5. 对于新的实例$x=(x^{(1)},x^{(2)},\cdots ,x^{(p)})$,由贝叶斯定理推出:
$$
P\left(Y=c_k \mid X=x\right)=\frac{P\left(X=x \mid Y=c_k\right) P\left(Y=c_k\right)}{\sum_k P\left(X=x \mid Y=c_k\right) P\left(Y=c_k\right)}
$$
将**朴素**条件带入上式：
$$
P\left(Y=c_k \mid X=x\right)=\frac{P\left(Y=c_k\right) \prod_j P\left(X^{(j)}=x^{(j)} \mid Y=c_k\right)}{\sum_k P\left(Y=c_k\right) \prod_j P\left(X^{(j)}=x^{(j)} \mid Y=c_k\right)}, \quad k=1,2, \cdots, K
$$
这是朴素贝叶斯法分类的基本公式。于是, 朴素贝叶斯分类器可表示为
$$
f(x)=\arg \max _{c_k} \frac{P\left(Y=c_k\right) \prod_j P\left(X^{(j)}=x^{(j)} \mid Y=c_k\right)}{\sum_k P\left(Y=c_k\right) \prod_j P\left(X^{(j)}=x^{(j)} \mid Y=c_k\right)}
$$
上式中分母对所有 $c_k$ 都是相同的, 所以,
$$
y=\arg \max _{c_k} P\left(Y=c_k\right) \prod_j P\left(X^{(j)}=x^{(j)} \mid Y=c_k\right)
$$
也就是将（4）中概率矩阵对应的$a_{lp}$连乘积与$P(Y=C_k)$相乘，寻找最大的乘积对应的类$C_k$。

### 2、朴素贝叶斯策略
- 取后验概率最大化，等价于0-1损失函数时的期望风险最小化（证明见P61）

### 3、朴素贝叶斯算法（参数估计）
#### 3.1 极大似然估计
- 用极大似然估计计算先验概率及条件概率：
$$
\begin{gathered}
P(Y=c_k)=\frac{\sum\limits_{i=1}\limits^{N}I(y_i=c_k)}{N}， \\ 
P\left(X^{(j)}=a_{j l} \mid Y=c_k\right)=\frac{\sum_{i=1}^N I\left(x_i^{(j)}=a_{j l}, y_i=c_k\right)}{\sum_{i=1}^N I\left(y_i=c_k\right)}， \\
j=1,2, \cdots, n ; \quad l=1,2, \cdots, S_j ; \quad k=1,2, \cdots, K
\end{gathered}
$$
- 第一个公式是先验概率。
- 第二个公式是条件概率，其中$a_{jl}$表示第j个特征取值为$l$，解释为当Y取值为$c_k$类时，x的第j个特征向量取值为$a_{jl}$的概率，计算方式为：所有满足条件的x的个数除以$c_k$类中x的总个数。

#### 3.2 分类算法
1. 对于给定的实例 $x=\left(x^{(1)}, x^{(2)}, \cdots, x^{(n)}\right)^{\mathrm{T}}$, 计算
$$
P\left(Y=c_k\right) \prod_{j=1}^n P\left(X^{(j)}=x^{(j)} \mid Y=c_k\right), \quad k=1,2, \cdots, K
$$

2. 确定实例 $x$ 的类
$$
y=\arg \max _{c_k} P\left(Y=c_k\right) \prod_{j=1}^n P\left(X^{(j)}=x^{(j)} \mid Y=c_k\right)
$$
- 从1.中计算出的K个值中选择最大的值，以最大的值所归属的类作为实例的类输出。

#### 3.3 贝叶斯估计
贝叶斯估计适用于极大似然估计中可能会出现的概率值为0的情况：
$$
P_{\lambda}\left(X^{(j)}=a_{j l} \mid Y=c_k\right)=\frac{\sum_{i=1}^N I\left(x_i^{(j)}=a_{j l}, y_i=c_k\right)+\lambda}{\sum_{i=1}^N I\left(y_i=c_k\right)+S_j\lambda}，
$$
等价于在随机变量的各个取值的频数上赋予一个正数$\lambda >0$,$\lambda =1$时称之为拉普拉斯平滑，上式满足：
$$\begin{aligned}
P_{\lambda}(X^{(j)}=a_{jl}\mid Y=c_k)&>0\\ \sum\limits_{l=1}\limits^{S_j}P(X^{(j)}=a_{jl}\mid Y=c_k)&=1
\end{aligned}$$
表明$P_{\lambda}$确实是一种概率分布，同样先验概率的贝叶斯估计是：
$$P_{\lambda}=\frac{\sum\limits_{i=1}\limits^{N}I(y_i=c_k)+\lambda }{N+K\lambda}$$

### 4、代码实现
#### 4.1 手撕代码
#### 4.2 我是调包侠

### 参考文献
> 版权声明：本文为CSDN博主「开贝塔的舒克」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/weixin_42466690/article/details/88765378
