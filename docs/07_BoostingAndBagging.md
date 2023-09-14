---
title: 这是一个机器学习笔记
author: 程俊彦
date: 2022-10-29
---
[TOC]
## 集成方法
### 1、定义与分类
1. “三个臭皮匠顶个诸葛亮”，基学习器应该**好而不同**。
2. 我们考虑二分类问题$y\in \{+1,-1\}$，和真实函数$f$,假设基分类器的错误率为$\epsilon$，对每个基分类器$h_i$：
$$P(h_i(x)\neq f(x))=\epsilon$$
假设集成通过简单投票法结合T个基分类器，若有超过半数的基分类器正确，则集成分类就正确：
$$H(x)=sign\left(\sum\limits_{i=1}\limits^Th_i(x)\right)$$
假设基分类器的错误率**相互独立**，则由Hoeffding不等式可知，继承的错误率为：
$$\begin{aligned}
P(H(x)\neq f(x))&=\sum\limits_{k=0}\limits^{T/2}C_T^k(1-\epsilon)^k\epsilon^{T-k}\\
&\leq exp\left(-\frac{1}{2}T(1-2\epsilon)^2\right)
\end{aligned}$$
上式显示出，随着集成中个体分类器数目T 的增大，集成的错误率将指数级下降,最终趋向于零。
3. 按照基学习器的并行与串行，分为bagging和boosting两种方法，bagging中有代表性的是随机森林，boosting中adaboost和xgboost。
  
### Boosting
Boosting是一族可将弱学习器提升为强学习器的算法.这族算法的工作机制类似：先从初始训练集训练出一个基学习器，再根据基学习器的表现对训练样本分布进行调整，使得先前基学习器做错的训练样本在后续受到更多关注,然后基于调整后的样本分布来训练下一个基学习器；如此重复进行，直至基学习器数目达到事先指定的值T ,最终将这T个基学习器进行加权结合.