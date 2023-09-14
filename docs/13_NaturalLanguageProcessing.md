## NLP

### 基础概念
词是文本中可分析的最小单位，词的表示可分为：
1. 分布式表示
2. 词嵌入表示（词向量）

自然语言处理有如下任务：
1. 语言模型
2. 基础任务（分词，词性，词义，句子意思，语义分析）
3. 应用任务（信息抽取，情感分析）
- 以上都是文本分类问题，还有结构预测和序列到序列问题

### 潜在语义分析（Latent semantic analysis）
潜在语义分析Clatent semantic analysis, L8A) 种无监督学习方法，主要用于文本的话题分析，其特点是通过矩阵分解发现文本与单词之间的基于话题的语义关系。

潜在语义分析使用的是非概率的话题分析模型。具体地，将文本集合表示为单
词-文本矩阵，对单词.文本矩阵进行奇异值分解，从而得到话题向量空间，以及文本在话题向量空间的表示。奇异值分解 Csingular value decomposition, SVD) 即在第 15章介绍的矩阵因子分解方法，其特点是分解的矩阵正交。非负矩阵分解 Cnon-negative matrix factorization , NMF) 是另 种矩阵的因子分解方法，其特点是分解的矩阵非负。 1999 Lee 8heung 的论文 [3J 发表之后，非负矩阵分解引起高度重视和广泛使用。非负矩阵分解也可以用于话题分析。

#### 单词向量空间
假设有n个文本$D={d_1,d_2,...,d_n}$,所有文本中的单词一共有m个，构成m维单词向量$W={w_1,w_2,...,w_m}$，可以组合出W-D矩阵（单词-文本矩阵）mxn维，用来表示文本,$x_{ij}$表示单词i在文档j中出现的频数或者权值。

定义TF-IDF（词频-逆文本概率）权值为：
$$\frac{TF_{i j}}{TF_{\cdot j}}\mathbb{log}\frac{df}{df_i}$$
其中$TF_{i,j}$表示第j篇文本中单词$w_i$出现的频数,$TF_{\cdot j}$表示第j篇文章中所有词出现的频数，df是指所有的文本总数n，$df_i$表示文本中出现过单词$w_i$的文章篇数。前一部分用来判断一篇文章中某一单词的出现频率，某一部分用来判断某一次是不是停用词（“的”之类不重要的词）。

#### 话题向量空间
假设文本还有k个话题，每个话题由一个定义在单词集合W上的m维向量表示。也就是对w加权，使其变成一个话题向量，则话题空间T是单词向量X的一个子空间。表示成W-T矩阵，mxk维。

用矩阵Y表示话题在文本中出现的情况，kxn维T-D矩阵（话题-文本矩阵），有n个文本，每个文本k个话题组成的话题向量。$Y={y_1,y_2,,...,y_n}$.

我们的目标就是把单词文本矩阵转化为话题文本矩阵。$$X\approx TY$$

#### 潜在语义分析算法
1. 矩阵奇异值分解算法

根据确定的话题个数，对单词文本矩阵X进行截断奇异值分解：
$$
X \approx U_k \Sigma_k V_k^{\mathrm{T}}=\left[\begin{array}{llll}
u_1 & u_2 & \cdots & u_k
\end{array}\right]\left[\begin{array}{cccc}
\sigma_1 & 0 & 0 & 0 \\
0 & \sigma_2 & 0 & 0 \\
0 & 0 & \ddots & 0 \\
0 & 0 & 0 & \sigma_k
\end{array}\right]\left[\begin{array}{c}
v_1^{\mathrm{T}} \\
v_2^{\mathrm{T}} \\
\vdots \\
v_k^{\mathrm{T}}
\end{array}\right]
$$
式中 $k \leqslant n \leqslant m, U_k$ 是 $m \times k$ 矩阵, 它的列由 $X$ 的前 $k$ 个互相正交的左奇异向量组 成, $\Sigma_k$ 是 $k$ 阶对角方阵, 对角元素为前 $k$ 个最大奇异值, $V_k$ 是 $n \times k$ 矩阵, 它的列 由 $X$ 的前 $k$ 个互相正交的右奇异向量组成。

在单词-文本矩阵 $X$ 的截断奇异值分解式 (17.13) 中, 矩阵 $U_k$ 的每一个列向量 $u_1, u_2, \cdots, u_k$ 表示一个话题, 称为话题向量。由这 $k$ 个话题向量张成一个子空间
$$
U_k=\left[\begin{array}{llll}
u_1 & u_2 & \cdots & u_k
\end{array}\right]
$$
称为话题向量空间。

有了话题向量空间, 接着考虑文本在话题空间的表示。将式 (17.13) 写作
$$
\begin{aligned}
X & =\left[\begin{array}{llll}
x_1 & x_2 & \cdots & x_n
\end{array}\right] \approx U_k \Sigma_k V_k^{\mathrm{T}} \\
& =\left[\begin{array}{llll}
u_1 & u_2 & \cdots & u_k
\end{array}\right]\left[\begin{array}{cccc}
\sigma_1& & & \\
&\sigma_2 & & 0 \\
& &\ddots & \\
&0 & &\sigma_k
\end{array}\right]\left[\begin{array}{cccc}
v_{11} & v_{21} & \cdots & v_{n 1} \\
v_{12} & v_{22} & \cdots & v_{n 2} \\
\vdots & \vdots & & \vdots \\
v_{1 k} & v_{2 k} & \cdots & v_{n k}
\end{array}\right] \\
& =\left[\begin{array}{llll}
u_1 & u_2 & \cdots & u_k
\end{array}\right]\left[\begin{array}{cccc}
\sigma_1 v_{11} & \sigma_1 v_{21} & \cdots & \sigma_1 v_{n 1} \\
\sigma_2 v_{12} & \sigma_2 v_{22} & \cdots & \sigma_2 v_{n 2} \\
\vdots & \vdots & & \vdots \\
\sigma_k v_{1 k} & \sigma_k v_{2 k} & \cdots & \sigma_k v_{n k}
\end{array}\right]
\end{aligned}
$$

2. 非负矩阵分解
损失函数：平方损失或者散度

#### 概率潜在语义分析