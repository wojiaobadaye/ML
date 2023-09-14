---
title: 这是一个机器学习笔记
author: 程俊彦
date: 2022-10-29
---
[TOC]

## 八、SVM
### 1、主要概念辨析
- 硬间隔与软间隔
是指是否能够将正负样本完全分开，软间隔允许存在误分类的点。
- 线性与核化
考虑“异或”问题，使用线性SVM不能正确的划分样本，核化是将特征展成更加高维的向量进行运算，也就说，可以构造出非线性的划分超平面。
- 基本型与对偶型
考虑使用拉格朗日乘数法求解划分超平面时的求解顺序，基本型是指$\min\limits_u \max\limits_{\alpha, \beta}$,先求解$\alpha,\beta$再求参数u；对偶型是指$\max\limits_{\alpha, \beta}\min\limits_u$,这样的好处是内层最小化与约束$\alpha, \beta$无关，相当于内层是一个无约束的优化问题，相当于只处理$\max\limits_{\alpha,\beta}$部分，方便很多。同时max,min可以互换的条件是KKT条件，分别是，1）主问题可行；2）对偶问题可行；3）主问题最优，4）互补松弛。
- 总结
SVM的种类实际上就是以上主要概念的排列组合：软/硬间隔 + 线性/核化 + 基本型/对偶型

### 2、硬间隔线性基本型SVM
从最简单的二分类问题入手，我们需要找到一个可以将正负样本完全分开的超平面$w\cdot x+b=0$，x是n维向量，表示样本的各项特征，我们假设$y\in \{+1,-1\}$,是样本的标签。我们就需要找到合适的w和b，$wx+b=0$就是分离超平面。

接下来就是如何找到合适的w和b，假如我们设定正实例点在超平面的上方，$y=+1$表示正类，$y=-1$表示负类，如果分类正确的话$y_i(w\cdot x_i+b)>0$,如果分类错误$y_i(w\cdot x_i+b)<0$，那么只需要找到对所有样本点均满足$y_i(w\cdot x_i+b)>0$的w和b即可。
![](images/2023-02-22-13-51-04.png)

这样的w和b有可能不止一组，我们如何找到最合适的那一个分离超平面呢？答案是我们选择使得距离分类超平面最近的点，到分类超平面的距离最远的那一组w和b。用间隔定义“距离最远”，间隔就是点到超平面的距离,定义为所有样本到超平面距离最小值的两倍：
$$\gamma:=2\min\limits_i\frac{1}{\Vert w\Vert}\vert w^Tx_i+b\vert \in\Reals.$$

就是先找到距离分类超平面最近的点，然后最大化这些点到超平面的距离，这些点就是“支持向量”。我们的目标是：
$$\begin{aligned}
\max\limits_{w,b}\gamma&=\max\limits_{w,b}\min\limits_{i}\frac{2}{\Vert w\Vert}\vert w^Tx_i+b\vert\\
s.t&\quad y_i(w\cdot x_i+b)>0
\end{aligned}$$
对（w,b）的缩放不影响求解最小值点，为了简化问题，我们可以使得$\min\limits_i\vert w^Tx_i+b\vert =1$,于是$\min\limits_iy_i(w^Tx_i+b)=1$,上式变形为:

$$\begin{aligned}
&\quad\argmax\limits_{w,b}\min\limits_{i}\frac{2}{\Vert w\Vert}\vert w^Tx_i+b\vert \\
&=\argmax\limits_{w,b}\frac{2}{\Vert w\Vert}\left(\min\limits_{i}y_i(w^Tx_i+b)\right)\\
&=\argmin\limits_{w,b}\frac{1}{2}\Vert w\Vert\\
&=\argmin\limits_{w,b}\frac{1}{2}\Vert w\Vert^2\\
&=\argmin\limits_{w,b}\frac{1}{2}w^Tw
\end{aligned}$$

最终最优化问题为：
$$\min _{w, b} \frac{1}{2}\|w\|^{2}$$
$$s.t. \quad y_{i}\left(w \cdot x_{i}+b\right)-1 \geqslant 0, \quad i=1,2, \cdots, N$$

### 3、硬间隔线性对偶型SVM

### 4、硬间隔核化SVM

### 5、软间隔核化SVM

### 6、算法
#### 6.1 Pagesos
#### 6.2 坐标下降
#### 6.3 DCD
#### 6.4 SMO
### 7 多分类SVM

### 8、代码
```python
class SVM:
    def __init__(self, max_iter=100, kernel='linear'):
        self.max_iter = max_iter
        self._kernel = kernel

    def init_args(self, features, labels):
        self.m, self.n = features.shape
        self.X = features
        self.Y = labels
        self.b = 0.0

        # 将Ei保存在一个列表里
        self.alpha = np.ones(self.m)
        self.E = [self._E(i) for i in range(self.m)]
        # 松弛变量
        self.C = 1.0

    def _KKT(self, i):
        y_g = self._g(i) * self.Y[i]
        if self.alpha[i] == 0:
            return y_g >= 1
        elif 0 < self.alpha[i] < self.C:
            return y_g == 1
        else:
            return y_g <= 1

    # g(x)预测值，输入xi（X[i]）
    def _g(self, i):
        r = self.b
        for j in range(self.m):
            r += self.alpha[j] * self.Y[j] * self.kernel(self.X[i], self.X[j])
        return r

    # 核函数
    def kernel(self, x1, x2):
        if self._kernel == 'linear':
            return sum([x1[k] * x2[k] for k in range(self.n)])
        elif self._kernel == 'poly':
            return (sum([x1[k] * x2[k] for k in range(self.n)]) + 1)**2

        return 0

    # E（x）为g(x)对输入x的预测值和y的差
    def _E(self, i):
        return self._g(i) - self.Y[i]

    def _init_alpha(self):
        # 外层循环首先遍历所有满足0<a<C的样本点，检验是否满足KKT
        index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
        # 否则遍历整个训练集
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(non_satisfy_list)

        for i in index_list:
            if self._KKT(i):
                continue

            E1 = self.E[i]
            # 如果E2是+，选择最小的；如果E2是负的，选择最大的
            if E1 >= 0:
                j = min(range(self.m), key=lambda x: self.E[x])
            else:
                j = max(range(self.m), key=lambda x: self.E[x])
            return i, j

    def _compare(self, _alpha, L, H):
        if _alpha > H:
            return H
        elif _alpha < L:
            return L
        else:
            return _alpha

    def fit(self, features, labels):
        self.init_args(features, labels)

        for t in range(self.max_iter):
            # train
            i1, i2 = self._init_alpha()

            # 边界
            if self.Y[i1] == self.Y[i2]:
                L = max(0, self.alpha[i1] + self.alpha[i2] - self.C)
                H = min(self.C, self.alpha[i1] + self.alpha[i2])
            else:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])

            E1 = self.E[i1]
            E2 = self.E[i2]
            # eta=K11+K22-2K12
            eta = self.kernel(self.X[i1], self.X[i1]) + self.kernel(
                self.X[i2],
                self.X[i2]) - 2 * self.kernel(self.X[i1], self.X[i2])
            if eta <= 0:
                # print('eta <= 0')
                continue

            alpha2_new_unc = self.alpha[i2] + self.Y[i2] * (
                E1 - E2) / eta  #此处有修改，根据书上应该是E1 - E2，书上130-131页
            alpha2_new = self._compare(alpha2_new_unc, L, H)

            alpha1_new = self.alpha[i1] + self.Y[i1] * self.Y[i2] * (
                self.alpha[i2] - alpha2_new)

            b1_new = -E1 - self.Y[i1] * self.kernel(self.X[i1], self.X[i1]) * (
                alpha1_new - self.alpha[i1]) - self.Y[i2] * self.kernel(
                    self.X[i2],
                    self.X[i1]) * (alpha2_new - self.alpha[i2]) + self.b
            b2_new = -E2 - self.Y[i1] * self.kernel(self.X[i1], self.X[i2]) * (
                alpha1_new - self.alpha[i1]) - self.Y[i2] * self.kernel(
                    self.X[i2],
                    self.X[i2]) * (alpha2_new - self.alpha[i2]) + self.b

            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                # 选择中点
                b_new = (b1_new + b2_new) / 2

            # 更新参数
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new

            self.E[i1] = self._E(i1)
            self.E[i2] = self._E(i2)
        return 'train done!'

    def predict(self, data):
        r = self.b
        for i in range(self.m):
            r += self.alpha[i] * self.Y[i] * self.kernel(data, self.X[i])

        return 1 if r > 0 else -1

    def score(self, X_test, y_test):
        right_count = 0
        for i in range(len(X_test)):
            result = self.predict(X_test[i])
            if result == y_test[i]:
                right_count += 1
        return right_count / len(X_test)

    def _weight(self):
        # linear model
        yx = self.Y.reshape(-1, 1) * self.X
        self.w = np.dot(yx.T, self.alpha)
        return self.w
```