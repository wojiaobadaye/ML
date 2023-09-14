---
title: 这是一个机器学习笔记
author: 程俊彦
date: 2022-10-29
---
[TOC]

## 感知机(Perceptron)
### 1、感知机模型
1. $f(x)=sign(w\cdot x+b)$，wx+b是x的特征向量的线性组合，所以是一种线性分类器，感知机用于二分类问题，要求数据集必须是线性可分的

### 2、感知机策略
1. 点到超平面S的距离：$$\frac{1}{\Vert w\Vert}{\lvert w\cdot x_0+b\rvert}$$
2. 由感知机的模型可知：对任意误分类数据，有$$-y_{i}\frac{1}{\Vert w\Vert}{\lvert w\cdot x_0+b\rvert} > 0$$
3. 进一步推出所有误分类点到超平面S的距离总和，作为损失函数：
$$-\frac{1}{\Vert w\Vert}\sum \limits_{x_{i}\in M}y_i(w\cdot x_i+b )$$
4. 最终定义为：
$$L(w,b)=-\frac{1}{\Vert w\Vert}\sum \limits_{x_{i}\in M}y_i(w\cdot x_i+b )$$其中M是误分类点的集合，函数L是经验风险函数，寻找$w,b$使得L取得最小值。

### 3、感知机算法
1. 方法是随机梯度下降法(但我认为OLS也不是不能做)
2. 梯度由下式给出：
$$\nabla_wL(w,b)=-\sum\limits_{x_i\in M}y_ix_i$$
$$\nabla_bL(w,b)=-\sum\limits_{x_i\in M}y_i$$
3. 参数更新方法：
$$w\gets w+\eta y_ix_i$$
$$b\gets w+\eta y_i$$
$\eta$是步长，也叫学习率。

### 4、感知机补充
1. 算法收敛性
 Novikoff定理(很牛，但我看不懂，以后再说)  
2. 对偶形式
- 提前计算出了$x_i$的内积Gram矩阵，简化了计算。
- 当学习率为1的时候，$a_i$表示$x_i$的更新次数，方便观察。
- 当学习率不是1的时候，可以类似的计算出$x_i$的更新次数，找出那些距离S很近，不好分类的点。

### 5、代码实现
#### 5.1 手撕代码
```python
import numpy as np  
#传入的数据要求列是变量，行是观测，假设n个观测，p个变量
class PerceptionMethod(object):  
    # 定义 感知机学习 类
    def __init__(self, X, Y, eta):  
        # 类中参数是 X,Y（X,Y)均为numpy数组,eta,eta是学习率
        if X.shape[0] != Y.shape[0]:  
            # 要求X,Y中的行数一样，(n相同)
            raise ValueError('Error,X and Y must be same when axis=0 ')
        else:  # 在类中储存参数
            self.X = X
            self.Y = Y
            self.eta = eta

    def ini_Per(self):  
        # 感知机的原始形式
        weight = np.zeros(self.X.shape[1])  
        # 初始化weight,b（weight是p维的向量）
        b = 0
        number = 0  
        # 记录训练次数
        mistake = True  
        # mistake是变量用来说明分类是否有错误
        while mistake is True:  # 当有错时
            mistake = False  
            # 开始下一轮纠错前需要将mistake变为true，一来判断这一轮是否有错误
            for index in range(self.X.shape[0]):  
                # index取值0~n-1(一共n个观测)
                if self.Y[index] * (weight @ self.X[index] + b) <= 0:  
                    # 错误判断条件,@表示内积，*表示对应元素相乘
                    weight += self.eta * self.Y[index] * self.X[index]  # 进行更新weight，b
                    b += self.eta * self.Y[index]
                    number += 1
                    print(weight, b)
                    mistake = True  # 此轮检查出错误，表明mistake为true，进行下列一轮
                    break  # 找出第一个错误后调出循环
                #假如说有(12345)其中2，4是误分类点，那么计算顺序是12/1234/12345就是说每更新一次,
                #下一次都是从头开始。
        return weight, b  # 返回值
        
    #对偶形式
    def dual_Per(self):
        Gram = np.dot(self.X, self.X.T) #计算Gram矩阵
        alpha = np.zeros(self.X.shape[0]) #初始化a向量，学习率为1时，a_i代表X_i的更新次数
        b = 0
        mistake = True
        while mistake is True:
            mistake = False
            for index in range(self.X.shape[0]):
                if self.Y[index] * (alpha * self.Y @ Gram[index] + b) <= 0: 
                #y_i*(求和a_j*x_j*y_j+b)
                #@表示对应元素相乘再相加（a,b中有一个是一维数组），*表示对应元素相乘，计算顺序从左往右
                    alpha[index] += self.eta
                    b += self.eta * self.Y[index]
                    print(alpha, b)
                    mistake = True
                    break
        weight = self.Y * alpha @ self.X #w是p维向量，@是矩阵乘法，
        #(我也不清楚np里面的矩阵行向量与列向量的区别，反正乘再加就是@或者np.dot,只乘不加就是*)
        return weight, b
```
#### 5.2 我是调包侠
```python
import sklearn
from sklearn.linear_model import Perceptron
import pandas as pd
import numpy as np

# 数据预处理
iris = load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
data = np.array(df.iloc[:100, [0, 1, -1]])
x, y = data[:, :-1], data[:, -1]
y = np.array([1 if i == 1 else -1 for i in y])

# 模型训练
clf = Perceptron(fit_intercept = True, max_iter = 1000, tol = None, shuffle = True)
clf.fit(x, y)

# 输出参数w, b
print('w = ' + str(clf.coef_) + ', b = ' + str(clf.intercept_))
```
- Perceptron参数表：
  
| 参数名 | 取值(类型,默认) | 说明 |
| :---: | :---: | :---: |
|penalty|none;l1;l2;elasticnet(混合正则)|防止过拟合|
|alpha|0.0001,浮点|penalty乘以这个数|
|fit_intercept|bool,True|是否对参数 截距项b进行估计，若为False则数据应是中心化的|
|max_iter|int,1000|最大迭代次数|
|n_iter_no_change|int,5|在提前停止之前等待验证无改进的迭代次数，用于提前停止迭代|
|early_stopping|bool,False|当验证得分不再提高时是否设置提前停止来终止训练。若设置此项，当验证得分在n_iter_no_change轮内没有提升时提前停止训练|
|tol|float or None,10^(-3)|当loss小于tol时停止迭代|
|eta0|double,1|学习率|
|shuffle|bool,True|每轮训练后是否打乱数据|
|random_state|int or None,None|当shuffle=True，用于打乱数据|
|verbose|int,0|verbose = 0 为不在标准输出流输出日志信息，verbose = 1 为输出进度条记录；verbose = 2 为每个epoch输出一行记录|
|class_weight|取值为dict, {class_label: weight} or “balanced” or None，默认=None|用于拟合参数时，每一类的权重是多少。当为None时，所有类的权重为1，等权重；当为balanced时，某类的权重为该类频数的反比，当为字典时，则key为类的标签，值为对应的权重|
|warm_start|bool,False|若为True则调用前一次设置的参数，使用新设置的参数|

- Perceptron属性表：

| 属性名 | 类型 | 解释 |
| :---: | :---: | :---: |
|classes_|array一维数组,维数是y的类别数量|感知机是array([-1,1])|
|coef_|array二维数组|训练后的模型参数w的数组，不包含截距项b。当为二分类时，该数组shape=(1,p)，p为特征数量。当为多分类时shape=（k, n)|
|intercept_|array一维数组|输出训练后的模型截距b的数组。当为二分类时，该数组shape=(1,)。当为多分类时shape=（k, )|
|lose_functiom_|损失函数的类别|即用的哪种损失函数来定义模型输出值与真实值之间的差异|
|n_iter_|int|模型停止时共迭代的次数|
|t_|int|模型训练时，权重w更新的总次数，等于n_iter_*样本数量|
  
### 参考文献
> 原文链接：https://blog.csdn.net/zero33325/article/details/118733836
> 版权声明：本文为CSDN博主「_dingzhen」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
> 原文链接：https://blog.csdn.net/codedz/article/details/108707540