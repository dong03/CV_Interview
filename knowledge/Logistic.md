
逻辑回归假设**数据服从伯努利分布(因为是二分类)**,通过**极大化似然函数**的方法，运用**梯度下降**来求解参数，来达到将**数据二分类**的目的。

## Q1: 损失函数推导

https://zhuanlan.zhihu.com/p/439298352

## Q2: LR与线性回归的区别与联系

* LR是分类，线性回归是回归。

* 实现形式上：逻辑回归和线性回归首先都是广义的线性回归，在本质上没多大区别，区别在于逻辑回归多了个Sigmoid函数，使样本映射到[0,1]之间的数值，从而来处理分类问题。

* 数据假设上:另外逻辑回归是假设变量服从伯努利分布，线性回归假设变量服从高斯分布。逻辑回归输出的是离散型变量，用于分类，线性回归输出的是连续性的，用于预测。
  
* 求解上：逻辑回归是用最大似然法去计算预测函数中的最优参数值，而线性回归是用最小二乘法去对自变量量关系进行拟合。


## Q3:可以进行多分类吗？

* 采用一对多的方法扩展到多分类：
    每次将一个类型作为正例，其他的作为反例，训练 N个分类器。但容易造成训练集样本数量的不平衡（Unbalance），尤其在类别较多的情况下，经常容易出现正类样本的数量远远不及负类样本的数量，这样就会造成分类器的偏向性。

*  一对一的方法：
    将 N 类别两两配对，预测的时候，将样本通过所有分类器，通过投票来得到结果，最后票数最多的类别被认定为该样本的类


参考：
https://blog.csdn.net/Anthony_hit/article/details/123051027
https://zhuanlan.zhihu.com/p/439298352