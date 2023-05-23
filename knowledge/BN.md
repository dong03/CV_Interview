## 1. 标准化与归一化的异同？

归一化：min max scale

标准化：使数据服从正态分布N(0,1)

同：

1. 避免数值引起问题(梯度爆炸、消失)

2. 加速算法收敛，提高算法收敛性能

异：

如果数据存在异常值和较多噪音，用标准化，可以间接通过中心化避免异常值和极端值的影响。

## 2.什么是批量归一化BN？



![](https://secure2.wostatic.cn/static/79pCxWX9GftZU7h1dQL9bS/image.png)

## 3.为什么使用BN？

1.网络的每一层要不断的适应输入数据的分布，这会影响网络的性能。

2.网络的深度可能导致数值的爆炸与消失

3.不同batch之间的数据存在一定的分布不同，我们称为内部协变量飘移【“Internal Covariate Shift”】

4.加速网络的收敛

## 4.γ与β两个参数的引入有什么效果？

1. 相比于普通的标准化为每层的神经元自适应的学习一个量身定制的分布，有效保留每个神经元的学习效果。
2. 普通的标准化会把大部分值映射到激活函数近似线性的区域，而γ与β可以使得BN后的值进入激活函数的非线性区域。

## 5.具体实现与参数量

![](https://secure2.wostatic.cn/static/8rB2U8gv6xzB2347XnmU8V/image.png)

均值的计算，就是在一个批次内，将每个通道中的数字单独加起来，再除以HWN。

方差也同理，所以均值方差的channl-wise，大小等于通道数。

![](https://secure2.wostatic.cn/static/wLYwrSKkK5TM2tB2k8xTfu/image.png)

![](https://secure2.wostatic.cn/static/e3eics1EC1WimPpBWp1iPB/image.png)

可训练参数gamma与beta的维度也等于通道数。

训练与推理时BN中的均值、方差分别是什么？

训练时，均值、方差分别是该批次内数据相应维度的均值与方差。

推理时，均值、方差是基于训练时所有批次的期望方差，实体实现是通过批次的均值与方差移动平均所得。

## 6.其他的Normalization是效果与针对的问题是什么？

BN缺点是需要较大的 batchsize 才能合理估训练数据的均值和方差，同时它也很难应用在训练数据长度不同的 RNN 模型上。Layer Normalization (LN) 的一个优势是不需要批训练，在单条数据内部就能归一化。

![](https://secure2.wostatic.cn/static/7zZCxVCbdeg8gK1PfGp7PN/image.png)

![](https://secure2.wostatic.cn/static/oQL6wWm6V5EMApwfZJjkaw/image.png)

Instance Normalization (IN) 最初用于图像的风格迁移。削弱不同channl之间的影响。

![](https://secure2.wostatic.cn/static/8eouVXQ2ZeSZz866hxpeZn/image.png)

![](https://secure2.wostatic.cn/static/5xrWRbk2cN1DMPdF2kiNsU/image.png)



## 7.BN层的Pytorch代码

```Python
def batch_norm(X,gamma,beta,moving_mean,moving_var,eps,momentum):
  # 推理
  if not torch.is_grad_enabled():
    X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
  # 训练
  else:
    mean = X.mean(dim=(0,2,3),keepdim=True)
    var = ((X - mean)**2).mean(dim=(0,2,3),keepdim=True)
    X_hat = (X - mean) / torch.sqrt(var + eps)
    
    # 滑动平均记录训练过程中BN的均值和方差
    moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
    moving_var = momentum * moving_var + (1.0 - momentum) * var
    
  Y = gamma * X_hat + beta
  return Y, moving_mean.data, moving_var.data
  
class BatchNorm(nn.module):
  def __init__(self,channl):
    super().__init__()
    
    shape = (1,channl,1,1)
    # 参数初始化
    self.gamma = nn.Parameter(torch.ones(shape))
    self.beta = nn.Parameter(torch.ones(shape))
    self.moving_mean = torch.zeros(shape)
    self.moving_var = torch.zeros(shape)

  def forward(self,X):
    Y, self.gamma, self.beta, self.moving_mean, self.moving_var = batch_norm(X,self.gamma,self.beta,self.moving_mean,self.moving_var,eps=1e-5,momentum=0.9)

    return Y

```



参考：

[https://zhuanlan.zhihu.com/p/86765356](https://zhuanlan.zhihu.com/p/86765356)

[https://zhuanlan.zhihu.com/p/93643523](https://zhuanlan.zhihu.com/p/93643523)

[https://blog.csdn.net/junqing_wu/article/details/105431919?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_title~default-0.no_search_link&spm=1001.2101.3001.4242.1](https://blog.csdn.net/junqing_wu/article/details/105431919?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_title~default-0.no_search_link&spm=1001.2101.3001.4242.1)

[https://blog.csdn.net/liuxiao214/article/details/81037416](https://blog.csdn.net/liuxiao214/article/details/81037416)

