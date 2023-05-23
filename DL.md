# CV/多模态


## 参数初始化
- 全0，等同线性模型，对每个参数导数都相同为0，很快梯度消失（梯度弥散）
    - wx+b, b的梯度不为0， 仅依靠b拟合数据

- 正态随机
    - 通常 x 0.1,避免sigmoid截顶
    - loss可以正常下降，但正态分布以0为中心，层数增加时输出接近中间值，梯度消失
- Xavier随机
    - 输入和输出的分布一致
    - 对称，随着深度增加会收缩
    - 适用sigmoid，tanh等对称的激活函数
- Kaiming随机
    - 思路和xavier类似，适用relu等非对称的

------------------
## 各种layer

### 手推梯度
- 上采样之后梯度和一致即可
- avgpooling：均分给上采样后的结果
- maxpooling：forward时记住maxindex，backward全部给它，其余置0

- sigmoid

  $x' =/frac{1}{1+e^{-x}}$, 求导后为$/frac{1}{(e^x+1)^2}$, 在0处的导数为0.25；整体导数是0处为0.25 两侧为0的类似正态分布的值

- [softmax](https://blog.51cto.com/u_15091060/2664060)+交叉熵

  - 设最后一层fc的参数：W: I * O；输入 x：B * I

  - $S_{ij} = xW^T, q_{ij}=Softmax(S_{ij})=e^{s_{ij}} / /sum(e^S)$, 

  - $loss= -/sum(p logq)$, p是onehot向量

    ![Softmax梯度推导_c++_14](https://s5.51cto.com/images/blog/202103/18/98169c526ae0f45b9d17badbe8a07cfb.png)

  - $x_iq_{ij}$ 和 $x_i(q_{iy_i}-1)$

    - softmax梯度： $q_j(1-q_j), -q_iq_j$

  - $a_{j}是输出，则L(a,y) = -/sum(y_{i}loga_i)$,  $y_i$是onehot向量，所以$L(a,y) = y_{j}loga_j, y_j=1$， 导数为$/frac{y_j}{a_j} $

  - ![softmax](https://raw.githubusercontent.com/dong03/picture/main/20220713085031.png)

    

- dropout层

  ![image-20220721082907972](https://raw.githubusercontent.com/dong03/picture/main/image-20220721082907972.png)

  以随机概率码掉x，剩下的要放大以保证总体信号强度不变

  梯度就$0; 1/(1-p)$

### 感受野

- 理论感受野
  - $r_l=r_{l-1} + (k_l-1)/times /prod_{i=1}^{l-1}s_i$， 其中r为感受野，k为kernelsize， s为stride，基本与深度成正比，与kernelsize成正比
  - 对于空洞卷积，相当于k更大，$k'=dilation /times(k_l-1)+1$
- 有效感受野
  - 和正态分布相关，虽然理论上能覆盖外围，但是权重影响很少
  - 和sqrt(层数)成正比，和kernelsize成正比
  - 堆层数不如大kernel

### 上采样

- PixelShuffle： depth to space，设scalefactor=r，产生$(1,r^2,H,W)$的特征图，循环采样为$(1,rH,rW)$,不同通道的相同位置放在一起（空间不变性？特征总是相似）
- Upsample（各种插值：最近邻、双线性、三次），unmaxpooling（和maxpooling的权重分配一样，记住index然后只赋值给最大的，其余为0）
- deconv：反卷积（并非数学意义的反卷积，只能恢复形状，不能恢复数值，需要学习。先通过stride补0，然后卷积核转置后正常卷积）

### FLOPs(浮点运算数)和计算量

加、乘各算一次（所以*2）, 涉及bias要+1

- 卷积： $(2*C_{int}*k^2-1)*C_{out}*H*W$， 参数量：($k^2 * C_{int} + bias) * C_{out}$

- 分组卷积：一个卷积核只对组内负责，则分成n个组参数量就减少到原本的$1/n$
  
- 深度可分离卷积：分组个数就是outchannel个数
  
  - $(2*C_{int}*k^2-1)*H*W$,$C_{out} = 1$, 参数($k^2 * C_{int} + bias)$
  - $(2*C_{int} -1) * H*W $, $k=1$, 参数$(C_{int} + bias) * C_{out}$
  - 加起来
  
- 池化：
  - 全局池化： $H_{int}*W_{int}*C_{int}$, 每个元素只计算一次
  - 一般窗池化：$k^2*H_{out}*W_{out}*C_{out}$，和窗的尺寸有关
  
- 全连接层
  - $2 * C_{int} * C_{out} - 1$, 参数$(C_{int} + bias) * C_{out}$,和1*1conv一致。
  
- 激活函数
  - ReLu：涉及判断，$H*W*C$
  - sigmoid: 四次计算，$4*H*W*C$
  
- 加不加bias：
  Wx+bias,后面有BN的话bias没有用，不用加，浪费显存。
  
- multi-head attention

  ![在这里插入图片描述](https://img-blog.csdnimg.cn/dcb2f732b856441189dbf833bc9ad6e2.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5bCP5ZGoXw==,size_20,color_FFFFFF,t_70,g_se,x_16)

  - 全连接层部分是$(4HW/times C^2)$ 

  - qk-> attention map 和av->output是矩阵乘法，$2 (HW)^2/times C$

  - 如果是swin transformer用sliding windows的方式，将attention局限在每个windows内；会影响矩阵乘法的复杂度（不会影响全连接层的）：

    每一个窗口是$M/times M$, 有$/frac{H}{M} /times /frac{W}{M}$个窗口要算，$2(MM)^2C/frac{H}{M}/frac{W}{M}=2M^2HWC$

  

### 激活函数

- tanh多用于rnn，因为会在不同时刻共享一个权重参数，所以对参数矩阵做了连乘，如果用relu 大于1，则容易梯度爆炸（CNN不考虑，权重共享，总有大有小）
- sigmoid：输出都大于0，有饱和区，导致收敛速度变慢； 梯度消失；幂运算计算慢
- h-sigmoid： $Hsigmoid=[0.5x+0.5]$截断到[0,1]，硬件友好的近似
- hswish： $x/times relu(x+3) / 6$
- relu：神经元会死；leaky relu， prelu（参数可学习）

### SE Layer  

$H /times W /times C$ -> mean pooling  $1 /times 1 /times C$ -> fc ->$1 /times 1 /times C$ , 糊到原始feature map上

- Dual attention的channel attention：保留了attention的基础形式  用C*C

  ![space attention](https://pic4.zhimg.com/80/v2-c5e61e5f85890bac20f266e767f66803_1440w.jpg)

  ![channel attention](https://pic3.zhimg.com/80/v2-3df5b7df514e7fcf9c07527ee2c7d7ae_1440w.jpg)

### 一些工程问题

- 在推理时翻转后融合的有效性
  - 应该是和卷积实现的对齐有关，feature map在降采样的时候左对齐， 所以左边和右边不一致，flip后不同
- 训练时BN层的方差和均值：
  - 训练时每个batch的值的一阶滑动？ $/alpha /times Now + (1-/alpha) /times Prev$， 其实可以看成 $/sum/alpha^pMean$，越早的影响越小

------------------
## 图像数据的归一化
- 其他数据：统一量纲，加速模型收敛，提高精度
- 图像数据0-255归一化
    - 输入值不要太大，对梯度有影响
    - 如果以0为中心分布，数据中心化，便于训练
        - 如果输入全为正，反向传播后局部梯度不会改变方向，权重或同时增加或同时减小，更新的效率较低



## 损失函数

### softmax实现
减去最大值防止溢出
```python
def softmax(f):
    f -= np.max(f)
    return np.exp(f) / np.sum(np.exp(f))
```
### softmax变体
<font color=red>[TBD](./softmax及其相关变形.md)</font>

- T-softmax
    - $exp(q/T) / sum(exp(q/T))$
    - 知识蒸馏，用来控制student输出分布与teacher输出分布相似度（而不只是对最终的logit约束
    - 和真值的onehot CE, 和teacher的分布CE
    - 如果将T取1，这个公式就是softmax，根据logit输出各个类别的概率。如果T接近于0，则最大的值会越近1，其它值会接近0，近似于onehot编码。如果T越大，则输出的结果的分布越平缓，相当于平滑的一个作用，起到保留相似信息的作用。如果T等于无穷，就是一个均匀分布。
### focal loss
- $-/alpha(1-p)^/gamma log(p)$
- $alpha$控制正负样本比例，$gamma$控制难度
- 当$gamma=2,alpha=0.25$，效果最好，这样损失函数在训练的过程中关注的样本优先级就是正难> 负样本难例 > 正易 > 负易了

### label smoothing

以e为概率，将标签预测错误。代入交叉熵后，等效为
1 -> 1-e, 0 -> e

### DiceLoss

面向指标的优化

梯度可视化(predict, dice, ce)

正样本梯度大于负样本，尤其是训练开始都接近0.5时

在接近0-1附近梯度值很小，存在饱和现象，失败了很难扭转回来

ce平等对待正负样本，当前梯度仅和label距离相关



![predict](https://pic4.zhimg.com/80/v2-b14bc4d3c182da701eb2155f6036755f_1440w.jpg)

![DICE梯度](https://pic3.zhimg.com/80/v2-b54caa8cf2a0f7ec2ae05b2f76bc2c42_1440w.png)

![celoss](https://pic1.zhimg.com/80/v2-cb59da9bf0327cf6da89def394599d68_1440w.jpg)



### 对loss的正则约束

- L1的鲁棒性更强，简单可解释：极值出现在坐标轴上（菱形），这样会导致很多权重为0，形成稀疏的权重矩阵，防止过拟合

- L2有更复杂的数据模式，产生平滑值，最优值是个球，优化时从各个方向向原点逼近，所有参数都比较小，防止过拟合。

  - 越是复杂的模型，越是尝试对所有样本进行拟合，包括异常点。这就会造成在较小的区间中产生较大的波动，这个较大的波动也会反映在这个区间的导数比较大。只有越大的参数才可能产生较大的导数。因此参数越小，模型就越简单。

- 回归的时候为什么适用l2 loss

  - 在欧式空间约束距离，特征距离代表距离远近

  - 平滑值，便于优化

  - 计算简单（不需要指定0的导数）

  - **最小二乘方法，是对符合正态分布的极大似然估计 **

    我们设真实值与预测值之间的误差为：

    $/epsilon_i = y_i - /hat{y_i}$

    我们通常认为误差 $/epsilon$ 服从标准正态分布$(/mu = 0, /sigma^2 = 1)$ ，即给定一个 $x_i$ ，模型输出真实值为 $y_i$ 的概率为：

    $$p(y_i |x_i) = /frac{1}{/sqrt{2pi}}*/exp(-/frac{/epsilon_i^2}{2})$$

    进一步我们假设数据集中N个样本点之间相互独立，则给定所有 $x$ 输出所有真实值 $y$ 的概率即似然Likeihood，为所有 $p(y_i|x_i)$ 的累乘：

    $$L(x,y) = /prod_{i=1}^{n}/frac{1}{/sqrt{2/pi}}*/exp[-/frac{/epsilon_i^2}{2}]$$

    取对数似然函数得：

    $$log[L(x,y)] = - /frac{n}{2}/log{2/pi}-/frac{1}{2}/sum_{i=1}^{n}/epsilon_i^2$$

    去掉与 $/hat{y_i}$ 无关的第一项，然后转化为最小化负对数似然：

    $$/frac{1}{2}/sum_{i=1}^{n}/epsilon_i^2=/frac{1}{2}/sum_{i=1}^{n}(y_i-/hat{y_i})^2$$

    

    和l2 loss一致

    

- 但是 一范数的约束有更强的鲁棒性，不产生平滑值

  - 对于取证恢复，通常不用l2，避免产生平滑值（同理不用pooling）

  - 归一化后l2的loss普遍偏小

  - 对抗攻击里，数字图像的攻击普遍使用二范数，但是迁移到物理世界的攻击，二范数根本没用，通常是一范数甚至零范数（是否变化）响应。



### 余弦距离-欧式距离

使用矩阵乘法快速计算余弦相似度

$dist=1-cos$

$educ = /sqrt{2dist}$， 成正相关

- 余弦距离不是真正的距离，满足非负性（距离大于等于0），同一性（当且仅当x相等时dist=0），对称性（dist(x1,x2) = dist(x2,x1)），但不满足三角不等式！欧式距离满足三角不等式，余弦有根号，不满足



------------------
## 优化器
不推公式
- 非自适应（SGD，学习率全程不变，或者按照一定的 learninglearning scheduleschedule 随时间变化）
  - 只依赖当前batch的梯度，不稳定
  - 带一阶动量（指数加权平均）的，抑制震荡，冲出局部最优
- 自适应（Adam）
- 快速验证的时候无脑adam就行，对稀疏数据有优势，追求精度可以花很大精力去调整SGDM。



New 公式推导[[Adam那么棒，为什么还对SGD念念不忘 | 吴良超的学习笔记 (wulc.me)](https://wulc.me/2019/03/18/Adam那么棒，为什么还对SGD念念不忘/)]

设待优化参数$/omega$, 初始学习率$/alpha$，在 $step: t$时，梯度$g_t$

- 则SGD有：

1. 下降梯度的一阶动量：$m_t=g_t$

2. $w_{t+1} = w_t - /alpha /times m_t$

   收敛速度慢，只和当前step相关，会持续震荡

- SGDM：SGD+momentum

给一阶动量加上指数移动平均值： $m_t = /beta.m_{t-1} + (1-/beta).g_t$

- Adagrad

**对于经常更新的参数，我们已经积累了大量关于它的知识，不希望被单个样本影响太大，希望学习速率慢一些；对于偶尔更新的参数，我们了解的信息太少，希望能从每个偶然出现的样本身上多学一些，即学习速率大一些。**

1. 使用二阶梯度衡量更新频率，调整学习率：$v_t=/sum g_t^2$
2. $/alpha_t= /alpha / /sqrt{v_t + /epsilon}$, 其中$/epsilon$ 为一防止分母为0的小值
3. 参数更新越频繁，二阶动量越大，学习率越小

- AdaDelta 

1. $v_t$累计，最终会使学习率趋近0， 未能收敛
2. 给二阶动量加指数移动平均值：$v_t = /beta_2.v_{t-1} + (1-/beta_2)/sum g_t^2$

- Adam

一阶、二阶动量都用上

------------------
## backbone
### mobilenet
- v1
    - 深度可分离卷积（deep-wise + point-wise）
    
      - ```python
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    **groups=in_ch**)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    **groups=1**)
        使用groups确定是否分组
        ```
    
        
    
    - 对每个channel各用一个卷积核做卷积（deep-wise）
    
    - 对上述特征使用一个1*1 point-wise变通道
    
    - 比直接使用一个卷积减少8-9倍参数量
    
- v2
    - 倒残差结构（ResNet：高-低-高，mobilenetv2：低-高-低）
    - 为了节约参数channel数更少，在低维空间容易丢特征，所以倒残差先升维再降维
    - 减少丢特征：最后一层不用relu；采用残差连接
    
- v3
    - swish激活函数 $swish(x) = x·sigmoid(x)$,更平滑，在0处可导
        - relu在0处的导数怎么选择？{0,1}中选一个就好，确定梯度下降的方向，一般选0，人为指定一条切线
    - h-swish: 对swish改进（sigmoid计算更耗时）$h-swish(x)=x·/frac {ReLU(6(x+3))}{6}$
    - 增加se模块
    - NAS搜索最后一层
### DenseNet
- 反复使用浅层特征，保留篡改痕迹
- 链式梯度相乘，跳层相加



### vision transoformer

- 划分为patch，展平；

- 1D position embedding拼给patch feature

  - 可学习，patch很大，只要有pos emb就没差；

  - patch数量和训练对不上时：对pos emb插值再finetune

    ```python
    def resize_pos_embed(posemb, posemb_new):
        ntok_new = posemb_new.shape[1]
        # 除去class token的pos_embed
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
        gs_old = int(math.sqrt(len(posemb_grid)))
        gs_new = int(math.sqrt(ntok_new))
        # 把pos_embed变换到2-D维度再进行插值
        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
        return posemb
    ```

    

- transformer encoder：获得长程信息（感受野大）

  - mhsa: multi-head, 把input_dim 拆成head份，分别做selfattention，再concat

    ![mhsa](https://pic4.zhimg.com/80/v2-6a11d3df9ff286e0f2b8194cbd4a16af_1440w.jpg)

  - 激活函数 GELU：高斯误差线性单元，以高斯概率分布mask掉一些值，越小越可能被mask掉，越大越会保留，有relu的思想

  - multi-head：总有一两个head与众不同，可以剪枝；各自关注local；效果更好

    ![img](https://pic4.zhimg.com/80/v2-2ef5ab8ca87fd443de9e730c4aa539ff_1440w.jpg)
  
  - layernorm：在样本内部，对hidden dim做归一化
  
    ![layernorm](https://pic2.zhimg.com/80/v2-beff996a551d1cbc78fbd2bf94df7af5_1440w.jpg)

​	CNN的BN在channel维度上统计均值和方差，同一个通道的featuremap由一个卷积核产生，分布要相对一致；transformer不同数据的相同token没什么道理

BN对于每一个通道，求batchsize * H *W的均值，最终留下的是channel-wise的值

LN对于每一个sequence位置在channel维度求均值，最终留下的是batch * token

- swin transormer：
  - 划分locality，把self-attention局限在local windows内，减少计算量
  - 多层，感受野
  - shift window

### make large conv great again

#### 重参数

- 对BN(conv)合并，且把形如conv的全部替换成conv3x3, 新的卷积完成工作

> conv: $z = wx + b$

>  bn: $y = /gamma  /frac{(z-mean)} {/sqrt{(var)}} + /beta$

>$w' = /beta w / /sqrt{(var)}$

>$b' = /beta(b-mean) / /sqrt{(var)} + /gamma$



- 对$K/times  K-BN + 1/times K-BN + K/times1-BN$三个平行结构，吸收为一个$K /times K$（ACNet）, 或其他类似设计，希望在训练时引入更多参数增加拟合效果，推理时都吸收掉

#### Transformer 设计经验

- 更大的感受野
  - 大核对局部能力较弱：设计小核分支，推理重参数化（和ACNet对上了）

- 更低的inductive bias（对目标的假设，CNN需要：空间不变性、kernel权重共享、local联系：距离近的联系更紧密），mlp更低

#### sth else

- 动态卷积核：（e.g. multi-expert），节省计算量
- glance & focus，多阶段抠图，涉及动态资源调度，工程上比较困难，难batch
- 卷积的FFT加速， TBD
- 空洞卷积：稀疏采样，有格点效应，不如直接大kernel； 不容易加速（访问、缓存机制）

## 指标

- AUC计算：

  1. 设M个正样本，N个负样本，共M*N个正负样本对。每一个对里，pos_score > neg_score记1，=记0.5，小于记0，求均值即可

  2. 按分数从小到大排序，取正样本的rank之和（如果分数相同，rank是相同分数的rank序号均值）

     ![auc](https://img-blog.csdn.net/20171129172101196)

## 杂项
- [常见数学概率题](knowledge/常见数学概率题.md)
- [反向传播算法](knowledge/反向传播算法.md)
- [极大似然估计](knowledge/极大似然估计.md)
- [卷积.池化.激活函数.损失.求导](knowledge/卷积.池化.激活函数.损失.求导.md)
- [评估指标](knowledge/评估指标.md)
- [优化器](knowledge/优化器.md)
- [BN](knowledge/BN.md)
- [Feature_engineer](knowledge/Feature_engineer.md)
- [Logistic](knowledge/Logistic.md)
- [Text_feature](knowledge/Text_feature.md)