# CV/多模态方向知识点储备

## 注意
- 招聘告一段落，目标方向是CV/多模态算法工程师，也拿到了若干大厂的SSP，简单记录下面试过程中的知识点供参考和存档。（***~~顺便积点德~~***）

- 方向包括面试技巧、数字图像处理知识点，leetcode题解记录，以及CV/多模态相关的知识点。

- **没写不代表不重要！** 只是记录了我自己遇到的问题，且不成体系。适合用于**查漏补缺**。

- 勘误请发issue。


## 目录

* [面试技巧](TIPS.md#面试技巧)
  * [反问](TIPS.md#反问)
  * [谈薪](TIPS.md#谈薪)
  * [面试问题](TIPS.md#面试问题)
  * [更多面经](TIPS.md#更多面经)
* [数字图像处理](DIP.md#数字图像处理)
  * [图像压缩](DIP.md#图像压缩)
    * [压缩步骤](DIP.md#压缩步骤)
    * [霍夫曼量化](DIP.md#霍夫曼量化)
    * [其他压缩](DIP.md#其他压缩)
  * [图像增强](DIP.md#图像增强)
    * [直方图匹配](DIP.md#直方图匹配)
    * [微分](DIP.md#微分)
  * [图像校正](DIP.md#图像校正)
    * [霍夫检测](DIP.md#霍夫检测)
      * [直线检测](DIP.md#直线检测)
      * [圆检测](DIP.md#圆检测)
    * [角点检测（FAST, SIFT)](DIP.md#角点检测fast-sift)
    * [校正](DIP.md#校正)
    * [HOG特征提取](DIP.md#hog特征提取)
    * [开闭操作](DIP.md#开闭操作)
    * [图像拼接](DIP.md#图像拼接)
* [leetcode](leetcode.md#leetcode)
  * [题](leetcode.md#题)
  * [Tips](leetcode.md#tips)
    * [图论](leetcode.md#图论)
        * [并查集：](leetcode.md#并查集)
        * [DIJKSTRA算法](leetcode.md#dijkstra算法)
        * [遍历图](leetcode.md#遍历图)
        * [<a href="https://leetcode\.cn/problems/course\-schedule/" rel="nofollow">课程表</a>](leetcode.md#课程表)
    * [排序](leetcode.md#排序)
        * [归并排序](leetcode.md#归并排序)
        * [堆排序](leetcode.md#堆排序)
        * [快速排序 / 快速选择](leetcode.md#快速排序--快速选择)
    * [树，堆，链表](leetcode.md#树堆链表)
        * [层序遍历恢复完全二叉树](leetcode.md#层序遍历恢复完全二叉树)
        * [二叉搜索树](leetcode.md#二叉搜索树)
        * [<a href="https://labuladong\.github\.io/algo/2/22/64/" rel="nofollow">二叉堆</a>](leetcode.md#二叉堆)
        * [双向带头链表](leetcode.md#双向带头链表)
        * [二叉树遍历(非递归)](leetcode.md#二叉树遍历非递归)
        * [others](leetcode.md#others)
        * [atoi,有效数字：有限状态机](leetcode.md#atoi有效数字有限状态机)
  * [python](leetcode.md#python)
    * [GIL](leetcode.md#gil)
    * [可变/不可变数据类型](leetcode.md#可变不可变数据类型)
    * [try\-except\-finally](leetcode.md#try-except-finally)
    * [class内的方法](leetcode.md#class内的方法)
    * [数组](leetcode.md#数组)

* [CV/多模态](DL.md#CV/多模态)
  * [参数初始化](DL.md#参数初始化)
  * [各种layer](DL.md#各种layer)
    * [手推梯度](DL.md#手推梯度)
    * [感受野](DL.md#感受野)
    * [上采样](DL.md#上采样)
    * [FLOPs(浮点运算数)和计算量](DL.md#flops浮点运算数和计算量)
    * [激活函数](DL.md#激活函数)
    * [SE Layer](DL.md#se-layer)
    * [一些工程问题](DL.md#一些工程问题)
  * [图像数据的归一化](DL.md#图像数据的归一化)
  * [损失函数](DL.md#损失函数)
    * [softmax实现](DL.md#softmax实现)
    * [softmax变体](DL.md#softmax变体)
    * [focal loss](DL.md#focal-loss)
    * [label smoothing](DL.md#label-smoothing)
    * [DiceLoss](DL.md#diceloss)
    * [对loss的正则约束](DL.md#对loss的正则约束)
    * [余弦距离\-欧式距离](DL.md#余弦距离-欧式距离)
  * [优化器](DL.md#优化器)
  * [backbone](DL.md#backbone)
    * [mobilenet](DL.md#mobilenet)
    * [DenseNet](DL.md#densenet)
    * [vision transoformer](DL.md#vision-transoformer)
    * [make large conv great again](DL.md#make-large-conv-great-again)
      * [重参数](DL.md#重参数)
      * [Transformer 设计经验](DL.md#transformer-设计经验)
      * [sth else](DL.md#sth-else)
  * [指标](DL.md#指标)
  * [杂项](DL.md#杂项)

