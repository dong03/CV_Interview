# leetcode

## 题

- [4. 寻找两个正序数组的中位数 - 力扣（LeetCode）](https://leetcode.cn/problems/median-of-two-sorted-arrays/) 两个有序数组寻找topk： 二者第k//2个，较小的前面全部去掉

- [11. 盛最多水的容器 - 力扣（LeetCode）](https://leetcode.cn/problems/container-with-most-water/) 找到左右两侧最大值的最小值

- [24. 两两交换链表中的节点 - 力扣（LeetCode）](https://leetcode.cn/problems/swap-nodes-in-pairs/) 迭代

- [22. 括号生成 - 力扣（LeetCode）](https://leetcode.cn/problems/generate-parentheses/submissions/)回溯左括号/右括号的数量；在_ ( _ )两个位置插n-1的和分解

- [31. 下一个排列 - 力扣（LeetCode）](https://leetcode.cn/problems/next-permutation/)

  - 首先从后向前查找第一个顺序对 (i,i+1)，满足 a[i] < a[i+1]。这样「较小数」即为 a[i]。此时 [i+1,n)必然是下降序列。
  - 如果找到了顺序对，那么在区间 [i+1,n) 中从后向前查找第一个元素 j 满足 a[i] < a[j]。这样「较大数」即为 a[j]。
  - 交换 a[i] 与 a[j]，此时可以证明区间 [i+1,n) 必为降序。我们可以直接使用双指针反转区间 [i+1,n) 使其变为升序，而无需对该区间进行排序。
  - 上一个排序：找到最后一个a[i]>a[i+1], 则必然[i+1,n）上升，找到第一个a[i] > a[j],交换，变为降序

- [32. 最长有效括号 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-valid-parentheses/)（dp, 贪心（左右遍历，一旦right数量大于left视为无效，==遍历两遍==））

- [跳跃游戏](https://leetcode.cn/problems/jump-game/) 贪心，维护一个rightmost，若ix在rightmost内，判断rightmost和ix + nums[ix]的最大值

  ![image-20220720215002921](https://raw.githubusercontent.com/dong03/picture/main/image-20220720215002921.png)

  如果遍历到上一个结尾，则一下子跳到目前最远的位置（超出范围也算能跳到）

- [除自己以外的乘积](https://leetcode.cn/problems/product-of-array-except-self/) 不用除法；O(n)复杂度前缀积、后缀积

- [寻找数组重复数](https://leetcode.cn/problems/find-the-duplicate-number/) ：成环 i->nums[i]

- [240. 搜索二维矩阵 II - 力扣（LeetCode）](https://leetcode.cn/problems/search-a-2d-matrix-ii/)从左到右，从上到下都是升序； 从右上角开始，向左变小，向下变大

- [312. 戳气球 - 力扣（LeetCode）](https://leetcode.cn/problems/burst-balloons/)戳气球，dp[i][j]表示在(i,j)范围内戳的最大分数,则 $dp[i][j] = dp[i][k] + nums[i] *nums[k]*nums[j] + dp[k][j]$, 从下往上 从左到右遍历

- [1042. 不邻接植花 - 力扣（LeetCode）](https://leetcode.cn/problems/flower-planting-with-no-adjacent/)回溯 / 本题度小于种类，所以任意不和已种植重合的均可

- [字符串正则匹配](https://leetcode.cn/problems/regular-expression-matching/ )

  - 当[字母*] 出现的时候，将其看作整体，如果匹配上了，既可以【保留该组合，考察字符串上一位是否与该组合匹配】，或【舍弃该组合，考察该位字符串是否与上一个匹配】；如果没有匹配上，则只能【舍弃该组合（相当于前一个字符匹配0次)】

    ```python
    if p[j-1] != '*'
        if s[i-1] == p[j-1] or p[j-1] == '.':
            dp[i][j] = dp[i-1][j-1]
        else:
            dp[i][j] = False
    else:
        if s[i-1] == p[j-2] or p[j-2] == '.':
    		dp[i][j] = dp[i][j-2] or dp[i-1][j]
        else:
            dp[i][j] = dp[i][j-2]
    ```
  
  - 边界条件：
  
    - i=0 j=0时，空字符串可以匹配；
    - i！=0 j=0的时候，空规则串一定不能匹配；
    - i=0，j！=0时候，且j=*，有可能匹配（匹配零次）有
  
    ```python
    for j in range(2, len(p)+1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]
    ```


- [贪心算法之区间调度问题 :: labuladong的算法小抄](https://labuladong.github.io/algo/3/27/97/) 安排会议室：扫描重合情况； 需要背

- [剑指 Offer 14- II. 剪绳子 II - 力扣（LeetCode）](https://leetcode.cn/problems/jian-sheng-zi-ii-lcof/)可以数学证明尽可能多的取3，取到小于4后直接乘剩下的长度（超长的就在乘的过程中一直取余）

- [剑指 Offer 33. 二叉搜索树的后序遍历序列 - 力扣（LeetCode）](https://leetcode.cn/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)若是后续遍历一定能构建一个二叉搜索树；则检查二叉搜索树的上下界即可

- [剑指 Offer 35. 复杂链表的复制 - 力扣（LeetCode）](https://leetcode.cn/problems/fu-za-lian-biao-de-fu-zhi-lcof/)复制的时候新节点和旧节点挨着，；注意有点像图的复制，可以用dsp复制，暂存dict，从源到clone

  ```python
  new = Node(val=cur)
  new.next = cur.next
  cur.next = new
  cur = cur.next.next
  
  if cur.random is not None:
  	cur.next.random = cur.random.next
  cur = cur.next.next
  
  #? cur.next = cur.next.next
  #  cur = cur.net
  ```

- [剑指 Offer 56 - I. 数组中数字出现的次数 - 力扣（LeetCode）](https://leetcode.cn/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/)独特次数的数字；用相同值异或为0；想办法分成两组去异或；先都异或一遍，然后找到任意位是1的数字把所有数字分成两组

- [剑指 Offer 62. 圆圈中最后剩下的数字 - 力扣（LeetCode）](https://leetcode.cn/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)约瑟夫环；f(n, m) n==1时索引肯定是0；每次删人都相当于前挪m位置，为$f(n,m) = (f(n-1, m ) + m) \% n$

- [LRU 缓存](https://leetcode.cn/problems/lru-cache/) 双线链表+字典；要求O(1)的复杂度；get O(1)用字典实现，但是数据没有固定顺序所以增删慢；链表有顺序，增删O(1)，但是查找慢；结合起来：使用hashmap获得要增删的node(key=值，value=节点），带一个带头的双向链表增删：增头删尾

- [格雷码](https://leetcode.cn/problems/gray-code/) 背吧， 从[0]开始，把上一位的翻转后（尾尾相同，首首相同）加2^(n-1)，（n-1的高位一定为0，只差一位）再拼起来。

- [最多删除一个字符得到回文](https://leetcode.cn/problems/RQku0D/) 双指针从两头 找到第一个不相等的地方，检查分别移除两端后是否可行

- [完全二叉树添加节点](https://leetcode.cn/problems/NaqhDT/) 除了最后一层都是满的，尽可能向左靠拢；层序遍历，先左后右，哪个空接到哪个后面；优化：保留上一次遍历后的queue，但是接了新节点后要把父节点加到队首方便下一次pull

- [日程表](https://leetcode.cn/problems/fi9suh/) 自平衡二叉树：做有序插入和检索

- [分割回文子串](https://leetcode.cn/problems/M99OJA/) 回溯，建立一个从ix位置，往后依次看是否构成回文

  ```python
  for ix in range(start_ix, len(s)):
      if dp[start_ix][ix]:
          temp.append(s[start_ix:ix+1])
          dfs(ix+1, temp)
          temp.pop(-1)
  
  ```

- [翻转字符](https://leetcode.cn/problems/cyJERH/) 两次遍历出前缀、后缀和，第三次遍历考察以某一位分界，前面全是0，后面全是1的翻转次数 遍历

- [外星词典](https://leetcode.cn/problems/Jf1JuT/) 拓扑排序，类似课程表，成对出现的，第一个不同的单词，有前后置关系。考察前置条件是否冲突（成环；前缀相同但前面的更长）。

- [最大的异或](https://leetcode.cn/problems/maximum-xor-of-two-numbers-in-an-array)：字典树，把数字编码为31位二进制的长度，考察第k位 (num>>k) & 1 是0 / 1 决定放左 / 右树，

  这样遍历到一个数字时，首先把它添加到树中。而后从根节点开找，根绝num的位是0 / 1，向相反的节点走（异或为1），如果该子节点不存在则本位为0	

  ```python
  class Trie:
      def __init__(self):
          self.left = None
          self.right = None
     	
      
      
  		HIGH_BIT = 30
      	def add(num: int):
              cur = root
              for k in range(HIGH_BIT, -1, -1):
                  bit = (num >> k) & 1
                  if bit == 0:
                      if not cur.left:
                          cur.left = Trie()
                      cur = cur.left
                  else:
                      if not cur.right:
                          cur.right = Trie()
                      cur = cur.right
  
      	def check(num: int) -> int:
              cur = root
              x = 0
              for k in range(HIGH_BIT, -1, -1):
                  bit = (num >> k) & 1
                  if bit == 0:
                      # a_i 的第 k 个二进制位为 0，应当往表示 1 的子节点 right 走
                      if cur.right:
                          cur = cur.right
                          x = x * 2 + 1
                      else:
                          cur = cur.left
                          x = x * 2
                  else:
                      # a_i 的第 k 个二进制位为 1，应当往表示 0 的子节点 left 走
                      if cur.left:
                          cur = cur.left
                          x = x * 2 + 1
                      else:
                          cur = cur.right
                          x = x * 2
              return x
  
  
  ```

- [162. 寻找峰值 - 力扣（LeetCode）](https://leetcode.cn/problems/find-peak-element/)注意是未排序数组，则只要从任意位置开始，一直往高处走，则可以到达峰值；二分每次扔掉一半不可行的区域（右边更大则往右走，左边更大则往左走，两侧补-inf

- [矩阵单词搜索（多个）](https://leetcode.cn/problems/word-search-ii/) 搜一个词是遍历i,j做dfs，搜多个不能重复搜；用前缀树做一次遍历

- [最长无重复子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/) 除了滑动窗口+counter外，维护一个table，记录该字母上一次出现位置。如果没有出现过 / 出现位置小于左边界，则更新res。否则左边界挪动到上次出现位置+1

- [最多的水](https://leetcode.cn/problems/container-with-most-water/) 贪心，盛水和宽度/两侧高度的最小值有关。双指针指向两侧，更新面积；然后挪动更小的一侧（否则高度被锁死、宽度还在减小）

- 阶乘中0的个数：数5的个数，数n中有多少个5^k 求和

- [324. 摆动排序 II - 力扣（LeetCode）](https://leetcode.cn/problems/wiggle-sort-ii/)

  - 不严格的比两侧大（≥，≤），则排序后，从第二个开始两两交换
  - 严格的比两侧大：

    - 需要排序后拆成前后两个数组，穿插；
    - 但是形如[1,2,2,3]的穿插后重复元素可能相邻。
    - 同一个数组内的重复元素会被隔开，必不相邻；说明是左数组的末尾和右数组的头部重复。
    - 让他们离得更远一点，拆分后分别翻转再插入。

- [376. 摆动序列 - 力扣（LeetCode）](https://leetcode.cn/problems/wiggle-subsequence/)

  ```python
  # dp[i][0,1] 以nums[i]为结尾的增、减序列的长度
  # 当nums[i] > nums[i-1]时：
  # 作为减序列
  # 任何以nums[i]结尾的减序列，都可以替换为nums[i-1]而长度不变
  # 作为增序列
  # 可以从增序列转移，替代nums[i-1]
  # 可以从减序列转移，在后面增加
  if nums[i] > nums[i-1]:
      dp[i][0] = dp[i-1][0]
      dp[i][1] = max(dp[i-1][1], dp[i-1][0] + 1)
  elif nums[i] < nums[i-1]:
      dp[i][1] = dp[i-1][1]
      dp[i][0] = max(dp[i-1][1]+1, dp[i-1][0])
  else:
      dp[i][0] = dp[i-1][0]
      dp[i][1] = dp[i-1][1]
  ```

- [239. 滑动窗口最大值 - 力扣（LeetCode）](https://leetcode.cn/problems/sliding-window-maximum/)单调减栈，比较pop出去的值和底部的是否一致

- [334. 递增的三元子序列](https://leetcode.cn/problems/increasing-triplet-subsequence/) 贪心，两次遍历，得到索引i处左侧最小值和右侧最大值，第三次遍历考察$left[i-1]<nums[i]<right[i+1]$

- [1541. 平衡括号字符串的最少插入次数 - 力扣（LeetCode）](https://leetcode.cn/problems/minimum-insertions-to-balance-a-parentheses-string/) 栈不太靠谱，记录目前有的左括号，看右括号数量够不够；右括号单多一个的话需要左右各需要补一个

- [855. 考场就座](https://leetcode.cn/problems/exam-room/) 只有O(n)解法，利用bisect维护有序数组，然后遍历

  ```python
  import bisect
  
  # A series of random numbers
  values = [14, 85, 77, 26, 50, 45, 66, 79, 10, 3, 84, 77, 1]
  
  print('New  Pos  Contents')
  print('---  ---  --------')
  
  l = []
  for i in values:
      position = bisect.bisect(l, i)
      bisect.insort(l, i)
      print('{:3}  {:3}'.format(i, position), l)
  ```

- [718. 最长重复子数组](https://leetcode.cn/problems/maximum-length-of-repeated-subarray/) 动态规划（注意和子序列的区别，当不同时直接为0）；滑动窗口：类似卷积，对上后统计overlap部分相同的长度。维护一个函数，四个参数：两个数组*两端；分为三部分；

  ![错开比较.gif](https://pic.leetcode-cn.com/9ed48b9b51214a8bafffcad17356d438b4c969b4999623247278d23f1e43977f-%E9%94%99%E5%BC%80%E6%AF%94%E8%BE%83.gif)

- [662. 二叉树最大宽度](https://leetcode.cn/problems/maximum-width-of-binary-tree/) 上层节点的pos的左右节点是pos\*2， pos\*2+1， 即便中间有空节点也能正确赋值；层序遍历把比较停在一层

- [264. 丑数 II](https://leetcode.cn/problems/ugly-number-ii/) 除了用小顶堆和set之外，还可以用dp，维护三个指针，指向乘以对应的倍数（2，3，5），取min作为对应的i+1，然后指针+=1

  ```
  相当于3个数组，分别是能被2、3、5整除的递增数组，且每个数组的第一个数都为1。
  然后就简单了，维护三个指针，将三个数组合并为一个严格递增的数组。就是传统的双指针法，只是这题是三个指针。
  然后优化一下，不要一下子列出这3个数组，因为并不知道数组预先算出多少合适。
  这样就一边移指针，一边算各个数组的下一个数，一边merge，就变成了题解的动态规划法的代码。
  ```

- [395. 至少有 K 个重复字符的最长子串](https://leetcode.cn/problems/longest-substring-with-at-least-k-repeating-characters/) 所有出现次数小于k的都不能出现，用它们split 然后递归

  ```python
  def longestSubstring(s, k):
  	if len(s) < k:
          return 0
      count = Counter(s)
      for s_ in count:
          if count[s_] < k:
              return max([longestSubstring(each, k) for each in s.split(s_)])
      return len(s)
  ```

- [32. 最长有效括号](https://leetcode.cn/problems/longest-valid-parentheses/) 两个flag记录左右括号的个数；从左往右时右边更大则清零，从右往左时左边更大清零；相等的时候更新结果，前后遍历两遍

- [41. 缺失的第一个正数](https://leetcode.cn/problems/first-missing-positive/) 依据数组实现1-N内的哈希；长为N的数组第一个没出现的正数一定在[1, n+1]之间，不在这个范围里的数都不用管；遇到了就把num-1作为索引 变为负数，考察第一个数大于零的索引

- [907. 子数组的最小值之和](https://leetcode.cn/problems/sum-of-subarray-minimums/) 单调栈，找到左右两个更小的值（不难，类似最大矩形），然后得到子数组的个数：（mid - left）* （right-mid）*nums[mid],  可以在nums两侧加-inf处理边界

- [227. 基本计算器 II](https://leetcode.cn/problems/basic-calculator-ii/) ，[224. 基本计算器](https://leetcode.cn/problems/basic-calculator/) 用栈来模拟，考察更多优先级。如果是+-法，根据符号append进栈；乘除法与栈里的最后元素合并；遇到左括号开始递归，到右括号break return，所以需要一个全局的量考察当前遍历到第几个元素了（一个flag，或者遍历的时候直接对数组左出）==TBD==

  ```python
  
  ```

- [300. 最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/)：除了dp，贪心做法：如果要增加，尽可能增加比上一位大，但小的值。维护d[i]表示长度为i+1的上升子序列中，结尾最小的值。维护一个max-, max-[ix]表示以ix结尾的最长长度；Onlogn，On复杂度

  ```python
  nums =[16, 2, 12, 10, 3, 5, 17, 9, 7, 2, 10, 20, 20, 5, 11, 4, 11, 11, 15, 7]
  d = []
  max_ = [1 for _ in range(len(nums))]
  for ix, n in enumerate(nums):
      if not d or n > d[-1]:
          d.append(n)
          max_[ix] = len(d)    
      else:
          l, r = 0, len(d) - 1
          loc = r
          while l <= r:
              mid = (l + r) // 2
              if d[mid] >= n:
                  loc = mid
                  r = mid - 1
              else:
                  l = mid + 1 
          d[loc] = n
          max_[ix] = loc + 1
  
  lens = len(d)
  res = [0 for _ in range(len(d))]
  # 从后往前找最长的去对应
  for i in range(len(nums)-1, -1,-1):
      if max_[i] == lens:
          res[lens-1] = nums[i]
          lens -= 1
          
  print(d), print(res)
  ```

- [862. 和至少为 K 的最短子数组](https://leetcode.cn/problems/shortest-subarray-with-sum-at-least-k/) 


  - 如果都是正数：滑窗解决 扩张-收缩

  - 带有负数：双端队列+单调栈

    ```python
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        pre = 0
        res = len(nums)+1
        queue = deque([(0,0)])
        for r in range(1,len(nums)+1):
            pre += nums[r-1]
            # 如果前面比当前大，没必要考虑，都pop出去
            while len(queue) and queue[-1][1] >= pre:
                queue.pop()
            # 单增序列，后面满足前面一定满足且前面更长，从前往后依次pop满足条件的
            while len(queue) and pre - queue[0][1] >= k:
                ix, _ = queue.popleft()
                res = min(r+1 - ix, res)
            queue.append((r+1, pre))
        return res if res != len(nums)+1 else -1
    ```

- [470. 用 Rand7() 实现 Rand10()](https://leetcode.cn/problems/implement-rand10-using-rand7/) 古典概率相乘，通过拒绝的方式获得更小的采样

  ![image-20220807144926432](https://raw.githubusercontent.com/dong03/picture/main/image-20220807144926432.png)

- [958. 二叉树的完全性检验](https://leetcode.cn/problems/check-completeness-of-a-binary-tree/) 完全二叉树层序遍历到最后一定全是None；这样None也要append，当第一次遇到None时，加一个flag；后续如果有机会检验到flag说明None后有值，不完全。

- [400. 第 N 位数字](https://leetcode.cn/problems/nth-digit/)

  ```python
  def findNthDigit(self, n: int) -> int:
      prev = 0
      d = 1
      # d位数总计数字：1：9，2：90*2，3：3*900
      while prev < n:
          prev += d * 9 * 10 ** (d-1)
          d += 1
      d -= 1
  	# 
      prev -= d * 9 * 10 ** (d-1)
      index = n - prev - 1
      start = 10 ** (d - 1)
      # 剩下的index够凑出多少个d位数
      num = start + index // d
      # 是当前num的第几位
      digit = index % d
      # 用tmd字符串吧 看不懂题解的那个计算方式
      return int(str(num)[digit])
  ```

  

  TBD:

  [391. 完美矩形](https://leetcode.cn/problems/perfect-rectangle/)

  [827. 最大人工岛](https://leetcode.cn/problems/making-a-large-island/)

  
  ## Tips
  ### 图论
  ##### 并查集：
  
  - 应用场景：m个元素，有n个相等关系，合并为k个集合
  
  - 初始化：每个node的父节点是自己
  
  - 合并关系：node1，node2各自找到自己的父节点，把秩低的接在高的后面（高的作为父节点，合并复杂度更低）
  
  - 查找：往父节点倒直到node==father_node；在查找过程中将当前节点移动到父节点作为优化； or 直接father[x] = find(father(x)), 彻底打平为一层（因为只在乎最终节点是什么，而不在乎中间路径）
  
  - 合并复杂度O(h) 树的高度
  
  - 除法问题（边带权重）；1 v N的合并
  
    ```python
    # extra weight
    def union(x, y, div=v_x/v_y):
        rootx = find(X)
        rooty = find(y)
        if rootx == rooty:
            return
        father[rootx] = rooty
        weight[rootx] = weight[y] *value / weight[x]
        # 跟rooty没关系
        
    def find(x):
        if father[x] != x:
            temp = father[x]
            father[x] = find(father[x])
            weight[x] *= weight[temp]
    
        return father[x]
    
    def div(x, y):
        rootx = find(x)
        rooty = find(y)
        if rootx != rooty:
            return -1
        return weight[x] / weight[y]
    ```
  
  ![image-20220719084850474](https://raw.githubusercontent.com/dong03/picture/main/image-20220719084850474.png)
  
  ##### DIJKSTRA算法
  
  - 从最初的节点开始，把直接相连的都入最小堆，更新相连接点的总距离；其余为max
  
  - 依次pop出堆里最小的（贪心算法），如果新的距离小于之前暂存的距离，则更新与之相连的边
  
    ```python
    graph = [[] for _ in range(n+1)]
    for from_, to_, time in times:
        graph[from_].append((to_, time))
    res = [float('inf') for _ in range(n+1)]
    
    hep = [(k, 0)]
    while len(hep):
        node, time = heapq.heappop(hep)
        if res[node] < time:
            continue
        res[node] = time
        for next_node, new_time in graph[node]:
            if res[next_node] > new_time + res[node]:
                res[next_node] = new_time + res[node]
                heapq.heappush(hep, (next_node, res[next_node]))
    ```
  
  ##### 遍历图
  
  - 最长路径：DFS，从任意起点搜到终点，再从该终点开始搜到下一个终点。后半段是最长路径
  - 最短路径：BFS，有节点出度为0时则停止（二叉树的最小层数？）；DIJKSTRA算法，权重就是1
  - 克隆图：都可以 习惯DFS [克隆图 - 克隆图 - 力扣（LeetCode）](https://leetcode.cn/problems/clone-graph/solution/ke-long-tu-by-leetcode-solution/) 相比树的克隆多了一个哈希表记录哪些已经被clone了，已有了就返回cloned，避免重复克隆（树不用是因为是单向无环）
  
  ##### [课程表](https://leetcode.cn/problems/course-schedule/ )
  
  - dfs 维护一个checklist，已搜索（满足依赖项，True），未搜索（递归），搜索中（成环，False）三态；后续遍历的翻转即是路径（先子节点再父节点，翻转就是 from->to；建图用from->to
  - bfs方案：记录每个节点的入度（依赖数），维护一个入度为0的queue；对于pop出的节点，其出度对应的节点入度-=1；考察最后是否所有节点都被遍历；pop顺序就是拓扑遍历顺序
  
  ### 排序
  
  ##### 归并排序
  
  - 双指针合并归并后的两半
  
     ```python
     import random
     import copy
     
     nums = [random.randint(0,30) for _ in range(30)]
     temp = copy.copy(nums)
     print(nums)
     
     def partition(left, right):
         if left >= right:
             return
         mid = (left + right) // 2
         partition(left, mid)
         partition(mid+1, right)
         merge(left, right, mid)
     
         return 
     
     def merge(left, right, mid):
         temp[left:right+1] = nums[left:right+1]
         l, r, m = left, mid+1, left
         while l <= mid and r <= right:
             if temp[l] < temp[r]:
                 nums[m] = temp[l]
                 l += 1
             else:
                 nums[m] = temp[r]
                 r += 1
             m += 1
         while l <= mid:
             nums[m] = temp[l]
             l+=1
             m+=1
         while r <= right:
             nums[m] = temp[r]
             r+= 1
             m += 1
         return
     
     partition(0, len(nums)-1)
     print(nums)
     李涛
     ```
  
  - [剑指 Offer 51. 数组中的逆序对](https://leetcode.cn/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)：
  
     - 合并时，如果$nums[lflag] > nums[rflag]$, 则$[lflag, mid]$全部与$nums[rflag]$构成逆序对,$res += (mid - lflag +1)$
     - 或者和下面对齐：当刚刚满足$nums[lflag] <= nums[rflag]$时，从$[mid+1到rflag-1]$全部与$nums[lflag]$构成逆序对，$res += (rflag - mid - 1)$
  
  - [315. 计算右侧小于当前元素的个数](https://leetcode.cn/problems/count-of-smaller-numbers-after-self/)：在向右增加rflag时，如果刚好满足nums[lflag] <= nums[rflag], 则mid+1到rflag-1之间全部比nums[lflag]小；增加一个idx数组记录索引（归并排序会改变顺序）
  
     ```
     在对 nuns[lo..hi] 合并的过程中，每当执行 nums[p] = temp[i] 时，就可以确定 temp[i] 这个元素后面比它小的元素个数为 j - mid - 1
     ```
  
  - <span id="merge">时间复杂度要求O(1)的merge过程</span>
  
     - 思路1：本质上还是次数少的冒泡
  
     ```python
     def sort(nums = [1,2,3,4,5,6,2,3,4,5]):
         # 找到前后两个数组的分界
         for ix in range(1, len(nums)-1):
             if nums[ix] < nums[ix-1] and nums[ix] < nums[ix+1]:
                 break
         if ix == len(nums) -2:
             return nums
     
         left, right = 0, ix
         # right从后半部分， 有比left小的就都提到前面去
         while left < right and right < len(nums):
             while nums[left] < nums[right]:
                 left += 1
             temp = nums[right]
             for ix in range(right, left, -1):
                 nums[ix] = nums[ix-1]
             nums[left] = temp
             right += 1
         return nums
     ```
  
     - 思路2：把归并排序的O(n)空间复杂度变为0(1),时间复杂度保留原本的O(nlogn)
  
       [一个数保存两个数](https://zhuanlan.zhihu.com/p/159446051)
  
     ```python
     nums = [...]
     def merger(left, mid, right):
         max_val = max(nums) + 1
         i, j, k  = left, mid+1, left
         while i <= mid and j <= right:
             if nums[i] % max_val < nums[j] % max_val:
                 nums[k] = nums[k] + max_val * (nums[i] % max_val)
                 i += 1
             else:
                 nums[k] = nums[k] + max_val * (nums[j] % max_val)
                 j += 1
             k += 1
         while i <= mid:
             nums[k] = nums[k] + max_Val * (nums[i] % max_val)
             k += 1
             i += 1
         while j <= right:
             nums[k] = nums[k] + max_Val * (nums[j] % max_val)
             k += 1
             j += 1        
         for i in range(left, right+1):
             nums[i] = nums[i] // max_val
     ```
  
  ##### 堆排序
  
  参考二叉堆的sink函数，需要比较父节点与子节点的关系并交换
  
  ```python
  nums = [:]
  def heap_adjust(f_ix, end):
      s_ix = f_ix * 2 +1 
      while s_ix < end:
          if s_ix +1 < end and nums[s_ix] < nums[s_ix+1]:
              s_ix = s_ix + 1
          
          if nums[f_ix] >= nums[s_ix]:
              break
          else:
              nums[f_ix], nums[s_ix]= nums[s_ix], nums[f_ix]
              f_ix = s_ix
              s_ix = 2*f_ix +1
  def heapsort():
      # 初始化
      # 从非叶子节点开始（叶子节点没有子节点）
      for i in range(len(nums)//2-1, -1, -1):
          heap_adjust(i, len(nums))
      # 把最大的调整到最后，然后排剩下的内容
      for i in range(len(nums)-1, 0, -1):
          nums[0], nums[i] = nums[i], nums[0]
          heap_adjuest(0, i)
  ```
  
  
  
  ##### 快速排序 / 快速选择
  
  ~~~python
  # nums:List
  # k:int (top k_th largest)
  
  def quick_sort(left, right):
      if left >= right:
          return
      mid = get_mid(left, right)
      quick_sort(left, mid-1)
      quick_sort(mid+1, right)
  
  def quick_selct():
      start, end = 0, len(nums)-1
      target = len(nums) - k
      mid = get_mid(start, end)
      while mid != target:
          if mid > target:
              end = mid - 1
          else:
              start = mid + 1
      	mid = get_mid(start, end)
  	return nums[mid]
  
  def get_mid(start, end):
      ix = random.randint(start, end)  # 避免有序
      nums[ix], nums[end] = nums[end], nums[ix] # 换到最右边
      split_ix = start
      for ix in range(start, end):
          if nums[ix] < nums[end]: 
              nums[ix], nums[split_ix] = nums[split_ix], nums[ix]
              split_ix += 1
              # 找到比分界线小的，就挪到split_ix的左边 
  
      nums[split_ix], nums[end] = nums[end], nums[split_ix]
      # nums[start:split_ix] < nums[split_ix] < nums[split_ix+1:end+1]
      return split_ix
  ```
  ~~~
  
  ### 树，堆，链表
  
  ##### 层序遍历恢复完全二叉树
  
  ```python
  root = TreeNode(nums[0])
  nodes = [root]
  ix =0 
  while len(nodes):
      node = nodes.pop(0)
      
      node.left = TreeNode(nums[ix+1]) if ix+1 < len(nums) else None
      
      if node.left is not None:
          nodes.append(node.left)
          
      node.right = TreeNode(nums[ix+2]) if ix+2 < len(nums) else None
      if node.right is not None:
          nodes.append(node.right)
         
      ix = ix+2   
      
  ```
  
  ##### 二叉搜索树
  
  - 二叉搜索树的下一个节点：
    - 如果空节点返回空
    - 如果有右子树，则到右子树的左下角
    - 往前递归父节点直到当前节点是父节点的左子树，返回父节点
  
  ##### [二叉堆](https://labuladong.github.io/algo/2/22/64/)
  
  - 最小堆：父节点比两个子节点都小；在空置idx=0的情况下， 索引ix的左子节点：2*ix，右子节点：2*ix+1
  
  - 对底部节点上浮：需要比较与父节点的大小，如果小于父节点则交换 ；直到左侧边界 / 小于父节点
  
    ```python
    # 对第k个节点上浮
    def father(x):
        return x // 2
    def swim(x):
        while x > 1:
            father_ix = father(x)
            if nums[father] > nums[x]:
                nums[x], nums[father_ix] = nums[father_ix], nums[x]
                x = father_ix
            else:
                break
    ```
  
  - 对顶部节点下沉：需要与两个子节点分别比较大小，选择最小的交换；直到右侧边界 / 小于子节点
  
    ```python
    def child(x):
        return x*2, x*2+1
    
    def sink(x):
        while True:
    	    left, right = child(x)
            if left >= len(nums):
                break
            # 换到左右节点中更小的
    		min_ix = left
            if right < len(nums) and nums[left] > nums[right]:
                min_ix = right
    		# 如果比父节点小，则交换
            if nums[min_ix] < nums[x]:
                nums[x], nums[min_ix] = nums[min_ix], nums[x]
             	x = min_ix
            else:
                break
    ```
  
  - 插入节点：先插到尾部，再上浮
  
    ```python
    def insert(x):
        nums.append(x)
        swin(len(nums)-1)
    ```
  
  - 删除节点：把尾部放到该位置，再下沉；删除最大：首尾互换
  
    ```python
    def delnode(ix):
        nums[ix] = nums[-1]
        nums.pop(-1)
        sink(ix)
    ```
  
  ##### 双向带头链表
  
  - 元素各不相同，用table记录 val->node
  - 删除不需要像单向链表一样从prev到next，找到当前node后直接把prev和next连起来
  - 头尾各一个虚拟节点
  - [146. LRU 缓存](https://leetcode.cn/problems/lru-cache/)， 维护 move_to_head, remove_tail, add_head, remove_node等功能，可以O(1)
  
  ##### 二叉树遍历(非递归)
  
  - 中序遍历（左中右）：一路append左节点，**pop(-1)后操作（print，添加，记录等）**从右节点再一路append左节点
  
    ```python
    cur = root
    stack = []
    res = []
    while cur and len(stack):
        while cur:
            stack.append(cur)
            cur = cur.left
        if len(stack):
            cur = stack.pop(-1)
            ####后操作
            res.append(cur.val)
            cur = cur.right
    ```
  
    
  
  - 前序遍历（中左右）：一路append左节点，**先操作再pop(-1)**,从右节点再一路向左
  
    ```python
    cur = root
    stack = []
    res  = []
    while cur and len(stack):
        while cur:
            ####先操作
            res.append(cur.val)
            stack.append(cur)
            cur = cur.left
        if len(stack):
    	    cur = stack.pop(-1)
    		cur = cur.right
    ```
  
    
  
  - 后序遍历（左右根）：比较麻烦，根右左再反序变成左右根
  
    ```python
    cur = root
    res, stack = [], []
    while cur and len(stack):
        while cur:
            res.append(cur.val)
            stack.append(cur)
            #### 根右左再反序
            cur = cur.right
       	if len(stack):
            cur = stack.pop(-1)
            cur = cur.left
    res[::-1]
    ```
  
  
  ##### others
  
  ---------------------------
  
  - 测试用例：树的，特殊形状（空、根、单侧左右偏，八字，普通树），排列组合生成复杂的情况
  
  - 位运算：
  
      - x & (x-1) 把二进制最后一个1变成0；如果是$2^n$则二进制位仅有一个1，操作后会变为0
  
      - bit = (num >> k) & 1，获得高位第k个是0还是1
  
      - 二进制不进位加法：异或
  
      - 判断异号：(x ^ y) < 0
  
      - [137. 只出现一次的数字 II](https://leetcode.cn/problems/single-number-ii/)  1个1次+其余三次：二进制位运算，统计每个二进制位的1的数量， %3，则剩下的一定是单独那一个的位
  
  - 取模：$(a\times b)\%c = (a\%c \times b \%c) \%c$
  
  - 罗马数字和整数互转：基本规则是大数在小数前面；如果小数在前面则减去小数字（IV：5-1=4）；整数转的时候优先减去最大的数
  
  - 字典序排序： return  int(a+b) -  int(b+a)
  
      - 使用O(n)时间复杂度和O(1)空间复杂度：按位递归（有O(n)的栈，改迭代）
  
  
  					![image-20220802085438702](https://raw.githubusercontent.com/dong03/picture/main/image-20220802085438702.png)	
  
  - 二分：上下有界且单调，最大值求最小、最小值求最大，可以在外面套用二分的框架
  
    [如何运用二分查找算法 - labuladong - 博客园 (cnblogs.com)](https://www.cnblogs.com/labuladong/p/12320485.html)
  
    ```python
    # 给定一个n的数组，分成M段，求和的最大值的最小值
    # 二分下界：单个的最大值；上界：求和
    # check(k) 检查最大值的最小值为k是否满足
    # 遍历的时候每够k就分段，考察能否分成m段
    ```
  
  - 求平方根：牛顿迭代法，让$x=(x+\frac{a}{x})/2$, x随意初始化
  
      ```python
      def mysqrt(x, a):    
          res = (x + a/x) / 2
          if abs(res -x) < 1e-6:
              return x
          else:
              return mysqrt(res, a) 
      # 初始值随便设
      mysqrt(10,10)
      ```
  
      
  
  - 搜索旋转数组：要找到有序的分界点；
  
      - 如果没有重复元素，简单比较下nums[mid]和nums[right]谁大就行，如果mid大则left左移到mid。此时旋转中心和最小值的最小索引重合
  
      - 如果有重复元素：最小值的最小索引和旋转中心不一定重合  $[1,1,1,1,1,2,2,1,1,1]$
  
          - 找最小索引，则在mid==right时，right -= 1
  
          - 找旋转中心，则应先把左侧边界重复的消除
  
              ```python
              while left < len(nums) and nums[right] == nums[left]:
              	left += 1 
              while left < right:
              	mid =left+(right-left)//2
              
              	if nums[mid] > nums[right]:
              		left = mid+1
                  else:
              		right = mid
              ```
  
              而后寻找target的操作也应注意nums[mid] == target时，应right=mid向左收缩
  
  - 数组：连续内存，可以O(1)索引随机元素，但只能对末尾位置O(1)增删，其余位置是O(n)；
  
      hashset：可以O(1)索引
  
  - 背包问题：
    - 01背包：不能复用，防止覆盖物品要倒序遍历，两层循环顺序都相同
      - 达到target的组合数： += dp[j-nums[i]]
      - 能否达成target：dp[j] or dp[j-nums[i]]
      - 最高价值：max(dp[i], dp[j-nums[i]] + nums[i])
    - 完全背包：
      - 能否凑成，遍历顺序不限
      - 求组合数：让物品出现的**先后顺序固定**；外层遍历物品，内层遍历背包
      - 求排列数：外层遍历背包，内层遍历物品
  
  
  
  - [字符串哈希编码](https://leetcode.cn/problems/longest-happy-prefix/)
  
    - 哈希值相同，字符串才相同
    - 进制base选择比26大的，为了防止越界，设置一个质数模（e,g, 10e7+7）,设置当前位数1
    - cur = ord(s) - ord('a'), prev = 0
    - 加到尾部的： (prev  * base + cur) % mod
    - 加到头部的：高位倍数mul = (mul * base) % mod,  (cur * mul + prev) % mods
  
  
  
  - 顺时针旋转图像：沿左上-右下对角线翻转；沿y轴翻转
  
  - 阶乘后0的个数：取决于有多少5. 讨论n中有多少5^k 求和
  
  
  
  
  - [股票买卖](https://labuladong.gitee.io/algo/3/27/96/)（动态规划转移方程，按最一般的情况讨论)
  
    - $dp[i][0,1][k]$ 第i天，最大k次交易，不持有/持有两种状态，k在买入时+1
  
      ```
      dp[-1][...][0] = 0
      解释：因为 i 是从 0 开始的，所以 i = -1 意味着还没有开始，这时候的利润当然是 0。
      
      dp[-1][...][1] = -infinity
      解释：还没开始的时候，是不可能持有股票的。
      因为我们的算法要求一个最大值，所以初始值设为一个最小值，方便取最大值。
      
      会被化简为max(-inf, 0-price[0])
      
      dp[...][0][0] = 0
      解释：因为 k 是从 1 开始的，所以 k = 0 意味着根本不允许交易，这时候利润当然是 0。
      
      dp[...][0][1] = -infinity
      解释：不允许交易的情况下，是不可能持有股票的。
      因为我们的算法要求一个最大值，所以初始值设为一个最小值，方便取最大值。
      ```
  
      完整转移：
  
      ```c++
      
          // k = 0 时的 base case
          for (int i = 0; i < n; i++) {
              dp[i][0][1] = Integer.MIN_VALUE;
              dp[i][0][0] = 0;
          }
      
          for (int i = 0; i < n; i++) 
              for (int k = max_k; k >= 1; k--) {
                  if (i - 1 == -1) {
                      // 处理 i = -1 时的 base case
                      dp[i][k][0] = 0;
                      dp[i][k][1] = -prices[i];
                      // dp[i][k][1] = max(dp[-1][k][1], dp[-1][k][0] - prices[i])
                      //             = max(-inf, 0-prices[i])           
                      //			   = -prices[i]
                      continue;
                  }
                  dp[i][k][0] = Math.max(dp[i-1][k][0], dp[i-1][k][1] + prices[i]);
                  dp[i][k][1] = Math.max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i]);     
              }
      ```
  
  
  
  ### 
  
  ##### atoi,有效数字：有限状态机
  
  ![image-20220726134246808](https://raw.githubusercontent.com/dong03/picture/main/picgo202207261342323.png)
  
  
  
  ![fig1](https://assets.leetcode-cn.com/solution-static/65/1.png)
  
  |                  | +-         | 数字     | ‘小数点’         | 字符e |
  | ---------------- | ---------- | -------- | ---------------- | ----- |
  | start            | +-号       | int      | 小数点（开头）   | end   |
  | +-号             | end        | int      | 小数点（开头）   | end   |
  | int              | end        | int      | 小数点（不开头） | 字符e |
  | 小数点（开头）   | end        | 小数部分 | end              | end   |
  | 小数点（不开头） | end        | 小数部分 | end              | 字符  |
  | 小数部分         | end        | 小数部分 | end              | 字符e |
  | 字符e            | 指数后+-号 | 指数位   | end              | end   |
  | 指数后+-号       | end        | 指数位   | end              | end   |
  | 指数位           | end        | 指数位   | end              | end   |
  | end              | break      | break    | break            | break |


## python
### GIL
- GIL是python的全局解释器锁，同一进程中假如有多个线程运行，一个线程在运行 python程序的时候会霸占python解释器(加了一把锁即GIL),使该进程内的其他程 无法运行，等该线程运行完后其他线程才能运行。如果线程运行过程中遇到耗时操作，则 解释器锁解开，使其他线程运行。所以在多线程中，线程的运行仍是先后顺序的，并不是同时进行。所以python没有多线程？
    - IO密集的程序用多线程，sleep时切换其他线程，一个崩溃全都崩溃

- 多进程中因为每个进程都能被系统分配资源，相当于每个进程有了一个python解释器， 所以多进程可以实现多个进程的同时运行，缺点是进程系统资源开销大
    - 多核要用多进程
### 可变/不可变数据类型
- 使用id()查看地址

- 不可变数据类型：数值型、字符串型string和元组tuple
    - 不允许变量的值发生变化，如果改变了变量的值，相当于是新建了一个对象，而对于相同的值的对象，在内存中则只有一个对象(一个地址),

- 可变数据类型：列表list和字典dict
    - 允许变量的值发生变化，即如果对变量进行append. +=等这种操作后，只是改变了变量的值，而不会新建一个对象，变量引用的对象的地址也不会变化，不过对于相同的值的不同对象，在内存中则会存在不同的对象，即每个对象都有自己的地址，相当于内存中对于同值的对象保存了多份，这里不存在引用计数，是实实在在的对象。
- 传值还是传址？
    - 可变的传地址，函数内操作也会变
    - 不可变的传值，无法改变函数外的变量
- =，copy，deepcopy
    - =对可变数据类型是给了相同的内存地址，修改会互相影响；对不可变的无影响
    - copy 对没有深度的列表已经可以复制走互不干扰了（copy.copy()）
    - deepcopy 对列表里的列表也可以赋新地址，复杂子对象，copy补星

### try-except-finally
```python
try:
    do sth
except Exception as e: # 捕捉错误信息
    print(e) 
finally:
    do sth # 一定会执行
```

### class内的方法
- 魔法方法也适用
```python
__str__(self) # print该类时会打印return值

```

### 数组

- 数组的内存是连续的，除了删增最后一项，否则要移动其他元素的地址（考虑复杂度）

  - 优先队列（最大最小堆）：

    - 从数组原地转换，时间复杂度O(n)  

      ```python
      heapq.heapify(data)
      heapq._heapify_max(data)
      ```

    - pop最小值时间复杂度O(1)

      ```python
      heapq.heappop(data)
      heapq._heappop_max(data)
      ```

  - 双端队列（deque）

    - 正常数组pop(-1)、insert(-1)是O(1), 但pop(0)、insert(0)是O(n)
    - 可以popleft, appendleft

- 是连续的，可以随机索引