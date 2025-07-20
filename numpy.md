[toc]
# 基本概念
1. 多维数组对象：NumPy的核心数据结构是ndarray，它是一个多维数组，用于存储同质数据类型的元素。这些数组可以是一维、二维、三维等，非常适用于向量化操作和矩阵运算。

2. 广播功能：NumPy允许在不同形状的数组之间执行操作，通过广播功能，它可以自动调整数组的形状，以使操作变得有效。

3. 丰富的数学函数：NumPy提供了大量的数学、统计和线性代数函数，包括基本的加减乘除、三角函数、指数和对数函数、随机数生成、矩阵操作等。

4. 索引和切片：NumPy允许使用索引和切片操作来访问和修改数组中的元素，这使得数据的选择和处理非常灵活。

5. 高性能计算：NumPy的底层实现是用C语言编写的，因此它在处理大规模数据时非常高效。此外，NumPy还与其他高性能计算库（如BLAS和LAPACK）集成，提供了快速的线性代数运算。

6. 互操作性：NumPy可以与许多其他Python库和数据格式（例如Pandas、SciPy、Matplotlib）无缝集成，这使得数据科学工作流更加流畅。
# 创建数组
## 一维数组
```python
import numpy as np
data=np.array([1,2,3,4])
```
## 多维数组
```python
import numpy as np
data = np.array([[1,2,3,4],[1,2,3,4]])
```
## 全0数组
```python
import numpy as np
# shape代表形状，这里创建的就是5行三列的2维数组
data = np.zeros(shape=(5,3))
```

## 全1数组
```python
import numpy as np
#shape代表形状，这里创建5行三列的2维数组
data=np.ones(shape=(5,3))
```

## 全空数组
区别于全0数组，这里生成的是无穷小
```python
import numpy as np
#shape代表维度，这里创建5行三列的2维数组
data = np.empty(shape=(5,3))
```

## 连续序列的数组 arange
```python
import numpy as np
data = np.arange(10,16,2) # 10-16的数据，步长为2
# 从10开始，到16截至（不包含16）
# 输出结果为10，12，14
```

## 有连续间隔的数组（线性等分向量） linspace 
在一个指定区间内按照指定的步长，将区间均等分，生成的是一个线段类型的数组。生成的线性间隔数据中，是有把区间的两端加进去的
> np.linspace(a,b,n)  
> 给定一个范围(a，b),并指定输出向量维数n，那么步长就为$\frac{a-b}{n-1}$
```python
import numpy as np
data = np.linspace(1,10,5)  # 从1开始，到10截至，取等分的5个数（注意是含两端的）
```
## 随机数组
- random.random(size = ())  
生成一个[0,1)的随机数(数组)
- random.rand(a,b)  
生成一个[0,1)的随机数数组
- random.randint(a,b,size = ())  
生成一个[a,b)的随机整数(数组)
``` python
data_rand = np.random.rand(2,3)

data_random = np.random.random((2,3))

data_randint = np.random.randint(3,9,(2,3))
```
## 改变数组形状 reshape
改变一个数组的形状，必须满足其中元素个数是一样的  
reshape本质是原本数组中的元素按顺序展开后，依次填入新定义的尺寸中去.注意 reshape后面填的是元组数据类型
```python
import numpy as np
data1=[1,2,3,4,5]
data2=[1,2,3,4,5]
data=np.array([data1,data2])
data=data.reshape((5,2))
# 此处调用的data自身的方法，也可以使用函数reshape平替
# data = np.reshape(data,(5,2)) 
```
## 方阵转置
对数组对象调用属性 .T 得到他的转置数组
``` python
# 转置前
data_T0 = np.random.randint(0,10,(3,2))
# 转置后
data_T = data_T0.T
```

# 数组显示操作 
## 数组维度 ndim
ndim属性代表数组维度
```python 
data = np.array([[1,2,3],[1,3,4]])
print(data.ndim)
```

## 数组形状shape
shape属性代表数组形状，既各个方向的维度(ndim)
```python
data = np.array([[1,2,3],[1,3,4]])
print(data.ndim)
```

## 数组中元素个数size
size属性代表一个数组中的元素个数
```python
data = np.array([[1,2,3],[1,3,4]])
print(data.size)
```

## 数组中元素的存储类型dtype
dtype属性表示元素的存储类型
```python
data = np.array([[1,2,3],[1,3,4]])
print(data.dtype)
```

# 数组运算
## 数组基础四则运算
对应的各元素相加减乘除
```python
data1 = np.array([[1,1,1],[1,1,1]])
data2 = np.array([[1,-1,1],[-1,1,-1]])
print(data1+data2)
print(data1-data2)
print(data1*data2)
print(data1/data2)
```
## 数组的矩阵乘法 @
```python
data1 = np.array([[1,1,1],[1,1,1]])
data2 = np.array([[1,-1,1],[-1,1,-1]])
print(data1 @ data2.T)
```

# 广播(Broadcast)
当两个形状不同的数组进行运算时，NumPy 会自动扩展（广播）它们的形状，使其在每个维度上匹配，然后进行逐元素操作。
1. 对于两个不同形状数组，判断其是否某一个维度为1
2. 若为一维的，在这个维度上将数组复制扩充成与另一个矩阵形状相同的矩阵  
```python
a = np.array([1, 2, 3])
b = 10
print(a + b)  # [11, 12, 13]
```
```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])
b = np.array([10, 20, 30])
print(a + b)         # [[11 22 33]
                        [14 25 36]]
```
```python
a = np.array([[1], [2], [3]])   
b = np.array([[10, 20, 30, 40]])
print(a+b)   # [[11 21 31 41]
                [12 22 32 42]
                [13 23 33 43]]
```

# 数组中的数据统计

```python
data = np.random.randint(0,3,(3,3))
# 平均值
print(np.mean(data, axis=1, dtype=None, out=None),'\n')

# 中位数
print(np.median(data, axis=None, out=None),'\n')

# 标准差
print(np.std(data, axis=None, dtype=None, out=None),'\n')

# 方差
print(np.var(data, axis=None, dtype=None, out=None),'\n')

# 最大/小值
print(np.max(data, axis=None, out=None))
print(np.min(data, axis=None, out=None))

# 元素整体求和
print(np.sum(data, axis=None, dtype=None, out=None),'\n')

# 元素累计和
print(np.cumsum(data, axis=None, dtype=None, out=None),'\n')

# 元素乘积
print(np.prod(data, axis=None, dtype=None, out=None),'\n\n')
```

# 切片
## 一维数组的切片
切片操作arr[a : b : c]  
a为起始检索位置，b为终止位置（不包含），c为步长（可选）
```python
arr = np.array([1,2,3,4,5])
print(arr[1:4])
```
## 多维数组切片
array[start1:stop1:step1, start2:stop2:step2, ...]  
每一维都用一个 start:stop:step，用逗号 , 分隔开，逐级展开  
1. ,分维度，逐级展开
2. ：表示该维度下全选
3. 符号-表示倒序

# 堆叠
## 竖直堆叠vstack()
```python
stacked_vertically = np.vstack((array1, array2))
```
## 水平堆叠hstack()
```python
stacked_horizontally = np.hstack((array1, array2))
```

# 数据文件npy
## 保存文件
```python
np.save('文件名.npy', data)
```
## 读取文件
```python
data_loaded = np.load('文件名.npy')
```

