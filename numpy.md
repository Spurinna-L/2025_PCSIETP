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
## 全0矩阵
```python
import numpy as np
# shape代表形状，这里创建的就是5行三列的2维数组
data = np.zeros(shape=(5,3))
```

## 全1矩阵
```python
import numpy as np
#shape代表形状，这里创建5行三列的2维数组
data=np.ones(shape=(5,3))
```

## 全空矩阵
区别于全0矩阵，这里生成的是无穷小
```python
import numpy as np
#shape代表维度，这里创建5行三列的2维数组
data = np.empty(shape=(5,3))
```

## 连续序列的矩阵 arange
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
## 随机矩阵
- random.random(size = ())  
生成一个[0,1)的随机数(矩阵)
- random.rand(a,b)  
生成一个[0,1)的随机数矩阵
- random.randint(a,b,size = ())  
生成一个[a,b)的随机整数(矩阵)
``` python
data_rand = np.random.rand(2,3)

data_random = np.random.random((2,3))

data_randint = np.random.randint(3,9,(2,3))
```
## 改变矩阵形状 reshape
改变一个矩阵的形状，必须满足其中元素个数是一样的  
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
对矩阵对象调用方法 .T 得到他的转置矩阵
``` python
# 转置前
data_T0 = np.random.randint(0,10,(3,2))
# 转置后
data_T = data_T0.T
```

