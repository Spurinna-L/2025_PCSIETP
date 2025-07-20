import numpy as np
# 一维矩阵
date1 = np.array([1,2,3])
print(date1)
print("\n\n")
# 二维矩阵
date2 = np.array([[1,2,3],[3,4,5]])
print(date2)
print("\n\n")
# 全零矩阵
date_0 = np.zeros(shape=(2,3))
print(date_0)
print("\n\n")
# 全1矩阵
date_1 = np.ones(shape=(3,4))
print(date_1)
print("\n\n")
# 全空矩阵
data_empty= np.empty(shape=(4,6))
print(data_empty)
print("\n\n")
# 连续序列的矩阵 arange
data_arange = np.arange(1,10,1)
print(data_arange)
print("\n\n")
# 有连续间隔的数组 linspace 
data_linspace = np.linspace(1,10,20)
print(data_linspace)
print("\n\n")
# 随机矩阵
data_rand = np.random.rand(2,3)
print(data_rand)
print("\n")
data_random = np.random.random((2,3))
print(data_random)
print("\n")
data_randint = np.random.randint(3,9,(2,3))
print(data_randint)
print("\n\n")
# 矩阵变形
data_reshape = np.reshape(data_randint,(1,6))
print(data_reshape)
print("\n\n")
# 矩阵转置
print('转置前')
data_T = np.random.randint(0,10,(3,2))
print(data_T)
print('转置后')
print(data_T.T)
print("\n\n")

# 数组维数.ndim
data = np.array([[1,2,3],[1,3,4]])
print(data.ndim)
print("\n\n")

# 矩阵形状
print(data.shape)
print('\n\n')
# 矩阵元素个数
print(data.size)
print('\n\n')
# 数组的数据类型 .dtype
print(data.dtype)
print('\n\n')
# 数组逐项加、减、乘、除法
data1 = np.array([[1,1,1],[1,1,1]])
data2 = np.array([[1,-1,1],[-1,1,-1]])

print(data1+data2)
print(data1-data2)
print(data1*data2)
print(data1/data2)
print('\n')
# 数组矩阵乘法
print(data1@data2.T)
print('\n\n')

# 广播机制
data3 = np.array([1,1,1])
print(data2+data3)
print('\n\n')

# 平均值
print(np.mean(data2, axis=1, dtype=None, out=None),'\n')

# 中位数
print(np.median(data2, axis=None, out=None),'\n')

# 标准差
print(np.std(data2, axis=None, dtype=None, out=None),'\n')

# 方差
print(np.var(data2, axis=None, dtype=None, out=None),'\n')

# 最大/小值
print(np.max(data2, axis=None, out=None))
print(np.min(data2, axis=None, out=None))

# 元素整体求和
print(np.sum(data2, axis=None, dtype=None, out=None),'\n')

# 元素累计和
print(np.cumsum(data2, axis=None, dtype=None, out=None),'\n')

# 元素乘积
print(np.prod(data2, axis=None, dtype=None, out=None),'\n\n')

# 一维切片
arr = np.array([1,2,3,4,5])
print(arr[1:4])
print("\n")
# 多维切片
data1=[1,2,3,4,5]
data2=[6,7,8,9,10]
data3=[11,12,13,14,15]
data4=[16,17,18,19,20]
data5=[21,22,23,24,25]
data6=[26,27,28,29,30]
data=np.array([[data1,data2,data3],[data4,data5,data6]])
print(data,'\t',data.shape)
# 行切片
print(data[0:2,1:4,:])  # 每一维度的前两行
# 列切片
print(data[0:1,:,1])    # 第一个维度的第二列
print('\n\n')

# 竖直堆叠
data_vertical = np.vstack((data1, data2))
print(data_vertical,'\n')
# 水平堆叠 
data_horizon = np.hstack([data1,data2])
print(data_horizon,'\n\n')