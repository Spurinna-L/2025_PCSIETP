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
