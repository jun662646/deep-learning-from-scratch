import numpy as np

A = np.array([1, 2, 3, 4])
print(A)

print(np.ndim(A))

print(A.shape)

print(A.shape[0])


B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)

print(np.ndim(B))

print(B.shape)

print(B.shape[1])


# 矩阵乘法 / 点积
A = np.array([[1,2], [3,4]])

print(A.shape)

B = np.array([[5,6], [7,8]])
print(B.shape)

print(np.dot(A, B))

# 点积的结果：A的行 x B的列
A = np.array([[1,2,3], [4,5,6]])
print(A.shape)

B = np.array([[1,2], [3,4], [5,6]])
print(B.shape)

print(np.dot(A, B).shape)

# A的列数必须等于B的行数
A = np.array([[1,2], [3, 4], [5,6]])
print(A.shape)

B = np.array([7,8])
print(B.shape)

print(np.dot(A, B))
