import numpy as np


# 输出层函数： 回归问题用恒等函数，分类问题用softmax函数
a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)  # 指数函数
print(exp_a)
sum_exp_a = np.sum(exp_a)  # 指数函数的和
print(sum_exp_a)
y = exp_a / sum_exp_a
print(y)


def softmax_(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


a = np.array([1010, 1000, 990])
x = np.exp(a) / np.sum(np.exp(a))  # softmax函数的运算
print(x)  # 没有被正确计算 TODO error
c = np.max(a)  # 1010
print(a - c)
print(np.exp(a - c) / np.sum(np.exp(a - c)))


# 改进版
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


# 使用softmax
a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
print(np.sum(y))
