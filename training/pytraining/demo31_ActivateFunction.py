import numpy as np
import matplotlib.pyplot as plt


def step_function(x):
    if x > 0:
        return 1
    else:
        return 0


# 传入的x是np.array
def step_function(x):
    y = x > 0
    return y.astype(np.int)


print(step_function(np.array([-1, 2])))


def step_function(x):
    return np.array(x > 0, dtype=np.int)


x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # 指定y轴的范围
plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x))


x = np.arange(-5.0, 5.0, 0.1)
y1 = step_function(x)
y2 = sigmoid(x)
plt.plot(x, y1, label='step')
plt.plot(x, y2, label='sigmoid', linestyle='-.')
plt.legend()
plt.ylim(-0.1, 1.1)  # 指定y轴的范围
plt.show()


def relu(x):
    return np.maximum(0, x)
