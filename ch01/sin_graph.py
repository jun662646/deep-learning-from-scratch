# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# 创建数据
x = np.arange(0, 6, 0.1)
y = np.sin(x)

# 绘制图表
plt.plot(x, y)
plt.show()
