# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# 创建数据
x = np.arange(0, 6, 0.1)  # 从0到6以0.1刻度生成
y = np.sin(x)

# グラフの描画
plt.plot(x, y)
plt.show()
