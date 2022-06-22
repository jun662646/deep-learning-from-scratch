import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

x = np.arange(0, 6, 0.1)
y = np.sin(x)
print(x)
print(y)
plt.plot(x, y)
plt.show()


y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x, y1, label='sin')
plt.plot(x, y2, label='cos', linestyle=':')
plt.xlabel('x')
plt.ylabel('y')
plt.title('sin-cos')
plt.legend()
plt.show()


img = imread('img/img.png')  # 读入图像（设定合适的路径！）
plt.imshow(img)
plt.show()
