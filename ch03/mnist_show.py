# coding: utf-8
import numpy as np
from dataset.mnist import load_mnist
# Python Image Library
from PIL import Image
import sys, os

sys.path.append(os.pardir)  # 用于导入父目录文件的设置


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# “(训练图像 ,训练标签 )，(测试图像，测试标签 )”
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=True)
print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000,)
print(x_test.shape)  # (10000, 784)
print(t_test.shape)  # (10000,)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 将形状变形为原始图像尺寸
print(img.shape)  # (28, 28)

img_show(img)
