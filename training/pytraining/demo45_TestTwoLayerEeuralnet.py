import numpy as np
from ch04.two_layer_net import TwoLayerNet


net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
print(net.params['W1'].shape)  # (784, 100)
print(net.params['b1'].shape)  # (100,)
print(net.params['W2'].shape)  # (100, 10)
print(net.params['b2'].shape)  # (10,)

x = np.random.rand(100, 784)  # 伪输入数据（100笔）
y = net.predict(x)

x = np.random.rand(100, 784)  # 伪输入数据（100笔）
t = np.random.rand(100, 10)  # 伪正确解标签（100笔）
grads = net.numerical_gradient(x, t)  # 计算梯度
print(grads['W1'].shape)  # (784, 100)
print(grads['b1'].shape)  # (100,)
print(grads['W2'].shape)  # (100, 10)
print(grads['b2'].shape)  # (10,)
