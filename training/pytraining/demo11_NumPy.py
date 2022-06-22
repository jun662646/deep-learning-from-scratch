import numpy as np


class Man:
    def __init__(self, name):
        self.name = name

    print("Initialized!")

    def hello(self):
        print("Hello " + self.name + "!")

    def goodbye(self):
        print("Good-bye " + self.name + "!")


m = Man("haha")
m.hello()
m.goodbye()
print(type(m))


x = [1, 2, 3, 4]
print(x)
x = np.array(x)
print(x)
print(type(x))


a = np.array(x)
b = np.array(x)
print(a+b)
print(a-b)
print(a*b)
print(a/b)
print(a+2)


x = [[1, 2, 1], [3, 4, 1]]
a = np.array(x)
print(x)
print(a)
print(a.shape)
print(a.dtype)


b = np.array([[1, 2, 3], [2, 2, 1]])
print(a+b)
print(a*b)


x = a.flatten()
print(x)
print(x[[0, 1]])
print(x[x <= 2])
print(x[[True, False, True, False, False, False]])
print('----------------------------\n')


def f(a, b, c, d):      # * 在函数定义中使用
    print(a, b, c, d)


a = {"a": 1, "b": 2, "c": 3, "d": 4}
f(*a)
f(**a)


w = np.random.randn(1,2,2,2)
ww = w.reshape(1,-1)
print(w)
print(ww)
