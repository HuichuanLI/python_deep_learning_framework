import numpy as np
from util import *
from NeuralNetwork import *
from train import *
import mnist_reader
import matplotlib.pyplot as plt

np.random.seed(1)

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
trainX = X_train.reshape(-1, 28, 28)
train_X = trainX.astype('float32') / 255.0

nn = NeuralNetwork()

nn.add_layer(Dense(784, 500))
nn.add_layer(Relu())

nn.add_layer(Dense(500, 200))
nn.add_layer(BatchNorm_1d(200))
nn.add_layer(Relu())

nn.add_layer(Dense(200, 100))
nn.add_layer(BatchNorm_1d(100))
nn.add_layer(Relu())

nn.add_layer(Dense(100, 10))

learning_rate = 0.01
momentum = 0.9
optimizer = SGD(nn.parameters(), learning_rate, momentum)

epochs = 8
batch_size = 64
reg = 0  # 1e-3
print_n = 1000

losses = train_nn(nn, train_X, y_train, optimizer, cross_entropy_grad_loss, epochs, batch_size, reg, print_n)

plt.plot(losses)

print(np.mean(nn.predict(train_X) == y_train))
test_X = X_test.reshape(-1, 28, 28).astype('float32') / 255.0
print(np.mean(nn.predict(test_X) == y_test))

from Layers import *


class Dropout(Layer):
    def __init__(self, drop_p):
        super().__init__()
        self.retain_p = 1 - drop_p

    def forward(self, x, training=True):
        retain_p = self.retain_p
        if training:
            self._mask = (np.random.rand(*x.shape) < retain_p) / retain_p
            out = x * self._mask
        else:
            out = x
        return out

    def backward(self, dx_output, training=True):
        dx = None
        if training:
            dx = dx_output * self._mask
        else:
            dx = dx_output
        return dx


np.random.seed(1)
dropout = Dropout(0.3)
X = np.random.rand(2, 4)
print(X)
print(dropout.forward(X))
dx_output = np.random.rand(2, 4)
print(dx_output)
print(dropout.backward(dx_output))
