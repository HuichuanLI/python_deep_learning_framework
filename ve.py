# %%time
import pickle, gzip, urllib.request, json
import numpy as np
import os.path
import util
import train
from core.Layers import *
from NeuralNetwork import NeuralNetwork, train_nn

if not os.path.isfile("mnist.pkl.gz"):
    # Load the dataset
    urllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")

with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

train_X, train_y = train_set
valid_X, valid_y = valid_set
print(train_X.dtype)
print(train_set[0].shape)
print(valid_X.shape)

import matplotlib.pyplot as plt

digit = train_set[0][9].reshape(28, 28)

plt.imshow(digit, cmap='gray')
plt.colorbar()
plt.show()


class Layer:
    def __init__(self):
        self.params = None
        pass

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x, grad):
        raise NotImplementedError

    def reg_grad(self, reg):
        pass

    def reg_loss(self, reg):
        return 0.

    # ----------加权和计算------------


nn = NeuralNetwork()
nn.add_layer(Dense(784, 32))
nn.add_layer(Relu())
nn.add_layer(Dense(32, 784))
nn.add_layer(Sigmoid())

learning_rate = 1e-2
momentum = 0.9

optimer = train.Adam(nn.parameters(), learning_rate, 0.5)
reg = 1e-3
loss_fn = util.mse_loss_grad
X = train_X
epochs = 1
print_n = 1500
batch_size = 128

losses = train_nn(nn, X, X, optimer, loss_fn, batch_size=batch_size, reg=reg, print_n=print_n)
# # nn.train_batch(train_X,train_y,ds.data_iter,loss_gradient_softmax_crossentropy,25,0.1,32,True,1e-3,2)
# print(np.mean(nn.predict(train_X) == train_y))
# print(np.mean(nn.predict(valid_X) == valid_y))
# print(nn.predict(valid_X[9]), valid_y[9])

import matplotlib.pyplot as plt

plt.plot(losses)


def draw_predict_mnists(plt, X, indices):
    for i, index in enumerate(indices):
        aimg = train_X[index]
        aimg = aimg.reshape(1, -1)
        aimg_out = nn(aimg)
        plt.subplot(2, 10, i + 1)
        plt.imshow(aimg.reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.subplot(2, 10, i + 11)
        plt.imshow(aimg_out.reshape(28, 28), cmap='gray')
        plt.axis('off')


draw_predict_mnists(plt, train_X, range(10))
plt.show()