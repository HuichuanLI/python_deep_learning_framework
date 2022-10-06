# %%time
import pickle, gzip, urllib.request, json
import numpy as np
import os.path
import util
import train
from core.Layers import *
from NeuralNetwork import NeuralNetwork, train_nn
from core.train import *
from util import *
import time

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


input_dim = 784
hidden = 256
nz = 2
encoder = NeuralNetwork()
encoder.add_layer(Dense(784, hidden))
encoder.add_layer(Relu())
encoder.add_layer(Dense(hidden, hidden))
encoder.add_layer(Relu())
encoder.add_layer(Dense(hidden, 2 * nz))

decoder = NeuralNetwork()
decoder.add_layer(Dense(nz, hidden))
decoder.add_layer(Relu())
decoder.add_layer(Dense(hidden, hidden))
decoder.add_layer(Relu())
decoder.add_layer(Dense(hidden, input_dim))
decoder.add_layer(Sigmoid())


class VAE:
    def __init__(self, encoder, decoder, e_optimezer, d_optimzer):
        self.encoder, self.decoder = encoder, decoder
        self.e_optimezer, self.d_optimzer = e_optimezer, d_optimzer

    def encode(self, x):
        e_out = self.encoder(x)
        mu, logvar = np.split(e_out, 2, axis=1)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        self.rand_sample = np.random.standard_normal(size=(mu.shape[0], mu.shape[1]))
        self.sample_z = mu + np.exp(logvar * .5) * self.rand_sample

        d_out = self.decoder(self.sample_z)
        return d_out, mu, logvar

    def ___call__(self, x):
        return self.forward(x)

    def backward(self, x, loss_fn=BCE_loss_grad):
        out, mu, logvar = self.forward(x)
        loss, loss_grad = loss_fn(out, x)
        dz = decoder.backward(loss_grad)
        du = dz
        dE = dz * np.exp(logvar * .5) * .5 * self.rand_sample

        duE = np.hstack([du, dE])

        # KL_Loss
        kl_loss = 00.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar))
        loss += kl_loss / len(out)

        kl_du = mu
        kl_dE = -0.5 * (1 - np.exp(logvar))
        kl_duE = np.hstack([kl_du, kl_dE])
        kl_duE /= len(out)

        encoder.backward(duE + kl_duE)

        return loss

    def tran_VAE_epoch(self, dataset, loss_fn=BCE_loss_grad, print_fn=None):
        iter = 0
        losses = []
        for x in dataset:
            self.e_optimezer.zero_grad()
            self.d_optimzer.zero_grad()

            loss = self.backward(x, loss_fn)
            self.e_optimezer.step()
            self.d_optimzer.step()
            losses.append(loss)
            if print_fn:
                print_fn(losses)
            iter += 1
        return losses


lr = 0.001
beta_1, beta_2 = 0.9, 0.999
e_optimizer = Adam(encoder.parameters(), lr, beta_1, beta_2)
d_optimizer = Adam(decoder.parameters(), lr, beta_1, beta_2)

loss_fn = mse_loss_grad
iterations = 10000
batch_size = 64

vae = VAE(encoder, decoder, e_optimizer, d_optimizer)

start = time.time()
epochs = 30
print_n = 1
epoch_losses = []

for epoch in range(epochs):
    data_it = data_iterator_X(train_X, batch_size=batch_size)
    epoch_loss = vae.tran_VAE_epoch(data_it, loss_fn)
    epoch_loss = np.array(epoch_loss).mean()

    if epochs % print_n == 0:
        print('Epoch {}, training loss{:.2f}:'.format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)
end = time.time()
print('time elasped:{:.2f}s'.format(end - start))

# learning_rate = 1e-2
# momentum = 0.9
#
# optimer = train.Adam(nn.parameters(), learning_rate, 0.5)
# reg = 1e-3
# loss_fn = util.mse_loss_grad
# X = train_X
# epochs = 1
# print_n = 1500
# batch_size = 128
#
# losses = train_nn(nn, X, X, optimer, loss_fn, batch_size=batch_size, reg=reg, print_n=print_n)
# # # nn.train_batch(train_X,train_y,ds.data_iter,loss_gradient_softmax_crossentropy,25,0.1,32,True,1e-3,2)
# # print(np.mean(nn.predict(train_X) == train_y))
# # print(np.mean(nn.predict(valid_X) == valid_y))
# # print(nn.predict(valid_X[9]), valid_y[9])
#
import matplotlib.pyplot as plt

plt.plot(epoch_losses)


def draw_predict_mnists(plt, vae, x, n_sample=10):
    np.random.seed(1)
    idx = np.random.choice(len(x), n_sample)
    _, axarr = plt.subplots(2, n_sample, figsize=(16, 4))
    for i, j in enumerate(idx):
        axarr[0, i].imshow(x[j].reshape(28, 28), cmap='Greys')
        if i == 0:
            axarr[0, i].set_title('original')
        out, _, _ = vae(x[j].reshape(1, -1))
        axarr[1, i].imshow(out.reshape(28, 28), cmap='Greys')
        if i == 0:
            axarr[1, i].set_title('recon')


draw_predict_mnists(plt, vae, valid_X, 10)
plt.show()
#
# draw_predict_mnists(plt, train_X, range(10))
# plt.show()
