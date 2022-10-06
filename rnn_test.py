import numpy as np

np.random.seed(1)


def rnn_params_init(input_dim, hidden_dim, output_dim, scale=0.01):
    Wx = np.random.randn(input_dim, hidden_dim) * scale  # input to hidden
    Wh = np.random.randn(hidden_dim, hidden_dim) * scale  # hidden to hidden
    bh = np.zeros((1, hidden_dim))  # hidden bias

    Wf = np.random.randn(hidden_dim, output_dim) * scale  # hidden to output
    bf = np.zeros((1, output_dim))  # output bias

    return [Wx, Wh, bh, Wf, bf]


def rnn_hidden_state_init(batch_dim, hidden_dim):
    return np.zeros((batch_dim, hidden_dim))


def rnn_forward(params, Xs, H_):
    Wx, Wh, bh, Wf, bf = params
    H = H_  # np.copy(H_)

    Fs = []
    Hs = {}
    Hs[-1] = np.copy(H)

    for t in range(len(Xs)):
        X = Xs[t]
        H = np.tanh(np.dot(X, Wx) + np.dot(H, Wh) + bh)
        F = np.dot(H, Wf) + bf

        Fs.append(F)
        Hs[t] = H
    return Fs, Hs


def rnn_forward_step(params, X, preH):
    Wx, Wh, bh, Wf, bf = params
    H = np.tanh(np.dot(X, Wx) + np.dot(preH, Wh) + bh)
    F = np.dot(H, Wf) + bf
    return F, H


def rnn_forward_(params, Xs, H_):
    Wx, Wh, bh, Wf, bf = params
    H = H_

    Fs = []
    Hs = {}
    Hs[-1] = np.copy(H)

    for t in range(len(Xs)):
        X = Xs[t]
        F, H = rnn_forward_step(params, X, H)
        Fs.append(F)
        Hs[t] = H
    return Fs, Hs


import util


def rnn_loss_grad(Fs, Ys, loss_fn=util.cross_entropy_grad_loss, flatten=True):
    loss = 0
    dFs = {}

    for t in range(len(Fs)):
        F = Fs[t]
        Y = Ys[t]
        if flatten and Y.ndim >= 2:
            Y = Y.flatten()
        loss_t, dF_t = loss_fn(F, Y)
        loss += loss_t
        dFs[t] = dF_t

    return loss, dFs


import math


def grad_clipping(grads, alpha):
    norm = math.sqrt(sum((grad ** 2).sum() for grad in grads))
    if norm > alpha:
        ratio = alpha / norm
        for i in range(len(grads)):
            grads[i] *= ratio


def rnn_backward(params, Xs, Hs, dZs, clip_value=5.):  # Ys,loss_function):
    Wx, Wh, bh, Wf, bf = params
    dWx, dWh, dWf = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(Wf)
    dbh, dbf = np.zeros_like(bh), np.zeros_like(bf)

    dh_next = np.zeros_like(Hs[0])
    h = Hs
    x = Xs

    T = len(Xs)  # 序列长度（时刻长度）
    for t in reversed(range(T)):
        dZ = dZs[t]

        dWf += np.dot(h[t].T, dZ)

        dbf += np.sum(dZ, axis=0, keepdims=True)
        dh = np.dot(dZ, Wf.T) + dh_next
        dZh = (1 - h[t] * h[t]) * dh

        dbh += np.sum(dZh, axis=0, keepdims=True)
        dWx += np.dot(x[t].T, dZh)
        dWh += np.dot(h[t - 1].T, dZh)
        dh_next = np.dot(dZh, Wh.T)

    grads = [dWx, dWh, dbh, dWf, dbf]
    if clip_value is not None:
        grad_clipping(grads, clip_value)
    return grads


def rnn_backward_step(params, dZ, X, H, H_, dh_next):
    Wx, Wh, bh, Wf, bf = params
    dWf = np.dot(H.T, dZ)

    dbf = np.sum(dZ, axis=0, keepdims=True)
    dh = np.dot(dZ, Wf.T) + dh_next
    dZh = (1 - H * H) * dh

    dbh = np.sum(dZh, axis=0, keepdims=True)
    dWx = np.dot(X.T, dZh)
    dWh = np.dot(H_.T, dZh)
    dh_next = np.dot(dZh, Wh.T)
    return dWx, dWh, dbh, dWf, dbf, dh_next


def rnn_backward_(params, Xs, Hs, dZs, clip_value=5.):
    Wx, Wh, bh, Wf, bf = params
    dWx, dWh, dWf = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(Wf)
    dbh, dbf = np.zeros_like(bh), np.zeros_like(bf)
    dh_next = np.zeros_like(Hs[0])

    T = len(Xs)  # 序列长度（时刻长度）
    for t in reversed(range(T)):
        dZ = dZs[t]
        H = Hs[t]
        H_ = Hs[t - 1]
        X = Xs[t]

        dWx_, dWh_, dbh_, dWf_, dbf_, dh_next = rnn_backward_step(params, dZ, X, H, H_, dh_next)
        for grad, grad_t in zip([dWx, dWh, dbh, dWf, dbf], [dWx_, dWh_, dbh_, dWf_, dbf_]):
            grad += grad_t

    grads = [dWx, dWh, dbh, dWf, dbf]
    if clip_value is not None:
        grad_clipping(grads, clip_value)
    return grads


import numpy as np

np.random.seed(1)

# 生成4个时刻，每批有2个样本的一批样本Xs及目标
# 定义一个输入、隐含层、输出层的大小分别是4、10、4的RNN模型
if True:
    T = 5
    input_dim, hidden_dim, output_dim = 4, 10, 4
    batch_size = 1
    seq_len = 5
    Xs = np.random.rand(seq_len, batch_size, input_dim)
    # Ys = np.random.randint(input_dim,size = (seq_len,batch_size,output_dim))
    Ys = np.random.randint(input_dim, size=(seq_len, batch_size))

print(Xs)
print(Ys)

# --------cheack gradient-------------
params = rnn_params_init(input_dim, hidden_dim, output_dim)
H_0 = rnn_hidden_state_init(batch_size, hidden_dim)

Fs, Hs = rnn_forward(params, Xs, H_0)
loss_function = rnn_loss_grad
print(Fs[0].shape, Ys[0].shape)
loss, dFs = loss_function(Fs, Ys)
grads = rnn_backward(params, Xs, Hs, dFs)


def rnn_loss():
    H_0 = np.zeros((1, hidden_dim))
    H = np.copy(H_0)
    Fs, Hs = rnn_forward(params, Xs, H)
    loss_function = rnn_loss_grad
    loss, dFs = loss_function(Fs, Ys)
    return loss


numerical_grads = util.numerical_gradient(rnn_loss, params, 1e-6)  # rnn_numerical_gradient(rnn_loss,params,1e-10)
# diff_error = lambda x, y: np.max(np.abs(x - y))
diff_error = lambda x, y: np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

print("loss", loss)
print("[dWx, dWh, dbh,dWf, dbf]")
for i in range(len(grads)):
    print(diff_error(grads[i], numerical_grads[i]))

print("grads", grads[1][:2])
print("numerical_grads", numerical_grads[1][:2])


class SGD():
    def __init__(self, model_params, learning_rate=0.01, momentum=0.9):
        self.params, self.lr, self.momentum = model_params, learning_rate, momentum
        self.vs = []
        for p in self.params:
            v = np.zeros_like(p)
            self.vs.append(v)

    def step(self, grads):
        for i in range(len(self.params)):
            grad = grads[i]
            self.vs[i] = self.momentum * self.vs[i] + self.lr * grad
            self.params[i] -= self.vs[i]

    def scale_learning_rate(self, scale):
        self.lr *= scale


T = 5000  # Generate a total of 1000 points
time = np.arange(0, T)
data = np.sin(time * 0.1) + np.cos(time * 0.2)
print(data.shape)

import numpy as np


def rnn_data_iter_consecutive(data, batch_size, seq_len, start_range=10, to_3D=True):
    # 每次在data[offset:]里采样，使得每一个epoch的训练样本不同
    start = np.random.randint(0, start_range)
    block_len = (len(data) - start - 1) // batch_size

    Xs = data[start:start + block_len * batch_size]
    Ys = data[start + 1:start + block_len * batch_size + 1]
    Xs = Xs.reshape(batch_size, -1)
    Ys = Ys.reshape(batch_size, -1)

    # 在每个块里可以i采样多少个长度为seq_len的样本序列
    reset = True
    num_batches = Xs.shape[1] // seq_len
    for i in range(0, num_batches * seq_len, seq_len):
        X = Xs[:, i:(i + seq_len)]
        Y = Ys[:, i:(i + seq_len)]
        if to_3D:
            X = np.swapaxes(X, 0, 1)
            X = X.reshape(X.shape[0], X.shape[1], -1)
            # X = np.expand_dims(X, axis=2)
            Y = np.swapaxes(Y, 0, 1)
            Y = Y.reshape(Y.shape[0], Y.shape[1], -1)
        else:
            X = np.swapaxes(X, 0, 1)
            Y = np.swapaxes(Y, 0, 1)
        if reset:
            reset = False
            yield X, Y, True
        else:
            yield X, Y, False


def rnn_train_epoch(params, data_iter, optimizer, iterations, loss_function, print_n=100):
    Wx, Wh, bh, Wf, bf = params
    losses = []
    iter = 0

    hidden_size = Wh.shape[0]

    for Xs, Ys, start in data_iter:

        batch_size = Xs[0].shape[0]
        if start:
            H = rnn_hidden_state_init(batch_size, hidden_size)

        Zs, Hs = rnn_forward(params, Xs, H)
        loss, dzs = loss_function(Zs, Ys)

        if False:
            print("Z.shape", Zs[0].shape)
            print("Y.shape", Ys[0].shape)
            print("H", H.shape)

        dWx, dWh, dbh, dWf, dbf = rnn_backward(params, Xs, Hs, dzs)

        H = Hs[len(Hs) - 2]  # 最后时刻的隐状态向量

        grads = [dWx, dWh, dbh, dWf, dbf]
        optimizer.step(grads)
        losses.append(loss)

        if iter % print_n == 0:
            print('iter %d, loss: %f' % (iter, loss))
        iter += 1

        if iter > iterations: break
    return losses, H


batch_size = 3
input_dim = 1
output_dim = 1
hidden_size = 100
seq_length = 50
params = rnn_params_init(input_dim, hidden_size, output_dim)
H = rnn_hidden_state_init(batch_size, hidden_size)

data_it = rnn_data_iter_consecutive(data, batch_size, seq_length, 2)
x, y, _ = next(data_it)
print("X:", x.shape, "Y:", y.shape, "H:", H.shape)

loss_function = lambda F, Y: rnn_loss_grad(F, Y, util.mse_loss_grad, False)

Zs, Hs = rnn_forward(params, x, H)
print("Z:", Zs[0].shape, "H:", Hs[0].shape)
loss, dzs = loss_function(Zs, y)
print(dzs[0].shape)

epoches = 10
learning_rate = 5e-4

iterations = 200
losses = []

# optimizer = AdaGrad(params,learning_rate)
momentum = 0.9
optimizer = SGD(params, learning_rate, momentum)

for epoch in range(epoches):
    data_it = rnn_data_iter_consecutive(data, batch_size, seq_length, 100)
    # epoch_losses,param,H = rnn_train(params,data_it,learning_rate,iterations,loss_function,print_n=100)
    epoch_losses, H = rnn_train_epoch(params, data_it, optimizer, iterations, loss_function, print_n=50)
    # losses.extend(epoch_losses)
    epoch_losses = np.array(epoch_losses).mean()
    losses.append(epoch_losses)

filename = 'input.txt'
data = open(filename, 'r').read()
chars = list(set(data))

data_size, vocab_size = len(data), len(chars)
print('总字符个数 %d,字符表的长度 %d unique.' % (data_size, vocab_size))
print('字符表的前10个字符：\n', chars[:10])
print('前148个字符：\n', data[:148])

char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}


def one_hot_idx(idx, vocab_size):
    x = np.zeros((1, vocab_size))
    x[0, idx] = 1
    return x


import numpy as np


def character_seq_data_iter_consecutive(data, batch_size, seq_len, start_range=10):
    # 每次在data[offset:]里采样，使得每一个epoch的训练样本不同
    start = np.random.randint(0, start_range)
    block_len = (len(data) - start - 1) // batch_size
    num_batches = block_len // seq_len  # 每块里最多能连续采样的批数
    bs = np.array(range(0, block_len * batch_size, block_len))  # 每个block起始位置

    i_end = num_batches * seq_len
    for i in range(0, i_end, seq_len):  # 一个block的序列开始位置
        s = start + i  # 在一个block里的位置
        X = np.empty((seq_len, batch_size), dtype=object)  # ,dtype = np.int32)
        Y = np.empty((seq_len, batch_size), dtype=object)  # ,dtype = np.int32)
        for b in range(batch_size):  # b表示一个批样本的第几个样本
            s_b = s + bs[b]
            for t in range(seq_len):
                X[t, b] = data[s_b]
                Y[t, b] = data[s_b + 1]
                s_b += 1
        if i == 0:
            yield X, Y, True
        else:
            yield X, Y, False


import numpy as np
import math


class RNNCellBase(object):
    __constants__ = ['input_size', 'hidden_size']

    def __init__(self, input_size, hidden_size, bias, num_chunks):
        super(RNNCellBase, self).__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.bias = bias
        self.W_ih = np.empty((input_size, num_chunks * hidden_size))  # input to hidden
        self.W_hh = np.empty((hidden_size, num_chunks * hidden_size))  # hidden to hidden
        if bias:
            self.b_ih = np.zeros((1, num_chunks * hidden_size))
            self.b_hh = np.zeros((1, num_chunks * hidden_size))
            self.params = [self.W_ih, self.W_hh, self.b_ih, self.b_hh]
        else:
            self.b_ih = None
            self.b_hh = None
            self.params = [self.W_ih, self.W_hh]

        self.grads = [np.zeros_like(param) for param in self.params]
        self.param_grads = self.params.copy()
        self.param_grads.extend(self.grads)

        self.reset_parameters()

    def parameters(self, no_grad=True):
        if no_grad:   return self.params;
        return self.param_grads;

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for param in self.params:
            w = param
            w[:] = np.random.uniform(-stdv, stdv, (w.shape))

    def check_forward_input(self, input):
        if input.shape[1] != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.shape[1], self.input_size))

    def check_forward_hidden(self, input, h, hidden_label=''):
        if input.shape[0] != h.shape[0]:
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.shape[0], hidden_label, h.shape[0]))

        if h.shape[1] != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, h.shape[1], self.hidden_size))


def relu(x):
    return x * (x > 0)


def rnn_tanh_cell(x, h, W_ih, W_hh, b_ih, b_hh):
    # h' = \tanh(W_{ih} x + b_{ih}  +  W_{hh} h + b_{hh})
    if b_ih is None:
        return np.tanh(np.dot(x, W_ih) + np.dot(h, W_hh))
    else:
        return np.tanh(np.dot(x, W_ih) + b_ih + np.dot(h, W_hh) + b_hh)


def rnn_relu_cell(x, h, W_ih, W_hh, b_ih, b_hh):
    # h' = \relu(W_{ih} x + b_{ih}  +  W_{hh} h + b_{hh})
    if b_ih is None:
        return relu(np.dot(x, W_ih) + np.dot(h, W_hh))
    else:
        return relu(np.dot(x, W_ih) + b_ih + np.dot(h, W_hh) + b_hh)


import numpy as np
import math


class RNNCellBase(object):
    __constants__ = ['input_size', 'hidden_size']

    def __init__(self, input_size, hidden_size, bias, num_chunks):
        super(RNNCellBase, self).__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.bias = bias
        self.W_ih = np.empty((input_size, num_chunks * hidden_size))  # input to hidden
        self.W_hh = np.empty((hidden_size, num_chunks * hidden_size))  # hidden to hidden
        if bias:
            self.b_ih = np.zeros((1, num_chunks * hidden_size))
            self.b_hh = np.zeros((1, num_chunks * hidden_size))
            self.params = [self.W_ih, self.W_hh, self.b_ih, self.b_hh]
        else:
            self.b_ih = None
            self.b_hh = None
            self.params = [self.W_ih, self.W_hh]

        self.grads = [np.zeros_like(param) for param in self.params]
        self.param_grads = self.params.copy()
        self.param_grads.extend(self.grads)

        self.reset_parameters()

    def parameters(self, no_grad=True):
        if no_grad:   return self.params;
        return self.param_grads;

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for param in self.params:
            w = param
            w[:] = np.random.uniform(-stdv, stdv, (w.shape))

    def check_forward_input(self, input):
        if input.shape[1] != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.shape[1], self.input_size))

    def check_forward_hidden(self, input, h, hidden_label=''):
        if input.shape[0] != h.shape[0]:
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.shape[0], hidden_label, h.shape[0]))

        if h.shape[1] != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, h.shape[1], self.hidden_size))




class RNNCell(RNNCellBase):
    """        h' = \tanh(W_{ih} x + b_{ih}  +  W_{hh} h + b_{hh})"""
    __constants__ = ['input_size', 'hidden_size', 'nonlinearity']

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        super(RNNCell, self).__init__(input_size, hidden_size, bias, num_chunks=1)
        self.nonlinearity = nonlinearity

    def forward(self, input, h=None):
        self.check_forward_input(input)
        if h is None:
            h = np.zeros(input.shape[0], self.hidden_size, dtype=input.dtype)
        self.check_forward_hidden(input, h, '')
        if self.nonlinearity == "tanh":
            ret = rnn_tanh_cell(input, h,
                                self.W_ih, self.W_hh,
                                self.b_ih, self.b_hh, )
        elif self.nonlinearity == "relu":
            ret = rnn_relu_cell(input, h,
                                self.W_ih, self.W_hh,
                                self.b_ih, self.b_hh, )
        else:
            ret = input
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))
        return ret

    def __call__(self, input, h=None):
        return self.forward(input, h)

    def backward(self, dh, H, X, H_pre):
        if self.nonlinearity == "tanh":
            dZh = (1 - H * H) * dh  # backprop through tanh nonlinearity
        else:
            dZh = H * (1 - H) * dh
        db_hh = np.sum(dZh, axis=0, keepdims=True)
        db_ih = np.sum(dZh, axis=0, keepdims=True)
        dW_ih = np.dot(X.T, dZh)
        dW_hh = np.dot(H_pre.T, dZh)
        dh_pre = np.dot(dZh, self.W_hh.T)
        dx = np.dot(dZh, self.W_ih.T)
        grads = (dW_ih, dW_hh, db_ih, db_hh)
        for a, b in zip(self.grads, grads):
            a += b
        return dx, dh_pre, grads


import numpy as np

np.random.seed(1)
x = np.random.randn(3, 10)  # (batch_size,input_dim)
h = np.random.randn(3, 20)  # (batch_size,hidden_dim)
rnn = RNNCell(10, 20)  # (input_dim,hidden_dim)

h_ = rnn(x, h)
print("h_:", h_)
dh_ = np.random.randn(*h.shape)
dx, dh, _ = rnn.backward(dh_, h_, x, h)
print("dh:", dh)

import numpy as np

x = np.random.randn(6, 3, 10)
h = np.random.randn(3, 20)
rnn = RNNCell(10, 20)

h_0 = h.copy()
hs = []
for i in range(6):
    h = rnn(x[i], h)
    hs.append(h)
print("h:", hs[0])

dh = np.random.randn(*h.shape)
for i in reversed(range(6)):
    if i == 0:
        dx, dh, _ = rnn.backward(dh, hs[i], x[i], h_0)
    else:
        dx, dh, _ = rnn.backward(dh, hs[i], x[i], hs[i - 1])
print("dh:", dh)
