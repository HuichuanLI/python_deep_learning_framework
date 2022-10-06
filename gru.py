import numpy as np
import math
import util
from util import *
import math


class GRU(object):
    def __init__(self, input_dim, hidden_dim, output_dim, scale=0.01):
        super(GRU, self).__init__()
        self.input_dim, self.hidden_dim, self.output_dim, self.scale = input_dim, hidden_dim, output_dim, scale

        normal = lambda m, n: np.random.randn(m, n) * scale
        three = lambda: (normal(input_dim, hidden_dim), normal(hidden_dim, hidden_dim), np.zeros((1, hidden_dim)))

        Wxu, Whu, bu = three()  # Update gate parameter
        Wxr, Whr, br = three()  # Reset gate parameter
        Wxh, Whh, bh = three()  # Candidate hidden state parameter

        Wy = normal(hidden_dim, output_dim)
        by = np.zeros((1, output_dim))

        self.Wxu, self.Whu, self.bu, self.Wxr, self.Whr, self.br, self.Wxh, self.Whh, self.bh, self.Wy, self.by = Wxu, Whu, bu, Wxr, Whr, br, Wxh, Whh, bh, Wy, by

        self.params = [Wxu, Whu, bu, Wxr, Whr, br, Wxh, Whh, bh, Wy, by]
        # [dWxu, dWhu, dbu, dWxr, dWhr, dbr, dWxh, dWhh, dbh, dWy,dby]
        self.grads = [np.zeros_like(param) for param in self.params]

        self.H = None
        # params = [Wxu, Whu, bu, Wxr, Whr, br, Wxh, Whh, bh, Wy,by]
        # return params

    def reset_state(self, batch_size):
        self.H = np.zeros((batch_size, self.hidden_dim))

    def forward_step(self, X):
        Wxu, Whu, bu, Wxr, Whr, br, Wxh, Whh, bh, Wy, by = self.params
        H = self.H  # previous state
        X = Xs[t]
        U = sigmoid(np.dot(X, Wxu) + np.dot(H, Whu) + bu)
        R = sigmoid(np.dot(X, Wxr) + np.dot(H, Whr) + br)
        H_tilda = np.tanh(np.dot(X, Wxh) + np.dot(R * H, Whh) + bh)
        H = U * H + (1 - U) * H_tilda
        Y = np.dot(H, Wy) + by

        Hs[t] = H
        Ys.append(Y)
        Rs.append(R)
        Us.append(U)
        H_tildas.append(H_tilda)

    def forward(self, Xs):
        Wxu, Whu, bu, Wxr, Whr, br, Wxh, Whh, bh, Wy, by = self.params
        if self.H is None:
            self.reset_state(Xs[0].shape[0])
        H = self.H
        Hs = {}
        Ys = []
        Hs[-1] = np.copy(H)
        Rs = []
        Us = []
        H_tildas = []

        for t in range(len(Xs)):
            X = Xs[t]
            U = sigmoid(np.dot(X, Wxu) + np.dot(H, Whu) + bu)
            R = sigmoid(np.dot(X, Wxr) + np.dot(H, Whr) + br)
            H_tilda = np.tanh(np.dot(X, Wxh) + np.dot(R * H, Whh) + bh)
            H = U * H + (1 - U) * H_tilda
            Y = np.dot(H, Wy) + by

            Hs[t] = H
            Ys.append(Y)
            Rs.append(R)
            Us.append(U)
            H_tildas.append(H_tilda)

        self.Ys, self.Hs, self.Rs, self.Us, self.H_tildas = Ys, Hs, Rs, Us, H_tildas
        return Ys, Hs
        # return Ys,Hs,(Rs,Us,H_tildas)

    def backward(self, dZs):  # Ys,loss_function):
        Wxu, Whu, bu, Wxr, Whr, br, Wxh, Whh, bh, Wy, by = self.params
        Ys, Hs, Rs, Us, H_tildas = self.Ys, self.Hs, self.Rs, self.Us, self.H_tildas

        dWxu, dWhu, dWxr, dWhr, dWxh, dWhh, dWy = np.zeros_like(Wxu), np.zeros_like(Whu), np.zeros_like \
            (Wxr), np.zeros_like(Whr) \
            , np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Wy)
        dbu, dbr, dbh, dby = np.zeros_like(bu), np.zeros_like(br), np.zeros_like(bh), np.zeros_like(by)

        dH_next = np.zeros_like(Hs[0])

        input_dim = Xs[0].shape[1]

        T = len(Xs)
        for t in reversed(range(T)):
            R = Rs[t]
            U = Us[t]
            H = Hs[t]
            X = Xs[t]
            H_tilda = H_tildas[t]
            H_pre = Hs[t - 1]

            dZ = dZs[t]
            # 输出f的模型参数的idu
            dWy += np.dot(H.T, dZ)
            dby += np.sum(dZ, axis=0, keepdims=True)

            # 隐状态h的梯度
            dH = np.dot(dZ, Wy.T) + dH_next

            #  H =  U H_pre+(1-U)H_tildas
            dH_tilda = d
            H * (1 - U)
            dH_pre = d
            H * U
            dU = H_pr
            e * dH - H_tilda * dH

            # H_tilda = tanh(X Wxh+(R*H_)Whh+bh)
            dH_tildaZ = (1 - np.square(H_tilda)) * dH_tilda
            dWxh += np.dot(X.T, dH_tildaZ)
            dWhh += np.dot((R * H_pre).T, dH_tildaZ)
            dbh += np.sum(dH_tildaZ, axis=0, keepdims=True)

            dR = np.dot(dH_tildaZ, Whh.T) * H_pre
            dH_pre += np.dot(dH_tildaZ, Whh.T) * R

            # U = \sigma(UZ)   R = \sigma(RZ)
            dUZ = U * (1 - U) * dU
            dRZ = R * (1 - R) * dR

            dH_pre += np.dot(dUZ, Whu.T)
            dH_pre += np.dot(dRZ, Whr.T)

            # R = \sigma(X Wxr+H_ Whr + br)
            dWxr += np.dot(X.T, dRZ)
            dWhr += np.dot(H_pre.T, dRZ)
            dbr += np.sum(dRZ, axis=0, keepdims=True)

            dWxu += np.dot(X.T, dUZ)
            dWhu += np.dot(H_pre.T, dUZ)
            dbu += np.sum(dUZ, axis=0, keepdims=True)

            if True:
                dX_RZ = np.dot(dRZ, Wxr.T)
                dX_UZ = np.dot(dUZ, Wxu.T)
                dX_H_tildaZ = np.dot(dH_tildaZ, Wxh.T)
                dX = dX_RZ + dX_UZ + dX_H_tildaZ

            dH_next = dH_pre

        grads = [dWxu, dWhu, dbu, dWxr, dWhr, dbr, dWxh, dWhh, dbh, dWy, dby]
        for i, _ in enumerate(self.grads):
            self.grads[i] += grads[i]

        return self.grads
        # return [dWxu, dWhu, dbu, dWxr, dWhr, dbr, dWxh, dWhh, dbh, dWy,dby]

    def get_states(self):
        return self.Hs

    def get_outputs(self):
        return self.Ys

    def parameters(self):
        return self.params


T = 3
input_dim, hidden_dim, output_dim = 4, 3, 4
batch_size = 2
Xs = np.random.randn(T, batch_size, input_dim)
Ys = np.random.randint(output_dim, size=(T, batch_size))

print("Xs", Xs)
print("Ys", Ys)

gru = GRU(input_dim, hidden_dim, output_dim)
Zs, Hs = gru.forward(Xs)

loss_function = rnn_loss_grad
loss, dZs = loss_function(Zs, Ys)
grads = gru.backward(dZs)


def rnn_loss():
    lstm.reset_state(batch_size)
    Zs, Hs = gru.forward(Xs)
    loss_function = rnn_loss_grad
    loss, dZs = loss_function(Zs, Ys)
    return loss


params = gru.parameters()
numerical_grads = util.numerical_gradient(rnn_loss, params, 1e-6)  # rnn_numerical_gradient(rnn_loss,params,1e-10)
# diff_error = lambda x, y: np.max( np.abs(x - y)/(np.maximum(1e-8, np.abs(x) + np.abs(y))))
diff_error = lambda x, y: np.max(np.abs(x - y))

print("loss", loss)
print("[Wxu, Whu, bu, Wxr, Whr, br, Wxh, Whh, bh, Wy,by] ")
for i in range(len(grads)):
    print(diff_error(grads[i], numerical_grads[i]))

print("grads", grads[0])
print("numerical_grads", numerical_grads[0])


