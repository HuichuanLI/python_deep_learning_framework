import numpy as np
import math
import util
from util import *
import math


def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def dtanh(x):
    return 1 - np.tanh(x) * np.tanh(x)


def grad_clipping(grads, alpha):
    norm = math.sqrt(sum((grad ** 2).sum() for grad in grads))
    if norm > alpha:
        ratio = alpha / norm
        for i in range(len(grads)):
            grads[i] *= ratio


def lstm_backward(params, Xs, Hs, Cs, dZs, cache, clip_value=5.):  # Ys,loss_function):
    [Wi, bi, Wf, bf, Wo, bo, Wc, bc, Wy, by] = params

    Is, Fs, Os, C_tildas = cache

    dWi, dWf, dWo, dWc, dWy = np.zeros_like(Wi), np.zeros_like(Wf), np.zeros_like(Wo), np.zeros_like(Wc), np.zeros_like(
        Wy)
    dbi, dbf, dbo, dbc, dby = np.zeros_like(bi), np.zeros_like(bf), np.zeros_like(bo), np.zeros_like(bc), np.zeros_like(
        by)

    dH_next = np.zeros_like(Hs[0])
    dC_next = np.zeros_like(Cs[0])

    input_dim = Xs[0].shape[1]

    h = Hs
    x = Xs

    T = len(Xs)
    for t in reversed(range(T)):
        I = Is[t]
        F = Fs[t]
        O = Os[t]
        C_tilda = C_tildas[t]
        H = Hs[t]
        X = Xs[t]
        C = Cs[t]
        H_pre = Hs[t - 1]
        C_prev = Cs[t - 1]
        XH_pre = np.column_stack((X, H_pre))
        XH_ = XH_pre

        dZ = dZs[t]

        # 输出f的模型参数的idu
        dWy += np.dot(H.T, dZ)
        dby += np.sum(dZ, axis=0, keepdims=True)

        # 隐状态h的梯度
        dH = np.dot(dZ, Wy.T) + dH_next

        dC = dH * O * dtanh(C) + dC_next  # H_t= O_t*tanh(C_t)

        dO = np.tanh(C) * dH
        dOZ = O * (1 - O) * dO  # O = sigma(Z_o)
        dWo += np.dot(XH_.T, dOZ)  # Z_o = (X,H_)W_o+b_o
        dbo += np.sum(dOZ, axis=0, keepdims=True)

        # di
        di = C_tilda * dC
        diZ = I * (1 - I) * di
        dWi += np.dot(XH_.T, diZ)
        dbi += np.sum(diZ, axis=0, keepdims=True)

        # df
        df = C_prev * dC
        dfZ = F * (1 - F) * df
        dWf += np.dot(XH_.T, dfZ)
        dbf += np.sum(dfZ, axis=0, keepdims=True)

        # dC_bar
        dC_tilda = I * dC  # C = F * C + I * C_tilda
        dC_tilda_Z = (1 - np.square(C_tilda)) * dC_tilda  # C_tilda = tanh(C_tilda_Z)
        dWc += np.dot(XH_.T, dC_tilda_Z)  # C_tilda_Z = (X,H_)W_c+b_c
        dbc += np.sum(dC_tilda_Z, axis=0, keepdims=True)

        dXH_ = (np.dot(dfZ, Wf.T)
                + np.dot(diZ, Wi.T)
                + np.dot(dC_tilda_Z, Wc.T)
                + np.dot(dOZ, Wo.T))

        dX_prev = dXH_[:, :input_dim]
        dH_prev = dXH_[:, input_dim:]
        dC_prev = F * dC

        dC_next = dC_prev
        dH_next = dH_prev

    grads = [dWi, dbi, dWf, dbf, dWo, dbo, dWc, dbc, dWy, dby]
    grad_clipping(grads, clip_value)
    # for dparam in [dWi, dbi,dWf, dbf, dWo,dbo,dWc, dbc,dWy,dby]:
    #    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
    return grads


def grad_clipping(grads, alpha):
    norm = math.sqrt(sum((grad ** 2).sum() for grad in grads))
    if norm > alpha:
        ratio = alpha / norm
        for i in range(len(grads)):
            grads[i] *= ratio


class LSTM(object):
    def __init__(self, input_dim, hidden_dim, output_dim, scale=0.01):
        # super(LSTM_cell, self).__init__()
        self.input_dim, self.hidden_dim, self.output_dim = input_dim, hidden_dim, output_dim
        normal = lambda m, n: np.random.randn(m, n) * scale
        two = lambda: (normal(input_dim + hidden_dim, hidden_dim), np.zeros((1, hidden_dim)))

        Wi, bi = two()  # Input gate parameters
        Wf, bf = two()  # Forget gate parameters
        Wo, bo = two()  # Output gate parameters
        Wc, bc = two()  # Candidate cell parameters

        Wy = normal(hidden_dim, output_dim)
        by = np.zeros((1, output_dim))

        # params = [Wi, bi,Wf, bf, Wo,bo, Wc,bc,Wy,by]
        #  return params
        self.params = [Wi, bi, Wf, bf, Wo, bo, Wc, bc, Wy, by]
        self.grads = [np.zeros_like(param) for param in self.params]
        self.H, self.C = None, None

    def reset_state(self, batch_size):
        self.H, self.C = (np.zeros((batch_size, self.hidden_dim)),
                          np.zeros((batch_size, self.hidden_dim)))

    def forward(self, Xs):
        [Wi, bi, Wf, bf, Wo, bo, Wc, bc, Wy, by] = self.params

        if self.H is None or self.C is None:
            self.reset_state(Xs[0].shape[0])

        H, C = self.H, self.C
        Hs = {}
        Cs = {}
        Zs = []

        Hs[-1] = np.copy(H)
        Cs[-1] = np.copy(C)

        Is = []
        Fs = []
        Os = []
        C_tildas = []

        for t in range(len(Xs)):
            X = Xs[t]
            XH = np.column_stack((X, H))

            I = sigmoid(np.dot(XH, Wi) + bi)
            F = sigmoid(np.dot(XH, Wf) + bf)
            O = sigmoid(np.dot(XH, Wo) + bo)
            C_tilda = np.tanh(np.dot(XH, Wc) + bc)

            C = F * C + I * C_tilda
            H = O * np.tanh(C)  # O * C.tanh()  #输出状态

            Y = np.dot(H, Wy) + by  # 输出

            Zs.append(Y)
            Hs[t] = H
            Cs[t] = C

            Is.append(I)
            Fs.append(F)
            Os.append(O)
            C_tildas.append(C_tilda)
        self.Zs, self.Hs, self.Cs, self.Is, self.Fs, self.Os, self.C_tildas = Zs, Hs, Cs, Is, Fs, Os, C_tildas
        self.Xs = Xs

        # return Zs,Hs,Cs,(Is,Fs,Os,C_tildas)
        return Zs, Hs

    def backward(self, dZs):  # Ys,loss_function):
        [Wi, bi, Wf, bf, Wo, bo, Wc, bc, Wy, by] = self.params

        Hs, Cs, Is, Fs, Os, C_tildas = self.Hs, self.Cs, self.Is, self.Fs, self.Os, self.C_tildas
        Xs = self.Xs

        dWi, dWf, dWo, dWc, dWy = np.zeros_like(Wi), np.zeros_like(Wf), np.zeros_like(Wo), np.zeros_like(
            Wc), np.zeros_like(Wy)
        dbi, dbf, dbo, dbc, dby = np.zeros_like(bi), np.zeros_like(bf), np.zeros_like(bo), np.zeros_like(
            bc), np.zeros_like(by)

        dH_next = np.zeros_like(Hs[0])
        dC_next = np.zeros_like(Cs[0])

        input_dim = Xs[0].shape[1]

        h = Hs
        x = Xs

        T = len(Xs)
        for t in reversed(range(T)):
            I = Is[t]
            F = Fs[t]
            O = Os[t]
            C_tilda = C_tildas[t]
            H = Hs[t]
            X = Xs[t]
            C = Cs[t]
            H_pre = Hs[t - 1]
            C_prev = Cs[t - 1]
            XH_pre = np.column_stack((X, H_pre))
            XH_ = XH_pre

            dZ = dZs[t]

            # 输出f的模型参数的idu
            dWy += np.dot(H.T, dZ)
            dby += np.sum(dZ, axis=0, keepdims=True)

            # 隐状态h的梯度
            dH = np.dot(dZ, Wy.T) + dH_next
            #  dC = dH_next*O*dtanh(C) +dC_next    #* H = O*np.tanh(C)
            #  dC = dH_next*O*(1-np.square(np.tanh(C))) +dC_next
            dC = dH * O * dtanh(C) + dC_next

            dO = np.tanh(C) * dH
            dOZ = O * (1 - O) * dO
            dWo += np.dot(XH_.T, dOZ)
            dbo += np.sum(dOZ, axis=0, keepdims=True)

            # di
            di = C_tilda * dC
            diZ = I * (1 - I) * di
            dWi += np.dot(XH_.T, diZ)
            dbi += np.sum(diZ, axis=0, keepdims=True)

            # df
            df = C_prev * dC
            dfZ = F * (1 - F) * df
            dWf += np.dot(XH_.T, dfZ)
            dbf += np.sum(dfZ, axis=0, keepdims=True)

            # dC_bar
            dC_tilda = I * dC  # C = F * C + I * C_tilda
            dC_tilda_Z = (1 - np.square(C_tilda)) * dC_tilda  # C_tilda = sigmoid(np.dot(XH, Wc)+bc)
            dWc += np.dot(XH_.T, dC_tilda_Z)
            dbc += np.sum(dC_tilda_Z, axis=0, keepdims=True)

            dXH_ = (np.dot(dfZ, Wf.T)
                    + np.dot(diZ, Wi.T)
                    + np.dot(dC_tilda_Z, Wc.T)
                    + np.dot(dOZ, Wo.T))

            dX_prev = dXH_[:, :input_dim]
            dH_prev = dXH_[:, input_dim:]
            dC_prev = F * dC

            dC_next = dC_prev
            dH_next = dH_prev

            # for dparam in [dWi, dbi,dWf, dbf, dWo,dbo,dWc, dbc,dWy,dby]:
        #    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
        grads = [dWi, dbi, dWf, dbf, dWo, dbo, dWc, dbc, dWy, dby]
        grad_clipping(grads, 5.)
        for i, _ in enumerate(self.grads):
            self.grads[i] += grads[i]

        return [dWi, dbi, dWf, dbf, dWo, dbo, dWc, dbc, dWy, dby]

    def parameters(self):
        return self.params


T = 3
input_dim, hidden_dim, output_dim = 4, 3, 4
batch_size = 2
Xs = np.random.randn(T, batch_size, input_dim)
Ys = np.random.randint(output_dim, size=(T, batch_size))

print("Xs", Xs)
print("Ys", Ys)

lstm = LSTM(input_dim, hidden_dim, output_dim)
Zs, Hs = lstm.forward(Xs)

loss_function = lambda F, Y: rnn_loss_grad(F, Y)  # ,util.loss_grad_least)

loss_function = rnn_loss_grad
loss, dZs = loss_function(Zs, Ys)
grads = lstm.backward(dZs)


def rnn_loss():
    lstm.reset_state(batch_size)
    Zs, Hs = lstm.forward(Xs)
    loss_function = rnn_loss_grad
    loss, dZs = loss_function(Zs, Ys)
    return loss


params = lstm.parameters()
numerical_grads = util.numerical_gradient(rnn_loss, params, 1e-6)  # rnn_numerical_gradient(rnn_loss,params,1e-10)
# diff_error = lambda x, y: np.max( np.abs(x - y)/(np.maximum(1e-8, np.abs(x) + np.abs(y))))
diff_error = lambda x, y: np.max(np.abs(x - y))

print("loss", loss)
print("[Wi, bi,Wf, bf, Wo,bo,Wc, bc,Wy,by] ")
for i in range(len(grads)):
    print(diff_error(grads[i], numerical_grads[i]))

print("grads", grads[0])
print("numerical_grads", numerical_grads[0])
