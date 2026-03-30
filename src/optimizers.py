import numpy as np


#  BASE OPTIMIZER


class Optimizer:
    def step(self, W, dW, lr, idx):
        raise NotImplementedError



#  SGD (Vanilla)


class SGD(Optimizer):
    def step(self, W, dW, lr, idx=None):
        return W - lr * dW



#  SGD + MOMENTUM


class Momentum(Optimizer):
    def __init__(self, beta=0.9):
        self.beta = beta
        self.v = {}

    def step(self, W, dW, lr, idx):
        if idx not in self.v:
            self.v[idx] = np.zeros_like(W)

        self.v[idx] = self.beta * self.v[idx] + (1 - self.beta) * dW

        return W - lr * self.v[idx]



#  RMSProp


class RMSProp(Optimizer):
    def __init__(self, beta=0.9, eps=1e-8):
        self.beta = beta
        self.eps = eps
        self.s = {}

    def step(self, W, dW, lr, idx):
        if idx not in self.s:
            self.s[idx] = np.zeros_like(W)

        self.s[idx] = self.beta * self.s[idx] + (1 - self.beta) * (dW ** 2)

        return W - lr * dW / (np.sqrt(self.s[idx]) + self.eps)



#  ADAM 


class Adam(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, W, dW, lr, idx):
        if idx not in self.m:
            self.m[idx] = np.zeros_like(W)
            self.v[idx] = np.zeros_like(W)

        self.t += 1

        # First moment
        self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * dW

        # Second moment
        self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (dW ** 2)

        # Bias correction
        m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
        v_hat = self.v[idx] / (1 - self.beta2 ** self.t)

        return W - lr * m_hat / (np.sqrt(v_hat) + self.eps)



#  OPTIMIZER FACTORY


def get_optimizer(name="sgd"):
    name = name.lower()

    if name == "sgd":
        return SGD()

    elif name == "momentum":
        return Momentum()

    elif name == "rmsprop":
        return RMSProp()

    elif name == "adam":
        return Adam()

    else:
        raise ValueError(f" Unknown optimizer: {name}")