"""Optimization module"""

import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for w in self.params:
            if w not in self.u.keys():
                self.u[w] = ndl.init.zeros(
                    *w.shape, device=w.device, dtype=w.dtype, requires_grad=False
                )
            self.u[w] = self.momentum * self.u[w].data + (1.0 - self.momentum) * (
                w.grad.data + self.weight_decay * w.data
            )
            w.data = w.data - self.lr * self.u[w].data
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)  # Initial parameter vector
        self.lr = lr  # Learning rate
        self.beta1 = beta1  # Exponential decay rate for the first moment estimates
        self.beta2 = beta2  # Exponential decay rate for the second moment estimates
        self.eps = eps  # A small constant for numerical stability
        self.weight_decay = weight_decay
        self.t = 0  # Initialize timestep

        self.m = {}  # Initialize first moment vector
        self.v = {}  # Initialize second moment vector

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for w in self.params:
            if w not in self.m.keys():
                self.m[w] = ndl.init.zeros(
                    *w.shape, device=w.device, dtype=w.dtype, requires_grad=False
                )
                self.v[w] = ndl.init.zeros(
                    *w.shape, device=w.device, dtype=w.dtype, requires_grad=False
                )
            l2_w = w.grad.data + self.weight_decay * w.data
            self.m[w] = self.beta1 * self.m[w].data + (1.0 - self.beta1) * l2_w
            self.v[w] = self.beta2 * self.v[w].data + (1.0 - self.beta2) * (l2_w * l2_w)
            m_hat = (
                self.m[w].data / (1.0 - self.beta1**self.t)
                if self.t > 0
                else self.m[w].data
            )
            v_hat = (
                self.v[w].data / (1.0 - self.beta2**self.t)
                if self.t > 0
                else self.v[w].data
            )
            w.data = w.data - self.lr * m_hat.data / (
                ndl.ops.power_scalar(v_hat.data, 0.5) + self.eps
            )
        ### END YOUR SOLUTION
