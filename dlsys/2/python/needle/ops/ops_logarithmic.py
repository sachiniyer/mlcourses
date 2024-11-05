from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api


class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        maxz = array_api.max(Z, axis=-1, keepdims=True)
        return (
            Z
            - maxz
            - array_api.log(
                array_api.sum(array_api.exp(Z - maxz), axis=-1, keepdims=True)
            )
        )
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        y = exp(Z)
        return out_grad - y * broadcast_to(
            summation(out_grad, axes=1).reshape((Z.shape[0], 1))
            / summation(y, axes=1).reshape((Z.shape[0], 1)),
            Z.shape,
        )
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        maxz = array_api.max(Z, axis=self.axes, keepdims=True)
        ret = maxz + array_api.log(
            array_api.sum(array_api.exp(Z - maxz), axis=self.axes, keepdims=True)
        )
        shape = ()
        if self.axes is not None:
            shape = [Z.shape[i] for i in range(len(Z.shape)) if i not in self.axes]
        ret.resize(tuple(shape))
        return ret
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        if self.axes is None:
            return out_grad * exp(Z - node)
        shape_stack = list(node.shape)
        shape = [
            shape_stack.pop() if i not in self.axes else 1 for i in range(len(Z.shape))
        ]
        return out_grad.reshape(shape) * exp(Z - node.reshape(shape))
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
