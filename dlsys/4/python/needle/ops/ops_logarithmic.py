from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND


class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        if self.axes is not None:
            maxz = Z.max(self.axes, keepdims=True)
        else:
            maxz = Z.max([1], keepdims=True)
        ret = Z - maxz.broadcast_to(Z.shape)
        ret = array_api.exp(ret)
        ret = ret.sum(axis=self.axes, keepdims=True)
        ret = array_api.log(ret).reshape(maxz.shape)
        ret = ret + maxz
        if self.axes is None:
            axes = list(range(len(Z.shape)))
        elif isinstance(self.axes, int):
            axes = [self.axes]
        else:
            axes = list(self.axes)

        if self.axes is not None:
            out_shape = [size for i, size in enumerate(Z.shape) if i not in axes]
        else:
            out_shape = [1]
        return ret.reshape(tuple(out_shape))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        if self.axes is None:
            return out_grad * exp(Z - node)
        shape = [1] * len(Z.shape)
        if isinstance(self.axes, int):
            s = set([self.axes])
        else:
            s = set(self.axes)
        j = 0
        for i in range(len(shape)):
            if i not in s:
                shape[i] = node.shape[j]
                j += 1
        return node.reshape(shape).broadcast_to(Z.shape) * exp(
            Z - out_grad.reshape(shape).broadcast_to(Z.shape)
        )
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
