"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad * (rhs * (lhs ** (rhs - 1)), out_grad * (log(lhs) * (lhs**rhs)))
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        return (out_grad * self.scalar * (lhs ** (self.scalar - 1)),)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / (rhs / 2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        order = list(range(len(a.shape)))
        if self.axes is None:
            order[-1] = order[-2]
            order[-2] = len(order) - 1
        else:
            order[self.axes[0]] = self.axes[1]
            order[self.axes[1]] = self.axes[0]
        return a.permute(tuple(order))

    ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return transpose(out_grad)
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        return out_grad.reshape(lhs.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        shape = list(node.inputs[0].shape)
        axes = []
        shape = [1] * (len(self.shape) - len(shape)) + shape
        for i, s in enumerate(self.shape):
            if i >= len(shape) or s != shape[i]:
                axes.append(i)
        return reshape(summation(out_grad, tuple(axes)), node.inputs[0].shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return a.sum()
        if isinstance(self.axes, int):
            return a.sum(self.axes)
        for i, axis in enumerate(sorted(list(self.axes))):
            a = a.sum(axis - i)
        return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        shape = node.inputs[0].shape
        shape_out = [1] * len(shape)
        if self.axes is not None:
            if isinstance(self.axes, int):
                s = set([self.axes])
            else:
                s = set(self.axes)
        else:
            s = set(range(len(shape)))

        j = 0
        for i in range(len(shape)):
            if i not in s:
                shape_out[i] = out_grad.shape[j]
                j += 1
        return broadcast_to(reshape(out_grad, tuple(shape_out)), shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        # Multiply the gradient by the transpose of the other matrix, as per the chain rule
        lhs_grad, rhs_grad = matmul(out_grad, transpose(rhs)), matmul(
            transpose(lhs), out_grad
        )
        # handle different shapes in the case of batched matmul
        if lhs_grad.shape != lhs.shape:
            lhs_axes = [i for i in range(len(lhs_grad.shape) - len(lhs.shape))]
            lhs_grad = summation(lhs_grad, axes=tuple(lhs_axes))
        if rhs_grad.shape != rhs.shape:
            rhs_axes = [i for i in range(len(rhs_grad.shape) - len(rhs.shape))]
            rhs_grad = summation(rhs_grad, axes=tuple(rhs_axes))
        return lhs_grad, rhs_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return exp(node.inputs[0]) * out_grad
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out = node.realize_cached_data()
        return out_grad * Tensor(out > 0, device=out_grad.device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.tanh()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (1 + (-tanh(node.inputs[0]) ** 2))
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        shape = args[0].shape
        new_shape = list(shape)
        new_shape.insert(self.axis, len(args))

        out = array_api.empty(new_shape, dtype=args[0].dtype, device=args[0].device)

        slices = []
        for i in range(len(new_shape)):
            if i != self.axis:
                slices.append(slice(new_shape[i]))
            else:
                slices.append(0)
        for i in range(len(args)):
            slices[self.axis] = i
            out[tuple(slices)] = args[i].reshape((1,) + shape)
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        slices = [slice(0, i) for i in A.shape]
        new_shape = tuple([A.shape[s] for s in range(len(A.shape)) if s != self.axis])
        tensors = []
        for i in range(A.shape[self.axis]):
            slices[self.axis] = slice(i, i + 1)
            tensors.append(A[tuple(slices)].compact().reshape(new_shape))
        return tuple(tensors)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(tuple(out_grad), self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        new_shape = list(a.shape)
        for axis in self.axes:
            new_shape[axis] = new_shape[axis] * (self.dilation + 1)
        new_shape = tuple(new_shape)
        res = a.device.full(new_shape, 0)
        slices = [slice(0, s) for s in res.shape]
        for axis in self.axes:
            slices[axis] = slice(0, res.shape[axis], self.dilation + 1)
        res[tuple(slices)] = a
        return res

    def gradient(self, out_grad, node):
        return UnDilate(self.axes, self.dilation)(out_grad)


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        slices = [slice(0, s) for s in a.shape]
        for axis in self.axes:
            slices[axis] = slice(0, a.shape[axis], self.dilation + 1)
        return a[tuple(slices)].compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        A = A.pad(
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
        )
        N, H, W, C_in = A.shape
        K, K2, KC_in, C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides

        inner_dim = K * K * C_in
        H_out, W_out = (H - K + 1) // self.stride, (W - K + 1) // self.stride
        A = (
            A.as_strided(
                shape=(N, H_out, W_out, K, K, C_in),
                strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs),
            )
            .compact()
            .reshape((N * H_out * W_out, inner_dim))
        )
        return (
            (A @ B.compact().reshape((K * K * C_in, C_out)))
            .compact()
            .reshape((N, H_out, W_out, C_out))
        )
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs
        K, _, _, _ = W.shape

        if self.stride > 1:
            out_grad = dilate(out_grad, (1, 2), self.stride - 1)
        return conv(
            out_grad, transpose(flip(W, (0, 1)), (2, 3)), padding=K - 1 - self.padding
        ), transpose(
            transpose(
                conv(
                    transpose(X, (0, 3)),
                    transpose(transpose(out_grad, (0, 1)), (1, 2)),
                    padding=self.padding,
                ),
                (0, 1),
            ),
            (1, 2),
        )
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
