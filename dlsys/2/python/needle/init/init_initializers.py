import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    scale = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return scale * rand(fan_in, fan_out, low=-1, high=1)
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    scale = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return scale * randn(fan_in, fan_out, mean=0, std=1)
    ### End YOUR SOLUTION


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    scale = math.sqrt(6.0 / fan_in)
    return scale * rand(fan_in, fan_out, low=-1, high=1)
    ### END YOUR SOLUTION


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    scale = math.sqrt(2.0 / fan_in)
    return scale * randn(fan_in, fan_out, mean=0, std=1)
    ### END YOUR SOLUTION
