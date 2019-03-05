"""This module is highly inspired by `pytorch-a2c-ppo-acktr <https://github.com/ikostrikov/pytorch-a2c-ppo-acktr>`__.
"""

import torch
import torch.nn as nn


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
def init_easy(gain=1, bias=0):
    def _f(module):
        return init(module=module, weight_init=nn.init.orthogonal_, bias_init=lambda x: nn.init.constant_(x, bias), gain=gain)
    return _f

def init_rnn(named_params, gain=1, bias=0):
    for name, param in named_params:
        if 'bias' in name:
            nn.init.constant_(param, bias)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)

# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
# def init_normc_(weight, gain=1):
#     weight.normal_(0, 1)
#     weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))

