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



"""
This section provides an interface for some PyTorch standard distributions, to provode a unified interface for different action PDF's.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

### Categorical
FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
# FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).unsqueeze(-1)
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

# FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)
FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)



### Normal
FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

# entropy = FixedNormal.entropy
# FixedNormal.entropy = lambda self: entropy(self).sum(-1)
normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean


# Bernoulli
FixedBernoulli = torch.distributions.Bernoulli

log_prob_bernoulli = FixedBernoulli.log_prob
FixedBernoulli.log_probs = lambda self, actions: log_prob_bernoulli(self, actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

# entropy = FixedBernoulli.entropy
# FixedBernoulli.entropy = lambda self: entropy(self).sum(-1)
bernoulli_entropy = FixedBernoulli.entropy
FixedBernoulli.entropy = lambda self: bernoulli_entropy(self).sum(-1)
FixedBernoulli.mode = lambda self: torch.gt(self.probs, 0.5).float()




class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        
        init_ = init_easy(gain=0.01, bias=0)
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        sampler = FixedCategorical(logits=x)
        return sampler



# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))
    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)
        return x + bias


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        # init_ = lambda m: init(m,
        #       init_normc_,
        #       lambda x: nn.init.constant_(x, 0))
        init_ = init_easy(gain=1, bias=0)
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        # An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        # action_logstd = zeros
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = init_easy(gain=1, bias=0)
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)






"""
This section provides building blocks for the policy.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class CNNBlock(nn.Module):
    def __init__(self, num_inputs, output_size):
        super(CNNBlock, self).__init__()
        init_ = init_easy(gain=nn.init.calculate_gain('relu'), bias=0)
        # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
        self.conv1 = init_(nn.Conv2d(num_inputs, 32, 8, stride=4))
        self.relu1 = nn.ReLU()
        self.conv2 = init_(nn.Conv2d(32, 64, 4, stride=2))
        self.relu2 = nn.ReLU()
        self.conv3 = init_(nn.Conv2d(64, 32, 3, stride=1))
        self.relu3 = nn.ReLU()
        # Flatten Here
        self.linear = init_(nn.Linear(32 * 7 * 7, output_size))
        self.relu4 = nn.ReLU()
    def forward(self, inputs):
        x = inputs
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        # flatten
        x = x.view(x.size(0), -1)
        x = self.relu4(self.linear(x))
        return x


class MLPBlock(nn.Module):
    def __init__(self, num_inputs, output_size):
        super(MLPBlock, self).__init__()
        init_ = init_easy(gain=np.sqrt(2), bias=0)
        self.linear1 = init_(nn.Linear(num_inputs, output_size))
        self.tanh1 = nn.Tanh()
        self.linear2 = init_(nn.Linear(output_size, output_size))
        self.tanh2 = nn.Tanh()
    
    def forward(self, inputs):
        x = self.tanh1(self.linear1(inputs))
        x = self.tanh2(self.linear2(x))
        return x


class RNNBlock(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(RNNBlock, self).__init__()

        self.gru = nn.GRU(num_inputs, hidden_size)
        init_rnn(named_params=self.gru.named_parameters(), gain=1, bias=0)
        
    def forward(self, x, hidden, masks):
        if not self.training:
            x, hidden = self.gru(x.unsqueeze(0), (hidden * masks).unsqueeze(0))
            x = x.squeeze(0)
            hidden = hidden.squeeze(0)
        else:
            # TODO: Decipher the following:
            N = hidden.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hidden = hidden.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hidden = self.gru(
                    x[start_idx:end_idx],
                    hidden * masks[start_idx].view(1, -1, 1)
                )

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hidden = hidden.squeeze(0)

        return x, hidden
