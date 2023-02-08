
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch import Tensor


class ConcreteDropout(nn.Module):

    """Concrete Dropout.
    Implementation of the Concrete Dropout module as described in the
    'Concrete Dropout' paper: https://arxiv.org/pdf/1705.07832
    """

    def __init__(self,
                 p_logit,
                 init_min: float = 0.1,
                 init_max: float = 0.1) -> None:

        """Concrete Dropout.
        Parameters
        ----------
        weight_regulariser : float
            Weight regulariser term.
        dropout_regulariser : float
            Dropout regulariser term.
        init_min : float
            Initial min value.
        init_max : float
            Initial max value.
        """

        super().__init__()

        # self.weight_regulariser = weight_regulariser
        # self.dropout_regulariser = dropout_regulariser

        init_min = np.log(init_min) - np.log(1.0 - init_min)
        init_max = np.log(init_max) - np.log(1.0 - init_max)

        #self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
        self.p_logit=p_logit
        self.p = torch.sigmoid(self.p_logit)

        self.regularisation = 0.0

    def forward(self, x: Tensor) -> Tensor:

        """Calculates the forward pass.
        The regularisation term for the layer is calculated and assigned to a
        class attribute - this can later be accessed to evaluate the loss.
        Parameters
        ----------
        x : Tensor
            Input to the Concrete Dropout.
        layer : nn.Module
            Layer for which to calculate the Concrete Dropout.
        Returns
        -------
        Tensor
            Output from the dropout layer.
        """

        output = self._concrete_dropout(x)

        # sum_of_squares = 0
        # for param in layer.parameters():
        #     sum_of_squares += torch.sum(torch.pow(param, 2))

        # weights_reg = self.weight_regulariser * sum_of_squares / (1.0 - self.p)

        # dropout_reg = self.p * torch.log(self.p)
        # dropout_reg += (1.0 - self.p) * torch.log(1.0 - self.p)
        # dropout_reg *= self.dropout_regulariser * x[0].numel()

        # self.regularisation = weights_reg + dropout_reg

        return output

    def _concrete_dropout(self, x: Tensor) -> Tensor:

        """Computes the Concrete Dropout.
        Parameters
        ----------
        x : Tensor
            Input tensor to the Concrete Dropout layer.
        Returns
        -------
        Tensor
            Outputs from Concrete Dropout.
        """

        eps = 1e-7
        tmp = 0.1

        self.p = torch.sigmoid(self.p_logit)
        u_noise = torch.rand_like(x)

        drop_prob = (torch.log(self.p + eps) -
                     torch.log(1 - self.p + eps) +
                     torch.log(u_noise + eps) -
                     torch.log(1 - u_noise + eps))

        drop_prob = torch.sigmoid(drop_prob / tmp)

        random_tensor = 1 - drop_prob
        retain_prob = 1 - self.p

        x = torch.mul(x, random_tensor) / retain_prob

        return x
#from torch.autograd import Variable
EPSILON = np.finfo(float).eps
def concrete_neuron(logit_p, train=True, temp=1.0 / 10.0, **kwargs):
    '''
    Use concrete distribution to approximate binary output. Here input is logit(keep_prob).
    '''
    if train is False:
        result = logit_p.data.new().resize_as_(logit_p.data).fill_(1.)
        result[logit_p.data < 0.] = 0.
        return result

    # Note that p is the retain probability here
    p = torch.sigmoid(logit_p)
    unif_noise = logit_p.data.new().resize_as_(logit_p.data).uniform_()

    approx = (
        torch.log(1. - p + EPSILON)
        - torch.log(p + EPSILON)
        + torch.log(unif_noise + EPSILON)
        - torch.log(1. - unif_noise + EPSILON)
    )
    drop_prob = torch.sigmoid(approx / temp)
    return (1. - drop_prob)


class GetSubnet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, sparsity, zeros, ones):
        k_val = percentile(scores, sparsity*100)
        out = torch.where(scores < k_val, zeros.to(scores.device), ones.to(scores.device))
        return out

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


class Multiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, subnet, weights):
        ctx.save_for_backward(weights)
        out=subnet*weights
        #k_val = percentile(scores, sparsity*100)
        #out = torch.where(scores < k_val, zeros.to(scores.device), ones.to(scores.device))
        return out

    @staticmethod
    def backward(ctx, g):
        weights=ctx.saved_tensors

        return g*weights[0], g




class GetSubnetG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, threshold, zeros, ones):
        #k_val = percentile(scores, sparsity*100)
        k_val=threshold
        out = torch.where(scores < k_val, zeros.to(scores.device), ones.to(scores.device))
        return out

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None




def percentile(t, q):
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values.item()


class SparseModule(nn.Module):
    def init_param_(self, param, init_mode=None, scale=None):
        if init_mode == 'kaiming_normal':
            nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="relu")
            param.data *= scale
        elif init_mode == 'uniform(-1,1)':
            nn.init.uniform_(param, a=-1, b=1)
            param.data *= scale
        elif init_mode == 'kaiming_uniform':
            nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
            param.data *= scale
        elif init_mode == 'signed_constant':
            # From github.com/allenai/hidden-networks
            fan = nn.init._calculate_correct_fan(param, 'fan_in')
            gain = nn.init.calculate_gain('relu')
            std = gain / math.sqrt(fan)
            nn.init.kaiming_normal_(param)    # use only its sign
            param.data = param.data.sign() * std
            param.data *= scale
        else:
            raise NotImplementedError

    def rerandomize_(self, param, mask, mode=None, la=None, mu=None,
                     init_mode=None, scale=None, param_twin=None):
        if param_twin is None:
            raise NotImplementedError
        else:
            param_twin = param_twin.to(param.device)

        with torch.no_grad():
            if mode == 'bernoulli':
                assert (la is not None) and (mu is None)
                rnd = param_twin
                self.init_param_(rnd, init_mode=init_mode, scale=scale)
                ones = torch.ones(param.size()).to(param.device)
                b = torch.bernoulli(ones * la)

                t1 = param.data * mask
                t2 = param.data * (1 - mask) * (1 - b)
                t3 = rnd.data * (1 - mask) * b

                param.data = t1 + t2 + t3
            elif mode == 'manual':
                assert (la is not None) and (mu is not None)

                t1 = param.data * (1 - mask)
                t2 = param.data * mask

                rnd = param_twin
                self.init_param_(rnd, init_mode=init_mode, scale=scale)
                rnd *= (1 - mask)

                param.data = (t1*la + rnd.data*mu) + t2
            else:
                raise NotImplementedError


class SparseConv2d(SparseModule):
    def __init__(self, in_ch, out_ch, **kwargs):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch

        self.kernel_size = kwargs['kernel_size']
        self.stride = kwargs['stride'] if 'stride' in kwargs else 1
        self.padding = kwargs['padding'] if 'padding' in kwargs else 0
        self.bias_flag = kwargs['bias'] if 'bias' in kwargs else True
        self.padding_mode = kwargs['padding_mode'] if 'padding_mode' in kwargs else None
        self.is_train=False
        cfg = kwargs['cfg']
        self.cfg=cfg
        self.couple=cfg['couple']
        self.gumbelsoftmax=cfg['gumbelsoftmax']
        self.sparsity = cfg['conv_sparsity']
        self.init_mode = cfg['init_mode']
        self.init_mode_mask = cfg['init_mode_mask']
        self.init_scale = cfg['init_scale']
        self.init_scale_score = cfg['init_scale_score']
        self.rerand_rate = cfg['rerand_rate']
        self.globalprune=cfg['globalprune']
        #self.ste=cfg['ste']
        self.function = F.conv2d
        self.initialize_weights(2)
        if self.gumbelsoftmax:
            self.concretedrop=ConcreteDropout(self.weight_score)
        self.globalthreshold=None

    def initialize_weights(self, convdim=None):
        if convdim == 1:
            self.weight = nn.Parameter(torch.ones(self.out_ch, self.in_ch, self.kernel_size))
        elif convdim == 2:
            self.weight = nn.Parameter(torch.ones(self.out_ch, self.in_ch, self.kernel_size, self.kernel_size))
        else:
            raise NotImplementedError

        self.weight_score = nn.Parameter(torch.ones(self.weight.size()))
        self.weight_score.is_score = True
        self.weight_score.sparsity = self.sparsity
        
        self.weight_twin = torch.zeros(self.weight.size())
        self.weight_twin.requires_grad = False

        if self.bias_flag:
            self.bias = nn.Parameter(torch.zeros(self.out_ch))
        else:
            self.bias = None

        self.init_param_(self.weight_score, init_mode=self.init_mode_mask, scale=self.init_scale_score)
        self.init_param_(self.weight, init_mode=self.init_mode, scale=self.init_scale)

        self.weight_zeros = torch.zeros(self.weight_score.size())
        self.weight_ones = torch.ones(self.weight_score.size())
        self.weight_zeros.requires_grad = False
        self.weight_ones.requires_grad = False

    def get_subnet(self, weight_score=None):
        if weight_score is None:
            weight_score = self.weight_score
        if self.globalprune:
            subnet = GetSubnetG.apply(self.weight_score, self.globalthreshold,self.weight_zeros, self.weight_ones)
        else:
            subnet = GetSubnet.apply(self.weight_score, self.sparsity,self.weight_zeros, self.weight_ones)
        return subnet

    def forward(self, input):
        if self.gumbelsoftmax:
            subnet=concrete_neuron(self.weight_score,train=self.is_train)
            #pruned_weight=self.concretedrop(self.weight)
            #pruned_weight=subnet*self.weight
        else:
            subnet = self.get_subnet(self.weight_score)
        
        if self.ste:
            pruned_weight=Multiply.apply(subnet,self.weight)
        else:
            pruned_weight = self.weight * subnet
        
        ret = self.function(
            input, pruned_weight, self.bias, self.stride, self.padding,
        )
        return ret

    def rerandomize(self, mode, la, mu):
        rate = self.rerand_rate
        mask = GetSubnet.apply(self.weight_score, self.sparsity * rate,
                               self.weight_zeros, self.weight_ones)
        scale = self.init_scale
        self.rerandomize_(self.weight, mask, mode, la, mu,
                self.init_mode, scale, self.weight_twin)

class SparseConv1d(SparseConv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.function = F.conv1d
        self.initialize_weights(1)


class SparseLinear(SparseModule):
    def __init__(self, in_ch, out_ch, bias=True, cfg=None):
        super().__init__()
        self.cfg=cfg
        if cfg['linear_sparsity'] is not None:
            self.sparsity = cfg['linear_sparsity']
        else:
            self.sparsity = cfg['conv_sparsity']

        if cfg['init_mode_linear'] is not None:
            self.init_mode = cfg['init_mode_linear']
        else:
            self.init_mode = cfg['init_mode']

        self.init_mode_mask = cfg['init_mode_mask']
        self.couple=cfg['couple']
        self.init_scale = cfg['init_scale']
        self.init_scale_score = cfg['init_scale_score']
        self.rerand_rate = cfg['rerand_rate']
        self.globalprune=cfg['globalprune']
        self.ste=cfg['ste']
        self.is_train=False
        self.gumbelsoftmax=cfg['gumbelsoftmax']
        self.weight = nn.Parameter(torch.ones(out_ch, in_ch))
        self.weight_score = nn.Parameter(torch.ones(self.weight.size()))
        self.weight_score.is_score = True
        self.weight_score.sparsity = self.sparsity
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_ch))
        else:
            self.bias = None

        self.weight_twin = torch.zeros(self.weight.size())
        self.weight_twin.requires_grad = False

        self.init_param_(self.weight_score, init_mode=self.init_mode_mask, scale=self.init_scale_score)
        self.init_param_(self.weight, init_mode=self.init_mode, scale=self.init_scale)

        self.weight_zeros = torch.zeros(self.weight_score.size())
        self.weight_ones = torch.ones(self.weight_score.size())
        self.weight_zeros.requires_grad = False
        self.weight_ones.requires_grad = False
        if self.gumbelsoftmax:
            self.concretedrop=ConcreteDropout(self.weight_score)
        self.globalthreshold=None
    def forward(self, x,manual_mask=None):
        if manual_mask is None:
            if self.gumbelsoftmax:
                subnet=concrete_neuron(self.weight_score,train=self.is_train)
                #pruned_weight=self.concretedrop(self.weight)
                #pruned_weight=subnet*self.weight
                
            else:
                if self.globalprune:
                    subnet = GetSubnetG.apply(self.weight_score, self.globalthreshold,self.weight_zeros, self.weight_ones)
                else:
                    subnet = GetSubnet.apply(self.weight_score, self.sparsity,self.weight_zeros, self.weight_ones)
            if self.ste:
                pruned_weight=Multiply.apply(subnet,self.weight)
            else:
                pruned_weight = self.weight * subnet
        else:
            pruned_weight = self.weight * manual_mask

        ret = F.linear(x, pruned_weight, self.bias)
        return ret

    def rerandomize(self, mode, la, mu, manual_mask=None):
        if manual_mask is None:
            rate = self.rerand_rate
            mask = GetSubnet.apply(self.weight_score, self.sparsity * rate,
                                   self.weight_zeros, self.weight_ones)
        else:
            mask = manual_mask

        scale = self.init_scale
        self.rerandomize_(self.weight, mask, mode, la, mu,
                self.init_mode, scale, self.weight_twin)

