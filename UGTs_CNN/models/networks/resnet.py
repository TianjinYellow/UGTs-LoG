"""
Implementation based on the code of https://github.com/kuangliu/pytorch-cifar/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.sparse_modules import SparseConv2d, SparseLinear


class ConvNormBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                    stride=1, padding=0, bias=True, cfg=None):
        super().__init__()

        self.conv = SparseConv2d(in_channels, out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, bias=bias, cfg=cfg)
        #if cfg['bn']:
        self.norm = nn.BatchNorm2d(out_channels,
                momentum=cfg['bn_momentum'],
                track_running_stats=cfg['bn_track_running_stats'],
                affine=cfg['bn_affine'])
        self.cfg=cfg
    def forward(self, x):
        out = self.conv(x)
        #if self.cfg['bn']:
        out = self.norm(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, cfg=None):
        super(BasicBlock, self).__init__()
        self.convnb1 = ConvNormBlock(in_planes, planes, kernel_size=3,
                stride=stride, padding=1, bias=False, cfg=cfg)
        self.convnb2 = ConvNormBlock(planes, planes, kernel_size=3,
                stride=1, padding=1, bias=False, cfg=cfg)

        self.cfg=cfg
        #if cfg['skipconnect']:
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = ConvNormBlock(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, cfg=cfg)

    def forward(self, x):
        out = F.relu(self.convnb1(x))
        out = self.convnb2(out)
        #if self.cfg['skipconnect']:
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride, cfg=None):
        super(Bottleneck, self).__init__()
        self.convnb1 = ConvNormBlock(in_planes, planes, kernel_size=1, bias=False, cfg=cfg)
        self.convnb2 = ConvNormBlock(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, cfg=cfg)
        self.convnb3 = ConvNormBlock(planes, self.expansion * planes, kernel_size=1, bias=False, cfg=cfg)
        self.cfg=cfg
        if cfg['skipconnect']:
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = ConvNormBlock(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, cfg=cfg)

    def forward(self, x):
        out = F.relu(self.convnb1(x))
        out = F.relu(self.convnb2(out))
        out = self.convnb3(out)
        if self.cfg['skipconnect']:
            out += self.shortcut(x)
        out = F.relu(out)
        return out

def percentile(t, q):
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values.item()
class ResNet(nn.Module):
    def __init__(self, dataset_cfg, model_cfg, cfg):
        super(ResNet, self).__init__()
        block = globals()[model_cfg['block_class']]
        num_blocks = model_cfg['num_blocks']
        factor = model_cfg['factor']
        self.sparsity=cfg['conv_sparsity']
        num_classes = dataset_cfg['num_classes']
        self.in_channel = dataset_cfg['num_channels']
        self.image_size = dataset_cfg['image_size']
        self.in_planes = 64
        if self.image_size not in [28, 32, 64, 84, 224]:
            raise NotImplementedError
        self.cfg=cfg
        if self.image_size in [224]:
            k1, s1, p1 = 7, 2, 3
        else:
            k1, s1, p1 = 3, 1, 1

        self.convnb1 = ConvNormBlock(self.in_channel, 64, kernel_size=k1,
                                     stride=s1, padding=p1, bias=False, cfg=cfg)

        self.layer1 = self._make_layer(block, int(64*factor), num_blocks[0], stride=1, cfg=cfg)
        self.layer2 = self._make_layer(block, int(128*factor), num_blocks[1], stride=2, cfg=cfg)
        self.layer3 = self._make_layer(block, int(256*factor), num_blocks[2], stride=2, cfg=cfg)
        self.layer4 = self._make_layer(block, int(512*factor), num_blocks[3], stride=2, cfg=cfg)

        self.linear = SparseLinear(int(512*factor*block.expansion), num_classes, cfg=cfg)

    def _make_layer(self, block, planes, num_blocks, stride, cfg):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, cfg))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def get_threshold(self,sparsity):
        local=[]
        for name, p in self.named_parameters():
            if hasattr(p, 'is_score') and p.is_score:
                local.append(p.detach().flatten())
        local=torch.cat(local)
        threshold=percentile(local,sparsity*100)
        return threshold
    
    def set_threshold(self,threshold):
        for module in self.modules():
            if hasattr(module,'globalthreshold'):
                module.globalthreshold=threshold
    def get_current_sparsity(self):
        s_e=0
        all_e=0
        score=0
        for name, p in self.named_parameters():
            if hasattr(p, 'is_score') and p.is_score:
                mask=p.clone().detach()<0
                score+=p.clone().detach().norm().item()
                s_e+=mask.float().sum().item()
                all_e+=mask.numel()
        return (s_e*1.0/all_e),score
    
    def forward(self, x):
        out = F.relu(self.convnb1(x))
        if self.image_size in [84, 224]:
            out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        elif self.image_size in [64]:
            out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if self.image_size in [224]:
            out = F.avg_pool2d(out, 7)
        elif self.image_size in [84]:
            out = F.avg_pool2d(out, 6)
        elif self.image_size in [64]:
            out = F.avg_pool2d(out, 4)
        else:
            out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    def set_train(self,mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        #self.training = mode
        for module in self.modules():
            if hasattr(module,'is_train'):
                module.is_train=mode
    def set_sparsity(self,s=0):
        #self.training = mode
        for module in self.modules():
            if hasattr(module,'sparsity'):
                module.sparsity=s
    
    def set_gumbelsoftmax(self,mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        #self.training = mode
        for module in self.modules():
            if hasattr(module,'gumbelsoftmax'):
                module.gumbelsoftmax=mode
      
    def rerandomize(self, mode, la, mu):
        for m in self.modules():
            if type(m) is SparseConv2d or type(m) is SparseLinear:
                m.rerandomize(mode, la, mu)


