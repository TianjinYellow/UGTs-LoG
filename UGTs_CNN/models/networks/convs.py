
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.sparse_modules import SparseConv2d, SparseLinear


class Conv4(nn.Module):
    def __init__(self, dataset_cfg, model_cfg, cfg=None):
        super(Conv4, self).__init__()
        
        assert dataset_cfg['image_size'] == 32
        assert dataset_cfg['num_channels'] == 3

        self.num_classes = dataset_cfg['num_classes']
        self.factor = model_cfg['factor']

        self.convs = nn.Sequential(
            SparseConv2d(3, int(64 * self.factor), kernel_size=3,
                         stride=1, padding=1, cfg=cfg),
            nn.ReLU(),
            SparseConv2d(int(64 * self.factor), int(64 * self.factor), kernel_size=3,
                         stride=1, padding=1, cfg=cfg),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            SparseConv2d(int(64 * self.factor), int(128 * self.factor), kernel_size=3,
                         stride=1, padding=1, cfg=cfg),
            nn.ReLU(),
            SparseConv2d(int(128 * self.factor), int(128 * self.factor), kernel_size=3,
                         stride=1, padding=1, cfg=cfg),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        if cfg['linear_sparsity'] is not None:
            linear_cfg = cfg.copy()
            linear_cfg['conv_sparsity'] = cfg['linear_sparsity']
        else:
            linear_cfg = cfg

        self.linear = nn.Sequential(
            SparseConv2d(int(128 * self.factor) * 8 * 8, int(256 * self.factor), kernel_size=1, cfg=cfg),
            nn.ReLU(),
            SparseConv2d(int(256 * self.factor), int(256 * self.factor), kernel_size=1, cfg=cfg),
            nn.ReLU(),
            SparseConv2d(int(256 * self.factor), self.num_classes, kernel_size=1, cfg=linear_cfg)
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), int(128 * self.factor) * 8 * 8, 1, 1)
        out = self.linear(out)
        return out.squeeze()

    def rerandomize(self, mode, la, mu):
        for m in self.modules():
            if type(m) is SparseConv2d or type(m) is SparseLinear:
                m.rerandomize(mode, la, mu)


class Conv6(nn.Module):
    def __init__(self, dataset_cfg, model_cfg, cfg=None):
        super(Conv6, self).__init__()
        
        assert dataset_cfg['image_size'] == 32
        assert dataset_cfg['num_channels'] == 3

        self.num_classes = dataset_cfg['num_classes']
        self.factor = model_cfg['factor']

        self.convs = nn.Sequential(
            SparseConv2d(3, int(64 * self.factor), kernel_size=3,
                         stride=1, padding=1, cfg=cfg),
            nn.ReLU(),
            SparseConv2d(int(64 * self.factor), int(64 * self.factor), kernel_size=3,
                         stride=1, padding=1, cfg=cfg),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            SparseConv2d(int(64 * self.factor), int(128 * self.factor), kernel_size=3,
                         stride=1, padding=1, cfg=cfg),
            nn.ReLU(),
            SparseConv2d(int(128 * self.factor), int(128 * self.factor), kernel_size=3,
                         stride=1, padding=1, cfg=cfg),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            SparseConv2d(int(128 * self.factor), int(256 * self.factor), kernel_size=3,
                         stride=1, padding=1, cfg=cfg),
            nn.ReLU(),
            SparseConv2d(int(256 * self.factor), int(256 * self.factor), kernel_size=3,
                         stride=1, padding=1, cfg=cfg),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        if cfg['linear_sparsity'] is not None:
            linear_cfg = cfg.copy()
            linear_cfg['conv_sparsity'] = cfg['linear_sparsity']
        else:
            linear_cfg = cfg

        self.linear = nn.Sequential(
            SparseConv2d(int(256 * self.factor) * 4 * 4, int(256 * self.factor), kernel_size=1, cfg=cfg),
            nn.ReLU(),
            SparseConv2d(int(256 * self.factor), int(256 * self.factor), kernel_size=1, cfg=cfg),
            nn.ReLU(),
            SparseConv2d(int(256 * self.factor), self.num_classes, kernel_size=1, cfg=linear_cfg)
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), int(256 * self.factor) * 4 * 4, 1, 1)
        out = self.linear(out)
        return out.squeeze()

    def rerandomize(self, mode, la, mu):
        for m in self.modules():
            if type(m) is SparseConv2d or type(m) is SparseLinear:
                m.rerandomize(mode, la, mu)


