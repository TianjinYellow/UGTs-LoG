import numpy as np

# This scheduler is based on https://github.com/allenai/hidden-networks/blob/master/utils/schedulers.py

class CustomCosineLR(object):
    def __init__(self, optimizer, init_lr, total_epoch, warmup_length, ft_length, ft_lr):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.total_epoch = total_epoch
        self.warmup_length = warmup_length
        self.ft_length = ft_length
        self.ft_lr = ft_lr
        self.step_before_train = True

    def step(self, epoch=None):
        assert epoch is not None

        if epoch < self.warmup_length:
            lr = _warmup_lr(self.init_lr, self.warmup_length, epoch)
        elif self.warmup_length <= epoch < self.total_epoch - self.ft_length:
            e = epoch - self.warmup_length
            es = self.total_epoch - self.warmup_length - self.ft_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * self.init_lr
        elif self.total_epoch - self.ft_length <= epoch < self.total_epoch:
            lr = self.ft_lr
        else:
            lr = None

        _assign_learning_rate(self.optimizer, lr)

    def state_dict(self):
        return {}

    def load_state_dict(self, dic):
        return

def _assign_learning_rate(optimizer, new_lr=None):
    if new_lr is not None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
    else:
        pass

def _warmup_lr(base_lr, warmup_length, epoch):
    return base_lr * (epoch + 1) / warmup_length

