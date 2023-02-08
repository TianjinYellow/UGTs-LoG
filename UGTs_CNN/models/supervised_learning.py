import torch
import torch.nn as nn
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import ToTensor, Resize, Compose, ColorJitter, RandomResizedCrop, RandomHorizontalFlip, Normalize, CenterCrop, Pad
from torch.nn import DataParallel
from torch.optim.lr_scheduler import MultiStepLR

from utils.subset_dataset import SubsetDataset, random_split
import utils.datasets
from utils.schedulers import CustomCosineLR
from models.networks.resnet import ResNet
from models.networks.convs import Conv6

import random


class SupervisedLearning(object):
    def __init__(self, outman, cfg, device, data_parallel):
        self.outman = outman
        self.cfg = cfg
        self.device = device
        self.data_parallel = data_parallel

        self.debug_max_iters = self.cfg['debug_max_iters']
        self.train_augmentation = self.cfg['train_augmentation']
        self.dataset_cfg = self.cfg['__other_configs__'][self.cfg['dataset.config_name']]

        self.model_cfg = self.cfg['__other_configs__'][self.cfg['model.config_name']]

        self.train_dataset, self.val_dataset, self.test_dataset = self._get_datasets()
        self.model = self._get_model().to(self.device)
        self.optimizer = self._get_optimizer(self.cfg['train_mode'])
        self.criterion = self._get_criterion()
        self.scheduler = self._get_scheduler()

    def train(self, epoch, total_iters, before_callback=None, after_callback=None):
        self.model.train()
        self.model.set_train(True)

        batch_size = self.cfg['batch_size']
        num_workers = self.cfg['num_workers']
        dataloader = DataLoader(self.train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers)

        results = []
        total_count = 0
        total_loss = 0.
        correct = 0
        if self.debug_max_iters is None:
            iters_per_epoch = len(dataloader)
        else:
            iters_per_epoch = min(len(dataloader), self.debug_max_iters)

        # for the case of self.scheduler == CustomCosineLR
        step_before_train = hasattr(self.scheduler, "step_before_train") and self.scheduler.step_before_train
        if step_before_train:
            try:
                self.scheduler.step(epoch=epoch)
                #if self.cfg['alternate']:
                #    self.scheduler1.step(epoch=epoch)
            except:
                self.scheduler.step()
                #if self.cfg['alternate']:
                #    self.scheduler1.step()
        gradient=0
        n=0
        for _it, (inputs, targets) in enumerate(dataloader):
            n+=1
            if self.debug_max_iters is not None and _it >= self.debug_max_iters:
                break

            if before_callback is not None:
                before_callback(self.model, epoch, total_iters, iters_per_epoch)
            
            #global prune stragety
            if self.cfg['globalprune']:
                threshold=self.model.get_threshold(self.model.sparsity)
                self.model.set_threshold(threshold)

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            if self.cfg['L1']:
                L1=0.0
                for p in self.model.parameters():
                    if (hasattr(p, 'is_score') and p.is_score):
                        L1=L1+torch.sum(torch.sigmoid(p).abs())
                loss=loss+L1*self.cfg['L1_lambda']
            loss.backward()

            if self.cfg['gradientflow']:                
                for p in self.model.parameters():
                    if not (hasattr(p, 'is_score') and p.is_score):
                        gradient+=p.grad.clone().detach().norm().item()

            self.optimizer.step()


            _, predicted = outputs.max(1)
            total_count += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            mean_loss = loss.item() * targets.size(0)
            results.append({
                'mean_loss': mean_loss,
                })

            total_loss += mean_loss
            total_iters += 1

            if after_callback is not None:
                after_callback(self.model, epoch, total_iters, iters_per_epoch)

        if not step_before_train:
            try:
                self.scheduler.step(epoch=epoch)
            except:
                self.scheduler.step()
        self.model.eval()
        lrs=""
        for i,p in enumerate(self.optimizer.param_groups):
            lrs+=" lr "+str(i)+": "
            lrs+=str(p['lr'])

        print("epoch",epoch,"gradients:",gradient/n,lrs,'loss:',total_loss / total_count)
        return {
                'iterations': total_iters,
                'per_iteration': results,
                'loss': total_loss / total_count,
                'moving_accuracy': correct / total_count
                }

    def evaluate(self, dataset_type='val'):
        self.model.eval()
        self.model.set_train(False)
                   #global prune stragety
        if self.cfg['globalprune']:
            threshold=self.model.get_threshold(self.model.sparsity)
            self.model.set_threshold(threshold)
        batch_size = self.cfg['batch_size_eval']
        num_workers = self.cfg['num_workers']
        if dataset_type == 'val':
            dataloader = DataLoader(self.val_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers)
        elif dataset_type == 'test':
            dataloader = DataLoader(self.test_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers)
        else:
            raise NotImplementedError

        results = []
        total_count = 0
        total_loss = 0.
        correct = 0
        for _it, (inputs, targets) in enumerate(dataloader):
            if self.debug_max_iters is not None and _it >= self.debug_max_iters:
                break

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                _, predicted = outputs.max(1)
                total_count += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                total_loss += loss.item() / targets.size(0)
        return {
                'loss': total_loss / total_count,
                'accuracy': correct / total_count,
                }

    def _get_datasets(self):
        dataset_dir = self.cfg['dataset_dir']
        max_size = self.cfg['max_train_dataset_size']
        dataset_download = self.cfg['dataset_download']
        dataset_classname = self.dataset_cfg['class']
        data_type = self.dataset_cfg['data_type']

        if dataset_classname in ['CIFAR10', 'CIFAR100', 'MNIST']:
            dataset_class = getattr(torchvision.datasets, dataset_classname)
        elif dataset_classname in ['ImageNet']:
            dataset_class = getattr(utils.datasets, dataset_classname)
        else:
            raise NotImplementedError

        if data_type == 'image':
            image_size = self.dataset_cfg['image_size']
            train_val_split = self.dataset_cfg['train_val_split']

            transform_train = self._create_transform(image_size, train=True)
            transform_val = self._create_transform(image_size, train=False)

            trainval_dataset = dataset_class(dataset_dir,
                                             train=True,
                                             transform=None,
                                             download=dataset_download)

            size = len(trainval_dataset)
            val_size = int(size * train_val_split)
            train_size = min(size - val_size,
                             max_size if max_size is not None else size)
            gen = torch.Generator()
            gen.manual_seed(777)
            train_subset, val_subset, _ = random_split(trainval_dataset,
                                                [train_size, val_size, size-(train_size+val_size)],
                                                generator=gen)
            self.outman.print('Train/val dataset size:', size)
            self.outman.print('Train dataset size:', len(train_subset),
                         ', Val dataset size:', len(val_subset))

            train_dataset = SubsetDataset(train_subset, transform=transform_train)
            val_dataset = SubsetDataset(val_subset, transform=transform_val)
            test_dataset = dataset_class(dataset_dir,
                                         train=False,
                                         transform=transform_val,
                                         download=dataset_download)
            val_dataset=test_dataset
        else:
            raise NotImplementedError

        return train_dataset, val_dataset, test_dataset

    def _create_transform(self, image_size, train=False):
        dataset_class = self.dataset_cfg['class']
        if dataset_class =='ImageNet':
            norm_param = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif dataset_class=='CIFAR10':
            norm_param = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
        elif dataset_class=='CIFAR100':
            norm_param = dict(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        jitter_param = dict(brightness=0.4, contrast=0.4, saturation=0.4)

        if train and self.train_augmentation:
            if dataset_class in ['MNIST']:
                train_transform = Compose([
                                Resize((image_size, image_size)),
                                ToTensor(),
                                Normalize((0.1307,), (0.3081,)),
                               ])
            else:
                train_transform = Compose([
                                RandomResizedCrop((image_size, image_size)),
                                RandomHorizontalFlip(),
                                ToTensor(),
                                Normalize(**norm_param),
                               ])
            if self.cfg['padding_before_crop']:  # this should be used in CIFAR-10 training
                train_transform.transforms.insert(0, Pad(4))
            return train_transform
        else:
            if dataset_class in ['ImageNet']:
                return Compose([Resize(256),
                                CenterCrop(image_size),
                                ToTensor(),
                                Normalize(**norm_param)
                               ])
            elif dataset_class in ['CIFAR10', 'CIFAR100']:
                return Compose([Resize((image_size, image_size)),
                                ToTensor(),
                                Normalize(**norm_param)
                               ])
            elif dataset_class in ['MNIST']:
                return Compose([Resize((image_size, image_size)),
                                ToTensor(),
                                Normalize((0.1307,), (0.3081,)),
                               ])
            else:
                raise NotImplementedError

    def _get_model(self, model_cfg=None):
        if model_cfg is None:
            model_cfg = self.model_cfg

        if model_cfg['class'] == 'ResNet':
            model = ResNet(self.dataset_cfg, model_cfg, self.cfg)
        elif model_cfg['class'] == 'Conv6':
            model = Conv6(self.dataset_cfg, model_cfg, self.cfg)
        else:
            raise NotImplementedError

        if self.data_parallel:
            gpu_ids = list(range(self.cfg['num_gpus']))
            return DataParallel(model, gpu_ids)
        else:
            return model

    def _get_optimizer(self,name,lr=None):
        optim_name = self.cfg['optimizer']

        if name == 'score_only':
            #lr = self.cfg['lr']
            if lr!=None:
                lr=lr
            else:
                lr = self.cfg['lr']

            weight_decay = self.cfg['weight_decay']
            params = [param for param in self.model.parameters() if hasattr(param, 'is_score') and param.is_score]
            return self._new_optimizer(optim_name, params, lr, weight_decay)
        elif name == 'normal':
            if lr!=None:
                lr=lr
            else:
                lr = self.cfg['lr']
            weight_decay = self.cfg['weight_decay']
            params = [param for param in self.model.parameters() if not (hasattr(param, 'is_score') and param.is_score)]
            return self._new_optimizer(optim_name, params,lr, weight_decay)
        else:
            raise NotImplementedError

    def _get_criterion(self):
        return nn.CrossEntropyLoss()

    def _new_optimizer(self, name, params, lr, weight_decay, momentum=0.9):
        #zprint(len(params))        
        if name == 'AdamW':
            if len(params)==2:
                return torch.optim.Adam([
                {'params': params[0],'lr':lr[0]},
                {'params':params[1], 'lr': lr[1]}])
            else:
                return torch.optim.Adam(params, lr=lr)
        elif name == 'SGD':
            if len(params)==2:
                print(name,":2")
                return torch.optim.SGD([
                {'params': params[0],'lr':lr[0]},
                {'params':params[1], 'lr': lr[1]}],momentum=self.cfg['sgd_momentum'], weight_decay=weight_decay)
            else:
                return torch.optim.SGD(params, lr=lr,momentum=self.cfg['sgd_momentum'], weight_decay=weight_decay)
        else:
            raise NotImplementedError

    def _get_scheduler(self,optimizer=None):
        class null_scheduler(object):
            def __init__(self, *args, **kwargs):
                return
            def step(self, *args, **kwargs):
                return
            def state_dict(self):
                return {}
            def load_state_dict(self, dic):
                return

        if self.cfg['lr_scheduler'] is None:
            return null_scheduler()
        elif self.cfg['lr_scheduler'] == 'CustomCosineLR':
            #total_epoch = self.cfg['finetuneEpoch']
            total_epoch = self.cfg['epoch']
            init_lr = self.cfg['lr']
            warmup_epochs = self.cfg['warmup_epochs']
            ft_epochs = self.cfg['finetuning_epochs']
            ft_lr = self.cfg['finetuning_lr']
            if optimizer!=None:
                return CustomCosineLR(optimizer, init_lr, total_epoch, warmup_epochs, ft_epochs, ft_lr)
            else:
                return CustomCosineLR(self.optimizer, init_lr, total_epoch, warmup_epochs, ft_epochs, ft_lr)
        elif self.cfg['lr_scheduler'] == 'MultiStepLR':
            if optimizer !=None:
                return MultiStepLR(optimizer, milestones=self.cfg['lr_milestones'], gamma=self.cfg['multisteplr_gamma'])
            else:
                return MultiStepLR(self.optimizer, milestones=self.cfg['lr_milestones'], gamma=self.cfg['multisteplr_gamma'])
        else:
            raise NotImplementedError

