import torch
import torch.nn as nn
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import ToTensor, Resize, Compose, ColorJitter, RandomResizedCrop, RandomHorizontalFlip, Normalize, CenterCrop, Pad
from torch.nn import DataParallel
from torch.optim.lr_scheduler import MultiStepLR
#from  torch_geometric.nn.models import GIN
#from models.utils import DataParallel
from utils.subset_dataset import SubsetDataset, random_split
import utils.datasets
from utils.schedulers import CustomCosineLR
#from models.networks.resnet import ResNet
#from models.networks.convs import Conv6
from models.networks.GCN import GCN
#from models.networks.SGC import SGC

#from models.networks.basic_gnn import GIN 
from models.networks.gin_net import GINNet
from models.networks.GAT import GAT
from models.Dataloader import load_data, load_ogbn
import dgl
import random
import numpy as np
from ogb.nodeproppred import Evaluator
from models.utils_gin import load_data_gin, load_adj_raw_gin
from models.calibration import expected_calibration_error
from models.ood import *
import sklearn.metrics as sk
from torch_geometric.utils import to_scipy_sparse_matrix
from models.networks.gat_net import GATNet
from sklearn.manifold import TSNE
from numpy import reshape
import seaborn as sns
import pandas as pd  



def get_sparsity(sparsity,current_epoches,start_epoches,end_epoches):
    sparsity=sparsity-sparsity*(1-(current_epoches-start_epoches)*1.0/(end_epoches-start_epoches))
    return sparsity
    

class SupervisedLearning(object):
    def __init__(self, outman,args, device):
        self.args=args
        #print(self.args)
        self.outman = outman
        #self.cfg = cfg
        self.device = device
        self.data_parallel = torch.cuda.is_available()

        self.model_cfg=args.type_model
        if self.model_cfg=='GIN' or self.model_cfg=='dgl_GAT':          
            if self.args.dataset=="ogbn-arxiv":
                self.data, self.split_idx = load_ogbn(self.args.dataset,self.args.sampling,self.args.samplingtype)

                self.idx_train=self.split_idx['train']
                self.idx_test=self.split_idx['test']
                self.idx_val=self.split_idx['valid']
                print("idx train shape",self.idx_train.shape)
                adj=to_scipy_sparse_matrix(self.data.edge_index).tocoo()

                self.g=dgl.DGLGraph()
                node_num=self.data.x.size(0)
                class_num=self.data.y.numpy().max()+1
                self.g.add_nodes(node_num)
                self.g.add_edges(adj.row,adj.col)

                self.g=self.g.to(device)
                self.features=self.data.x.to(device)
                self.labels=self.data.y.squeeze().to(device)
                self.evaluator = Evaluator(name='ogbn-arxiv')
            else:

                adj, features, labels, idx_train, idx_val, idx_test = load_data_gin(args.dataset.lower())
                adj = load_adj_raw_gin(args.dataset.lower())
                self.idx_train=idx_train
                self.idx_val=idx_val
                self.idx_test=idx_test
                node_num = features.size()[0]
                class_num = labels.numpy().max() + 1

                self.g = dgl.DGLGraph()
                self.g.add_nodes(node_num)
                adj = adj.tocoo()
                self.g.add_edges(adj.row, adj.col)
                self.g=self.g.to('cuda:0')
                self.features = features.to(self.device)
                self.labels = labels.to(self.device)

        else:
            if self.args.dataset=='ogbn-arxiv':
                self.data, self.split_idx = load_ogbn(self.args.dataset,self.args.sampling,self.args.samplingtype)
                self.data.to(self.device)

                self.idx_train=self.split_idx['train']
                self.idx_test=self.split_idx['test']
                self.idx_val=self.split_idx['valid']

                self.evaluator = Evaluator(name='ogbn-arxiv')
            else:
                self.data=load_data(self.args.dataset,self.args.random_seed,self.args.attack,self.args.attack_eps)
                if args.auroc:
                    self.data,class_num=get_ood_split(self.data.cpu(),ood_frac_left_out_classes=0.4)
                    self.args.num_classes=class_num
                    self.data.cuda()
                
        self.model = self._get_model().to(self.device)
        if self.model_cfg=='GCN':
            print("the first 27 layers are fixed!")
            if self.args.num_layers==32:
                for i in range(0,5):
                    self.model.module.layers_GCN[i].lin.weight_score.sparsity=self.args.linear_sparsity
        self.optimizer = self._get_optimizer(self.args.train_mode,self.model)
        
        self.criterion = self._get_criterion()
        self.scheduler = self._get_scheduler()
    def L1_norm(self):
        reg_loss=0.0
        ratio=0.0
        n=0
        for param in self.model.parameters():
            if hasattr(param, 'is_score') and param.is_score:
                n+=1
                #print(param.shape)
                param=param.squeeze()
                assert len(param.shape)==2 or len(param.shape)==1
                #param=torch.sigmoid(param)
                reg_loss=reg_loss+torch.mean(torch.sigmoid(param))
        return reg_loss,None


    def train(self, epoch, total_iters, before_callback=None, after_callback=None):
     
        self.model.train()
        if self.args.sparse_decay:
            if epoch<(self.args.epochs/2.0):
                sparsity=get_sparsity(self.args.linear_sparsity,epoch,0,self.args.epochs/2)
            else:
                sparsity=self.args.linear_sparsity
        else:
            sparsity=self.args.linear_sparsity
        
        results = []
        total_count = 0
        total_loss = 0.
        correct = 0
        iters_per_epoch=1
        step_before_train = hasattr(self.scheduler, "step_before_train") and self.scheduler.step_before_train
        if step_before_train:
            try:
                self.scheduler.step(epoch=epoch)
            except:
                self.scheduler.step()
        for _ in range(1):
            if before_callback is not None:
                before_callback(self.model, epoch, total_iters, iters_per_epoch)
            #print("shape",self.data.edge_index.shape)
            if self.model_cfg=='GIN':
                outputs = self.model(self.g, self.features, 0, 0,sparsity=sparsity)
                loss = self.criterion(outputs[self.idx_train], self.labels[self.idx_train]) 
                targets=self.labels[self.idx_train]
            elif  self.model_cfg=='dgl_GAT':
                outputs = self.model(self.g, self.features, sparsity=sparsity)
                loss = self.criterion(outputs[self.idx_train], self.labels[self.idx_train]) 
                targets=self.labels[self.idx_train]
            else:
                outputs=self.model(self.data.x,self.data.edge_index,sparsity=sparsity,epoch=epoch)
                if self.args.dataset=='ogbn-arxiv':
                    targets=self.data.y.squeeze()[self.idx_train]
                    loss=self.criterion(outputs[self.idx_train],targets)
                else:
                    targets=self.data.y[self.data.train_mask]
                    loss=self.criterion(outputs[self.data.train_mask], targets)
            if self.args.train_mode=='score_only':
                L1_loss,_=self.L1_norm()
                
            else:
                L1_loss=torch.tensor(0.0).to(outputs.device)
            loss=loss+self.args.weight_l1*L1_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.args.dataset=='ogbn-arxiv':
                _, predicted = outputs[self.idx_train].max(1)
            else:
                if self.model_cfg=='GIN' or self.model_cfg=='dgl_GAT':
                    _, predicted = outputs[self.idx_train].max(1)
                else:
                    _, predicted = outputs[self.data.train_mask].max(1)
            total_count += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            mean_loss = loss.item() 
            print("loss",mean_loss,"acc",correct / total_count)
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
        return {
                'iterations': total_iters,
                'per_iteration': results,
                'loss': total_loss / total_count,
                'moving_accuracy': correct / total_count,
                "L1_loss":L1_loss.item()
                }
    def get_ece(self):
        self.model.eval()
        if self.model_cfg=='GIN':
            labels=self.labels
            logits=self.model(self.g,self.features, 0, 0)
            yp=torch.softmax(logits,-1)
            ece=expected_calibration_error(labels[self.idx_test].cpu().detach().numpy(),yp[self.idx_test].cpu().detach().numpy())
            #_,indices=logits.max(1)
        else:
            labels=self.data.y
            logits=self.model(self.data.x,self.data.edge_index)
            #_, indices = torch.max(logits, dim=1)
            yp=torch.softmax(logits,-1)
            ece=expected_calibration_error(labels[self.data.test_mask].cpu().detach().numpy(),yp[self.data.test_mask].cpu().detach().numpy())
        return ece
    def plot_tsne(self,path):
        file_path=path+self.args.dataset+"_"+self.model_cfg+"_"+str(self.args.random_seed)+"_"+str(self.args.linear_sparsity)+"_"+self.args.train_mode+"_"+str(self.args.num_layers)+".jpg"
        self.model.eval()
        sns.set_style("white")
        tsne=TSNE(n_components=2,verbose=1,random_state=123,perplexity=50)
        if self.model_cfg=='GIN':
            labels=self.labels
            logits=self.model(self.g,self.features, 0, 0)
            z=tsne.fit_transform(logits[self.idx_test].cpu().detach().numpy())
            y=labels[self.idx_test].cpu().detach().numpy()
            
            #yp=torch.softmax(logits,-1)
            #ece=expected_calibration_error(labels[self.idx_test].cpu().detach().numpy(),yp[self.idx_test].cpu().detach().numpy())
            #_,indices=logits.max(1)
        else:
            labels=self.data.y
            logits=self.model(self.data.x,self.data.edge_index)
            z=tsne.fit_transform(logits[self.data.test_mask].cpu().detach().squeeze().numpy())
            y=labels[self.data.test_mask].cpu().detach().squeeze().numpy()
        df=pd.DataFrame()
        #print("y shape",y.shape,"z shape",z.shape)
        #df['y']=y
        #df['comp-1']=z[:,0]
        #df['comp-2']=z[:,1]
        length=int(z[:,0].shape[0]*1)
        all_len=z[:,0].shape[0]
        
        indexes=np.arange(all_len)
        np.random.shuffle(indexes)
        samples_indexes=indexes[:length]
        #print("z shape",z.shape)
        x_samples=z[:,0][samples_indexes]
        y_samples=z[:,1][samples_indexes]
        hue_samples=y[samples_indexes].tolist()
        
        df['y']=y[samples_indexes]
        df['comp-1']=x_samples
        df['comp-2']=y_samples
        y_set=set(y[samples_indexes])
        print(len(list(y_set)))
        #np.save(file_path[:-4]+"_z.npy",logits[self.idx_test].cpu().detach().squeeze().numpy())
        #np.save(file_path[:-4]+"_y.npy",y)
        #print("x shape",x_samples.shape)
        #print("y shape",y_samples.shape)
        #print(len(hue_samples ))

        fig=sns.scatterplot(x=x_samples, y=y_samples, hue=hue_samples,palette=sns.color_palette(n_colors= len(list(y_set))),data=df,s=100,legend=False,linewidth=0.5)
        fig.set(xticklabels=[],yticklabels=[])
        fig.set(xlabel=None,ylabel=None)
        fig.tick_params(bottom=False,left=False,pad=0)
        sns.despine(top=True, right=True, left=True, bottom=True, trim=True)
        scatter=fig.get_figure()
        scatter.tight_layout()
        scatter.savefig(file_path,pad_inches=0.0,dpi=600,bbox_inches='tight')  
    def get_roc(self):
        
        self.model.eval()
        if self.model_cfg=='GIN':
            labels=self.labels
            logits=self.model(self.g,self.features, 0, 0)
            logits=torch.softmax(logits,dim=-1)
            #_,indices=logits.max(1)
            #print("total acc",indices.eq(self.labels).sum().item()/labels.size(0))
            #val_idx=torch.tensor(self.idx_val).long()
            #test_idx=torch.tensor(self.idx_test).long()
            #train_idx=torch.tensor(self.idx_train).long()
        else:
            labels=self.data.y
            logits=self.model(self.data.x,self.data.edge_index)
            logits=torch.softmax(logits,dim=-1)
            #_, indices = torch.max(logits, dim=1)
            
        # in distribution
        ind_scores,_=logits[self.data.id_test_mask].max(dim=1)
        ind_scores=ind_scores.cpu().detach().numpy()
        ind_labels = np.zeros(ind_scores.shape[0])
        ind_scores=ind_scores*-1
        #ind_scores = np.max(y_pred_ind, 1)
    
        # out of distribution
        #y_pred_ood, _ = extract_prediction(out_loader, model, args)
        ood_scores,_=logits[self.data.ood_test_mask].max(dim=1)
        ood_scores=ood_scores.cpu().detach().numpy()
        ood_labels = np.ones(ood_scores.shape[0])
        #ood_scores = np.max(y_pred_ood, 1)
        ood_scores=ood_scores*-1
    
        labels = np.concatenate([ind_labels, ood_labels])
        scores = np.concatenate([ind_scores, ood_scores])
    
        auroc = sk.roc_auc_score(labels, scores)
        print('* AUROC = {}'.format(auroc))
        return auroc        
        
    def evaluate(self):
        #print("id val",id(self.model))
        self.model.eval()
        if self.model_cfg=='GIN':
            labels=self.labels
            logits=self.model(self.g,self.features, 0, 0)
            _,indices=logits.max(1)
        elif self.model_cfg=='dgl_GAT':
            labels=self.labels
            logits=self.model(self.g,self.features)
            _,indices=logits.max(1)
        else:
            labels=self.data.y
            logits=self.model(self.data.x,self.data.edge_index)
            _, indices = torch.max(logits, dim=1)
        if self.args.dataset=='ogbn-arxiv':
            if self.model_cfg=='GIN' or self.model_cfg=='dgl_GAT':
                y_pred=logits.argmax(dim=-1,keepdim=True)
                acc_val = self.evaluator.eval({
                    'y_true': self.data.y.squeeze().unsqueeze(-1)[self.idx_val],
                    'y_pred': y_pred[self.idx_val],
                })['acc']
                acc_test = self.evaluator.eval({
                    'y_true': self.data.y.squeeze().unsqueeze(-1)[self.idx_test],
                    'y_pred': y_pred[self.idx_test],
                })['acc']
            else:
                y_pred = logits.argmax(dim=-1, keepdim=True)
                acc_val = self.evaluator.eval({
                    'y_true': self.data.y.squeeze().unsqueeze(-1)[self.split_idx['valid']],
                    'y_pred': y_pred[self.split_idx['valid']],
                })['acc']
                acc_test = self.evaluator.eval({
                    'y_true': self.data.y.squeeze().unsqueeze(-1)[self.split_idx['test']],
                    'y_pred': y_pred[self.split_idx['test']],
                })['acc']
        else:
            if self.model_cfg=='GIN':
                correct_val = torch.sum(indices[self.idx_val] == labels[self.idx_val])
                correct_test=torch.sum(indices[self.idx_test]==labels[self.idx_test])
                correct_train=torch.sum(indices[self.idx_train]==labels[self.idx_train])
                acc_train=correct_train.item()*1.0/len(self.idx_train)
                acc_val=correct_val.item() * 1.0 / len(self.idx_val)
                acc_test=correct_test.item()*1.0/len(self.idx_test)

            else:
                val_idx=self.data.val_mask
                test_idx=self.data.test_mask
                correct_val = torch.sum(indices[val_idx] == labels[val_idx])
                correct_test=torch.sum(indices[test_idx]==labels[test_idx])
                #correct_train=torch.sum(indices[train_idx]==labels[train_idx])
                #acc_train=correct_train.item()*1.0/train_idx.sum().item()
                acc_val=correct_val.item() * 1.0 / val_idx.sum().item()
                acc_test=correct_test.item()*1.0/test_idx.sum().item()
        return acc_val,acc_test


    def _get_model(self, model_cfg=None):
        if model_cfg is None:
            model_cfg = self.model_cfg

        if model_cfg=='GCN':
            model=GCN(self.args)
        elif model_cfg=='GAT':
            #model=GAT(self.args.num_feats,self.args.dim_hidden,self.args.num_layers)
            model=GAT(self.args)
        elif model_cfg=='GIN' or model_cfg =='dgl_GAT':
            #model=GIN(self.args.num_feats,self.args.dim_hidden,self.args.num_layers)
            #model=GIN(self.args.num_feats,self.args.dim_hidden,self.args.num_layers,args=self.args)
            if model_cfg=='dgl_GAT':
                model=GATNet(self.args,self.g)
            else:
                model=GINNet(self.args,self.g)

        elif model_cfg=="SGC":
            model=SGC(self.args)
        else:
            raise NotImplementedError

        if self.data_parallel:
            gpu_ids = list(range(self.args.num_gpus))
            return DataParallel(model)
        else:
            return model

    def _get_optimizer(self,mode,model):
        #print("mode",mode) 
        optim_name = self.args.optimizer
        #for name, param in model.named_parameters():
        #    print(name)
        if mode == 'score_only':
            lr=self.args.lr
            weight_decay=self.args.weight_decay

            params = [param for param in model.parameters()
                                if hasattr(param, 'is_score') and param.is_score]
            return self._new_optimizer(optim_name, params, lr, weight_decay)
        elif mode == 'normal':
            lr=self.args.lr
            weight_decay=self.args.weight_decay
            params = [param for param in self.model.parameters() if not (hasattr(param, 'is_score') and param.is_score)]
            #print(params)
            return self._new_optimizer(optim_name, params, lr, weight_decay)
        else:
            raise NotImplementedError

    def _get_criterion(self):
        return nn.CrossEntropyLoss()

    def _new_optimizer(self, name, params, lr, weight_decay, momentum=0.9):
        if name == 'Adam':
            return torch.optim.AdamW(params, lr=lr,weight_decay=weight_decay)
        elif name == 'SGD':
            return torch.optim.SGD(params, lr=lr,momentum=self.args.sgd_momentum, weight_decay=weight_decay)
        else:
            raise NotImplementedError

    def _get_scheduler(self):
        class null_scheduler(object):
            def __init__(self, *args, **kwargs):
                return
            def step(self, *args, **kwargs):
                return
            def state_dict(self):
                return {}
            def load_state_dict(self, dic):
                return

        if self.args.lr_scheduler is None:
            return null_scheduler()
        elif self.args.lr_scheduler == 'CustomCosineLR':
            total_epoch = self.args.epoch
            init_lr = self.args.lr
            warmup_epochs = self.args.warmup_epochs
            ft_epochs = self.args.finetuning_epochs
            ft_lr = self.args.finetuning_lr
            return CustomCosineLR(self.optimizer, init_lr, total_epoch, warmup_epochs, ft_epochs, ft_lr)
        elif self.args.lr_scheduler == 'MultiStepLR':
            return MultiStepLR(self.optimizer, milestones=self.args.lr_milestones, gamma=self.args.multisteplr_gamma)
        else:
            raise NotImplementedError

