import torch
import torch.nn.functional as F
from torch import nn
#from torch_geometric.nn import GCNConv
from models.networks.sparse_modules_graph import GCNConv


def percentile(t, q):
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values.item()

class pair_norm(torch.nn.Module):
    def __init__(self):
        super(pair_norm, self).__init__()

    def forward(self, x):
        col_mean = x.mean(dim=0)
        x = x - col_mean
        rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
        x = x / rownorm_mean
        return x


class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.cached = self.transductive = args.transductive
        self.layers_GCN = nn.ModuleList([])
        self.layers_bn = nn.ModuleList([])
        self.args=args
        self.layers_GCN.append(GCNConv(self.num_feats, self.dim_hidden, cached=self.cached,args=args))
        if self.type_norm == 'batch':
            self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden,momentum=args.bn_momentum,track_running_stats=args.bn_track_running_stats,affine=args.bn_affine))
        elif self.type_norm == 'pair':
            self.layers_bn.append(pair_norm())

        for _ in range(self.num_layers - 2):
            self.layers_GCN.append(
                GCNConv(self.dim_hidden, self.dim_hidden, cached=self.cached,args=args))

            if self.type_norm == 'batch':
                self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden,momentum=args.bn_momentum,track_running_stats=args.bn_track_running_stats,affine=args.bn_affine))
            elif self.type_norm == 'pair':
                self.layers_bn.append(pair_norm())
        self.layers_GCN.append(GCNConv(self.dim_hidden, self.num_classes, cached=self.cached,args=args))

        if self.type_norm == 'batch':
            self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden,momentum=args.bn_momentum,track_running_stats=args.bn_track_running_stats,affine=args.bn_affine))
        elif self.type_norm == 'pair':
            self.layers_bn.append(pair_norm())

        #self.optimizer = torch.optim.Adam(self.parameters(),
        #                                 lr=self.lr, weight_decay=self.weight_decay)
    def get_threshold(self,sparsity,epoch=None):
        local=[]
        for name, p in self.named_parameters():
            if hasattr(p, 'is_score') and p.is_score and p.sparsity==self.args.linear_sparsity:
                local.append(p.detach().flatten())
        local=torch.cat(local)
        threshold=percentile(local,sparsity*100)
        # if epoch!=None and (epoch+1)%50==0:
        #     print("sparsity",sparsity,"threshold",threshold)
        #     total_n=0.0
        #     total_re=0.0
        #     for name, p in self.named_parameters():
        #         if hasattr(p, 'is_score') and p.is_score and p.sparsity==self.args.linear_sparsity:
        #             mask=p.detach()<threshold
        #             mask=mask.float()
        #             total_re+=mask.sum().item()
        #             total_n+=mask.numel()
        #             print(name,":masked ratio",mask.sum().item()/mask.numel())
        #     print("total remove",total_re/total_n)
        
        return threshold    
    def forward(self, x, edge_index,sparsity=None,epoch=None):
        if sparsity is None:
            sparsity=self.args.linear_sparsity
        threshold=self.get_threshold(sparsity,epoch=epoch)
        # implemented based on DeepGCN: https://github.com/LingxiaoShawn/PairNorm/blob/master/models.py
        for i in range(self.num_layers - 1):
            #x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layers_GCN[i](x, edge_index,threshold)
            if self.type_norm in ['batch', 'pair']:
                x = self.layers_bn[i](x)
            x = F.relu(x)
        #x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers_GCN[-1](x, edge_index,threshold)
        return x
    def rerandomize(self,mode,la,mu):
        for m in self.modules():
            if type(m) is GCNConv:
                m.rerandomize(mode,la,mu)

