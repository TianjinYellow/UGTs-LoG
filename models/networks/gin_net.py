import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.sparse_modules import SparseLinear,SparseParameter
import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""

from models.networks.gin_layer import GINLayer, ApplyNodeFunc, MLP
import pdb

def percentile(t, q):
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values.item()
class GINNet(nn.Module):
    
    def __init__(self, args, graph):
        super().__init__()
        #in_dim = net_params[0]
        #hidden_dim = net_params[1]
        #n_classes = net_params[2]
        self.args=args
        in_dim=args.num_feats
        hidden_dim=args.dim_hidden
        n_classes=args.num_classes
        #dropout = 0.5
        dropout=args.dropout
        #self.n_layers = 2
        self.n_layers=args.num_layers
        
        self.edge_num = graph.all_edges()[0].numel()
        n_mlp_layers = 1               # GIN
        learn_eps = False              # GIN
        neighbor_aggr_type = 'mean' # GIN
        graph_norm = False      
        batch_norm = False
        residual = False
        self.n_classes = n_classes
        
        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        
        for layer in range(self.n_layers):
            if layer == 0:
                mlp = MLP(n_mlp_layers, in_dim, hidden_dim, hidden_dim,args=args)
            elif layer<(self.n_layers-1):
                mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim,args=args)
            else:
                mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, n_classes,args=args)
                
            
            self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                           dropout, graph_norm, batch_norm, residual, 0, learn_eps))

        # Linear function for output of each layer
        # which maps the output of different layers into a prediction score

        self.linears_prediction = SparseLinear(hidden_dim, n_classes, bias=False,args=args)
        #self.adj_mask1_train = nn.Parameter(torch.ones(self.edge_num, 1), requires_grad=True)
        self.adj_mask2_fixed = nn.Parameter(torch.ones(self.edge_num, 1), requires_grad=False)
    def get_threshold(self,sparsity):
        local=[]
        for name, p in self.named_parameters():
            if hasattr(p, 'is_score') and p.is_score:
                local.append(p.detach().flatten())
        local=torch.cat(local)
        threshold=percentile(local,sparsity*100)
        return threshold     
    def forward(self, g, h, snorm_n, snorm_e,sparsity=None):
        if sparsity is None:
            sparsity=self.args.linear_sparsity
        threshold=self.get_threshold(sparsity)        
        g.edata['mask'] = self.adj_mask2_fixed
        hidden_rep = []

        for i in range(self.n_layers):
            h = self.ginlayers[i](g, h, snorm_n,threshold=threshold)
            hidden_rep.append(h)

        # score_over_layer = (self.linears_prediction(hidden_rep[0]) + hidden_rep[1]) / 2
        score_over_layer = (self.linears_prediction(hidden_rep[-2],threshold=threshold) + hidden_rep[-1]) / 2

        return score_over_layer
      
"""
class GINNet_ss(nn.Module):
    
    def __init__(self, net_params, num_par):
        super().__init__()
        in_dim = net_params[0]
        hidden_dim = net_params[1]
        n_classes = net_params[2]
        dropout = 0.5
        self.n_layers = 2
        n_mlp_layers = 1               # GIN
        learn_eps = True              # GIN
        neighbor_aggr_type = 'mean' # GIN
        graph_norm = False      
        batch_norm = False
        residual = False
        self.n_classes = n_classes
        
        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        
        for layer in range(self.n_layers):
            if layer == 0:
                mlp = MLP(n_mlp_layers, in_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, n_classes)
                
            
            self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                           dropout, graph_norm, batch_norm, residual, 0, learn_eps))

        # Linear function for output of each layer
        # which maps the output of different layers into a prediction score

        self.linears_prediction = nn.Linear(hidden_dim, n_classes, bias=False)
        self.classifier_ss = nn.Linear(hidden_dim, num_par, bias=False)
        
    def forward(self, g, h, snorm_n, snorm_e):
        
        # list of hidden representation at each layer (including input)
        hidden_rep = []

        for i in range(self.n_layers):
            h = self.ginlayers[i](g, h, snorm_n)
            hidden_rep.append(h)

        score_over_layer = (self.linears_prediction(hidden_rep[0]) + hidden_rep[1]) / 2
        h_ss = self.classifier_ss(hidden_rep[0])

        return score_over_layer, h_ss
       
""" 