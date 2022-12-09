import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.sparse_modules import SparseLinear,SparseParameter
import dgl

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
from models.networks.gat_layer import GATLayer
#from gnns.mlp_readout_layer import MLPReadout
import pdb

def percentile(t, q):
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values.item()

class GATNet(nn.Module):

    def __init__(self, args, graph):
        super().__init__()
        self.args=args
        #in_dim_node = net_params[0] # node_dim (feat is an integer)
        #hidden_dim = net_params[1]
        #out_dim = net_params[2]
        #n_classes = net_params[2]
        in_dim_node=args.num_feats
        hidden_dim=args.dim_hidden
        n_classes=args.num_classes
        out_dim=n_classes
        dropout=args.dropout

        #num_heads = 1
        #dropout = 0.6
        #n_layers = 1
        num_heads=args.heads
        dropout=args.dropout    
        n_layers=args.num_layers

        self.edge_num = graph.number_of_edges() + graph.number_of_nodes()
        self.graph_norm = False
        self.batch_norm = False
        self.residual = False
        self.dropout = dropout
        self.n_classes = n_classes
        
        #list of MLPs
        self.layers=nn.ModuleList()

        for l in range(n_layers):
            if l==0:
                glayer=GATLayer(in_dim_node, hidden_dim, num_heads,dropout, self.graph_norm, self.batch_norm, self.residual,args=args)
            elif l<n_layers-1:
                glayer=GATLayer(hidden_dim, hidden_dim, num_heads,dropout, self.graph_norm, self.batch_norm, self.residual,args=args)
            else:
                glayer=GATLayer(hidden_dim * num_heads, out_dim, 1, 0, self.graph_norm, self.batch_norm, self.residual,args=args)
        #self.layers.append(GATLayer(in_dim_node, hidden_dim, num_heads,dropout, self.graph_norm, self.batch_norm, self.residual) for _ in range(n_layers))
            self.layers.append(glayer)

        #self.adj_mask1_train = nn.Parameter(torch.ones(self.edge_num, 1), requires_grad=True)
        #self.adj_mask2_fixed = nn.Parameter(torch.ones(self.edge_num, 1), requires_grad=False)
    def get_threshold(self,sparsity):
        local=[]
        for name, p in self.named_parameters():
            if hasattr(p, 'is_score') and p.is_score:
                local.append(p.detach().flatten())
        local=torch.cat(local)
        threshold=percentile(local,sparsity*100)
        return threshold   
    def forward(self, g, h,sparsity=None):
        if sparsity is None:
            sparsity=self.args.linear_sparsity
        threshold=self.get_threshold(sparsity)

        # GAT
        for conv in self.layers:
            h = conv(g, h,  threshold=threshold)
            
        return h
    

class GATNet_ss(nn.Module):

    def __init__(self, net_params, num_par):
        super().__init__()

        in_dim_node = net_params[0] # node_dim (feat is an integer)
        hidden_dim = net_params[1]
        out_dim = net_params[2]
        n_classes = net_params[2]
        num_heads = 8
        dropout = 0.6
        n_layers = 1

        self.graph_norm = False
        self.batch_norm = False
        self.residual = False
        self.dropout = dropout
        self.n_classes = n_classes
        
        self.layers = nn.ModuleList([GATLayer(in_dim_node, hidden_dim, num_heads,
                                              dropout, self.graph_norm, self.batch_norm, self.residual) for _ in range(n_layers)])
        self.layers.append(GATLayer(hidden_dim * num_heads, out_dim, 1, 0, self.graph_norm, self.batch_norm, self.residual))
        self.classifier_ss = nn.Linear(hidden_dim * num_heads, num_par, bias=False)

    def forward(self, g, h, snorm_n, snorm_e):

        # GAT
        for conv in self.layers:
            h_ss = h
            h = conv(g, h, snorm_n)
            
        h_ss = self.classifier_ss(h_ss)

        return h, h_ss
 
