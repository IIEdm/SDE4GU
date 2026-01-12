import torch

import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
from backbone import *
from SGNN import *
import numpy as np
from data_utils import get_rw_adj

class GNSD(nn.Module):
    
    def __init__(self, d, c, args):
        super(GNSD, self).__init__()
        
        self.encoder = SGNN(d, c, args)
        

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, flag, device):
        
        x = dataset.x.to(device)
        return self.encoder(x, flag, device)


    def propagation(self, e, edge_index, prop_layers=1, alpha=0.5):
        
        e = e.unsqueeze(1)
        N = e.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm = 1. / d[col]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        for _ in range(prop_layers):
            e = e * alpha + matmul(adj, e) * (1 - alpha)
        return e.squeeze(1)

    

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):
        

        edge_index_in, edge_weight_in = get_rw_adj(dataset_ind.edge_index, edge_weight=dataset_ind.edge_attr, norm_dim=1,
                                             fill_value=args.self_loop_weight,
                                             num_nodes=dataset_ind.num_nodes,
                                             dtype=dataset_ind.x.dtype)

        x_in, edge_index_in, edge_weight_in = dataset_ind.x.to(device), edge_index_in.to(device), edge_weight_in.to(device)

        self.encoder.ind_edge_index = edge_index_in
        self.encoder.ind_edge_weight = edge_weight_in
        
        logits_in = self.encoder(x_in, True, device)
        

        train_in_idx, train_ood_idx = dataset_ind.splits['train'], dataset_ood.node_idx

        # compute supervised training loss
        
        pred_in = F.log_softmax(logits_in[train_in_idx], dim=1)
        sup_loss = criterion(pred_in, dataset_ind.y[train_in_idx].squeeze(1).to(device))

        
        loss = sup_loss

        return loss
