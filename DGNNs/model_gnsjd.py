import utils_dg as u
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
from sgnn_dynamic import SGNNDynamic
from sgnn_dynamic_dgl import SGNNDynamicDgl

import torch.nn.functional as F
from torch_geometric.nn import GATConv

import dgl
import dgl.function as fn
import numpy as np

class GNSJD(torch.nn.Module):
    def __init__(self, transformer_args, gcn_args, activation, device='cpu', skipfeats=False, data=''):
        super().__init__()
        RT_args = u.Namespace({})

        feats = [gcn_args.feats_per_node,
                 gcn_args.layer_1_feats,
                 gcn_args.layer_2_feats]
        self.device = device
        self.skipfeats = skipfeats
        self.RT_layers = []
    
        self.drift_input_feats = gcn_args.layer_1_feats
        self.drift_hidden_feats = gcn_args.layer_2_feats
        self.drift_net = GATConv(self.drift_input_feats, self.drift_hidden_feats, heads=1, concat=True)
        self.drift_net.to(self.device)
        self._parameters = nn.ParameterList()

        for i in range(1,len(feats)):
            RT_args = u.Namespace({
                'in_feats' : feats[i-1],
                'out_feats1': feats[i],
                'sgnn_in_feats': feats[i-1] if i==1 else transformer_args.out_dim,
                'activation': activation,
                'filter_order': transformer_args.filter_order,
                'in_channels_sgnn': feats[i] if i==1 else transformer_args.out_dim,
                'out_channels_sgnn': transformer_args.out_channels_sgnn,
                'fc1_dim': transformer_args.fc1_dim,
                'pe_dim': transformer_args.pe_dim,
                'out_feats': transformer_args.out_dim,
                'num_heads': transformer_args.num_heads,
                'layer_norm': transformer_args.layer_norm,
                'batch_norm': transformer_args.batch_norm,
                'is_recurrent': transformer_args.is_recurrent,
                'sgwt_scales': transformer_args.sgwt_scales,
                'device': device,
                'concat_in_skipfeat': transformer_args.concat_in_skipfeat,
                'skip_in_feat': transformer_args.skip_in_feat,
                'use_spatial_feat_in_lpe': transformer_args.use_spatial_feat_in_lpe,
                'use_spectral_in_lpe': transformer_args.use_spectral_in_lpe,
                'num_filter_subspaces': transformer_args.num_filter_subspaces,
                'use_spatial_feat_in_rgt_ip': transformer_args.use_spatial_feat_in_rgt_ip,
                'skip_rgt_in_feat': transformer_args.skip_rgt_in_feat,
                'device': device,
                'aggregator': transformer_args.aggregator,
                'use_sgnn_dgl':  transformer_args.use_sgnn_dgl,
                'data': data,
                'use_pe_module': transformer_args.use_pe_module,
                'time': transformer_args.Time,
                'time_steps': transformer_args.Time_steps
            })

            rt_i = GNNLayer(RT_args)
            self.RT_layers.append(rt_i.to(self.device))
            self._parameters.extend(list(self.RT_layers[-1].parameters()))
        
        self._parameters.extend(list(self.drift_net.parameters()))

    def parameters(self):
        return self._parameters

    def forward(self, A_list, Nodes_list, nodes_mask_list, graph_list, A_list_u, pos_enc_list=[], return_filter_coeff_list=False):
        assert len(graph_list)==len(A_list)
        node_feats= Nodes_list[-1]

        for unit in self.RT_layers:
            if return_filter_coeff_list:
                Nodes_list, filter_coeff_list = unit(A_list, self.drift_net, A_list_u, graph_list, Nodes_list, return_filter_coeff_list=return_filter_coeff_list)
            else:
                Nodes_list = unit(A_list, self.drift_net, A_list_u, graph_list, Nodes_list)

        out = Nodes_list[-1]
        if self.skipfeats:
            out = torch.cat((out,node_feats), dim=1)  

        if return_filter_coeff_list:
            return out, filter_coeff_list
        else:
            return out


class GNNLayer(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        cell_args = u.Namespace({})
        cell_args.rows = args.sgnn_in_feats
        cell_args.cols = args.out_feats1

        self.evolve_weights = mat_GRU_cell(cell_args)

        self.activation = self.args.activation

        self.SGNN_init_weights = Parameter(torch.Tensor(self.args.sgnn_in_feats,self.args.out_feats1))
        self.deltaT = self.args.time / self.args.time_steps
        self.SGNN_proj_weights_fc_list = nn.ParameterList([Parameter(torch.Tensor(self.args.out_feats1,self.args.filter_order)) for _ in range(self.args.num_filter_subspaces)])

        if self.args.use_spectral_in_lpe:  
            if self.args.use_spatial_feat_in_lpe and not self.args.use_spatial_feat_in_rgt_ip:
                self.FFN_pe = nn.Linear(args.out_feats+args.out_feats1+1, args.pe_dim)
            else:
               self.FFN_pe = nn.Linear(args.out_feats+1, args.pe_dim) 
        else:
            if self.args.use_spatial_feat_in_lpe:
                self.FFN_pe = nn.Linear(args.out_feats1+1, args.pe_dim)
        self.activation_pe = torch.nn.RReLU()
        self.FFN_pe2 = nn.Linear(args.pe_dim*2, args.out_feats1)
        self.FFN_pe3 = nn.Linear(args.out_feats1, args.out_feats1)

        in_channels = args.in_channels_sgnn
        out_channels = args.out_channels_sgnn
        filter_order = args.filter_order
        if self.args.use_sgnn_dgl:
            self.sgnn_list = nn.ModuleList([SGNNDynamicDgl(in_channels, out_channels, filter_order, device=self.args.device) for _ in range(self.args.num_filter_subspaces)])
        else:
            self.sgnn_list = nn.ModuleList([SGNNDynamic(in_channels, out_channels, filter_order) for _ in range(self.args.num_filter_subspaces)])

        self.fc_pool = ['sum','mean'][1]
        self.fc1 = nn.Linear(self.args.filter_order, self.args.fc1_dim)
        self.fc2 = nn.Linear(self.args.fc1_dim, self.args.filter_order)
        self.reset_param(self.SGNN_init_weights)

        for SGNN_proj_weights_fc in self.SGNN_proj_weights_fc_list:
            self.reset_param(SGNN_proj_weights_fc)

        in_dim_rgt = self.args.out_feats
        out_dim_rgt = self.args.out_feats
        num_heads = self.args.num_heads
        layer_norm = self.args.layer_norm
        batch_norm = self.args.batch_norm
        is_recurrent = self.args.is_recurrent

        self.use_pe_module = self.args.use_pe_module
        if self.use_pe_module:
            self.FFN_rtg_in1 = nn.Linear(self.args.out_feats1+self.args.pe_dim, in_dim_rgt)
        else:
            self.FFN_rtg_in1 = nn.Linear(self.args.out_feats1, in_dim_rgt)

        
    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def reset_param_fc(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv,stdv)

    def forward(self, A_list, drift_net, A_list_u, G_list, node_embs_list, return_filter_coeff_list=False):

        SGNN_weights = self.SGNN_init_weights
        out_seq = []
        seq_len = len(A_list)
        for t,Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]
            if len(A_list_u[t]['idx'].shape)==3:
                edge_index = A_list_u[t]['idx'][0].T.to(self.args.device)
            elif len(A_list_u[t]['idx'].shape)==2:
                edge_index = A_list_u[t]['idx'].T.to(self.args.device)
            else:
                raise Exception("Unable to get edge index")
            graph = G_list[t] 
            
            SGNN_weights = self.evolve_weights(SGNN_weights)
            
            gnn_out0 = self.activation(Ahat.matmul(node_embs.matmul(SGNN_weights)))
            #print('gnn_out0 size is: ', gnn_out0.size())
            #print('node_embs size is: ', node_embs.size())
            #new_gnn_out0 = drift_net(node_embs, edge_index)
            #print('new_gnn_out0 size is: ', new_gnn_out0.size())
            node_embs_in1 = node_embs.matmul(SGNN_weights)

            if self.args.use_spectral_in_lpe:

                filter_coeff_list = []
                for SGNN_proj_weights_fc in self.SGNN_proj_weights_fc_list:
                    gnn_out1 = self.activation(Ahat.matmul(node_embs.matmul(SGNN_weights.matmul(SGNN_proj_weights_fc))))
                    if self.fc_pool=='mean':
                        filter_coeff = torch.mean(gnn_out1, dim=0)
                    elif self.fc_pool=='sum':
                        filter_coeff = torch.sum(gnn_out1, dim=0)
                    filter_coeff = self.activation(self.fc1(filter_coeff))
                    filter_coeff = self.fc2(filter_coeff)
                    filter_coeff_list.append(filter_coeff)
            node_embs = node_embs.matmul(SGNN_weights)
                
                for step in range(self.args.time_steps):
                    all_node_embs = []
                    for fc_idx, sgnn in enumerate(self.sgnn_list):
                        filter_coeff = filter_coeff_list[fc_idx]
                        for scale in self.args.sgwt_scales:
                            if self.args.use_sgnn_dgl:
                        
                                graph.ndata['node_embs_sgnn_in'] = node_embs
                                node_embs_cur_scale = sgnn(graph, filter_coeff, feature_name='node_embs_sgnn_in', scale=scale) * (1./max(1,scale)**(self.args.filter_order))
                            else:
                                node_embs_cur_scale = sgnn(node_embs, edge_index, filter_coeff, scale=scale) * (1./max(1,scale)**(self.args.filter_order))
                            all_node_embs.append(node_embs_cur_scale)
                    all_node_embs = torch.cat([ne.unsqueeze(1) for ne in all_node_embs], dim=1)
                    node_embs = torch.sum(all_node_embs, dim=1)
                    diffusion_term = node_embs * math.sqrt(self.deltaT) * torch.randn_like(node_embs)
                    if self.args.use_spatial_feat_in_rgt_ip:
                        node_embs = node_embs + gnn_out0*self.deltaT + diffusion_term
                       

            if t==0:
                state_vectors = gnn_out0 
            time_pe = (1./seq_len)*torch.ones((node_embs.shape[0],1)).to(self.args.device)
            if not self.args.use_spectral_in_lpe and not self.args.use_spatial_feat_in_lpe:
                node_embs_in_rgt = node_embs_in1
            else:
                if self.args.use_spectral_in_lpe:
                    if self.args.use_spatial_feat_in_lpe and not self.args.use_spatial_feat_in_rgt_ip:
                        pe = torch.cat((node_embs,gnn_out0,time_pe), dim=1)
                    else:
                        pe = torch.cat((node_embs,time_pe), dim=1)
                else:
                    if self.args.use_spatial_feat_in_lpe:
                        pe = torch.cat((gnn_out0,time_pe), dim=1)
                pe = self.FFN_pe(pe)
                pe_cos = torch.cos(pe)
                pe_sin = torch.sin(pe)
                pe = torch.cat((pe_cos,pe_sin), dim=1)
                pe = self.activation_pe(self.FFN_pe2(pe))
                pe = self.FFN_pe3(pe)
                if self.args.use_spectral_in_lpe:
                    if self.args.concat_in_skipfeat:
        
                        node_embs = torch.cat((node_embs_in1,node_embs), dim=-1)
                        node_embs_in_rgt = self.FFN_skipcat1(node_embs)
                    else:
                        node_embs_in_rgt = node_embs_in1 + node_embs
                else:
                    node_embs_in_rgt = node_embs_in1
                if self.use_pe_module:
                    node_embs_in_rgt = torch.cat((node_embs_in_rgt,pe), dim=1)
                node_embs_in_rgt = self.FFN_rtg_in1(node_embs_in_rgt)
            node_embs = node_embs_in_rgt

            out_seq.append(node_embs)

        if return_filter_coeff_list:
            return out_seq, filter_coeff_list
        else:
            return out_seq

class mat_GRU_cell(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Tanh())
        
        self.choose_topk = TopK(feats = args.rows,
                                k = args.cols)

    def forward(self,prev_Q):
        z_topk = prev_Q

        update = self.update(z_topk,prev_Q)
        reset = self.reset(z_topk,prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q

        

class mat_GRU_gate(torch.nn.Module):
    def __init__(self,rows,cols,activation):
        super().__init__()
        self.activation = activation
        self.W = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows,cols))

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,x,hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out

class TopK(torch.nn.Module):
    def __init__(self,feats,k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats,1))
        self.reset_param(self.scorer)
        
        self.k = k

    def reset_param(self,t):
        #Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv,stdv)

    def forward(self,node_embs,mask):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores + mask

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = u.pad_with_last_val(topk_indices,self.k)
            
        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
           isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1,1))

        #we need to transpose the output
        return out.t()

