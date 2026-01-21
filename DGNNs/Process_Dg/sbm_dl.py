import os
import sys
sys.path.append(os.path.abspath('..'))
import torch
import utils as u
from .sbm_seed import stochastic_blockmodel_graph_with_seed

# 这个数据集 四列
class sbm_dataset():
    def __init__(self,args, ood_mode=None):
        assert args.task in ['link_pred'], 'sbm only implements link_pred'
        self.ecols = u.Namespace({'FromNodeId': 0,
                                  'ToNodeId': 1,
                                  'Weight': 2,
                                  'TimeStep': 3
                                })
        args.sbm_args = u.Namespace(args.sbm_args)

        #build edge data structure
        edges = self.load_edges(args.sbm_args)  # 子矩阵取法  ：取所有行，time取最后一列 当然取出来是一维张量  在torch中这个不做行列的严格区分
        timesteps = u.aggregate_by_time(edges[:,self.ecols.TimeStep], args.sbm_args.aggr_time) # 时间列减去最小值  然后除以聚合单位（一般就是除以1）
        print("edges[:,self.ecols.TimeStep]:",edges[:,self.ecols.TimeStep].shape)

        self.max_time = timesteps.max()
        self.min_time = timesteps.min()
        print ('TIME', self.max_time, self.min_time ) # 时间戳0to49  
        edges[:,self.ecols.TimeStep] = timesteps

        print(f"self.num_classes of weight before cluster:{edges[:,self.ecols.Weight].unique().size(0)}") #聚类之前也只有1类就是权重1
        edges[:,self.ecols.Weight] = self.cluster_negs_and_positives(edges[:,self.ecols.Weight]) #正数变1 负数变0 
        self.num_classes = edges[:,self.ecols.Weight].unique().size(0)
        print(f"self.num_classes of weight after cluster:{self.num_classes}")
        print(edges[:3],edges.size(0))
        self.edges = self.edges_to_sp_dict(edges)
        print(self.edges["idx"][:3],self.edges["vals"][:3])
        #random node features
        self.num_nodes = int(self.get_num_nodes(edges)) 
        self.feats_per_node = args.sbm_args.feats_per_node
        self.nodes_feats = torch.rand((self.num_nodes,self.feats_per_node))
        print(self.num_nodes,self.feats_per_node,len(self.nodes_feats))

        # self.num_non_existing = self.num_nodes ** 2 - edges.size(0)  #这个有问题 前面应该乘以一个50 后面是50个时间的点
        self.num_non_existing = 50*self.num_nodes ** 2 - edges.size(0)
        print("------:",self.num_non_existing)

    def cluster_negs_and_positives(self,ratings):
        pos_indices = ratings >= 0
        neg_indices = ratings < 0
        ratings[pos_indices] = 1
        ratings[neg_indices] = 0
        return ratings  

    def prepare_node_feats(self,node_feats):
        node_feats = node_feats[0]
        return node_feats

    def edges_to_sp_dict(self,edges):
        idx = edges[:,[self.ecols.FromNodeId,
                       self.ecols.ToNodeId,
                       self.ecols.TimeStep]]

        vals = edges[:,self.ecols.Weight]
        return {'idx': idx,
                'vals': vals}

    def get_num_nodes(self,edges):
        all_ids = edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]]
        num_nodes = all_ids.max() + 1
        return num_nodes

    def load_edges(self,sbm_args, starting_line = 1):
        file = os.path.join(sbm_args.folder,sbm_args.edges_file)
        with open(file) as f:
            lines = f.read().splitlines()
        edges = [[float(r) for r in row.split(',')] for row in lines[starting_line:]]
        edges = torch.tensor(edges,dtype = torch.long)
        return edges

    def make_contigous_node_ids(self,edges):
        new_edges = edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]]
        _, new_edges = new_edges.unique(return_inverse=True)
        edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]] = new_edges
        return edges

    def _add_laplacian_positional_encodings(self, pos_enc_dim):
        
        # Graph positional encoding v/ Laplacian eigenvectors
        self.train.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.train.graph_lists]
        self.val.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.val.graph_lists]
        self.test.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.test.graph_lists]

