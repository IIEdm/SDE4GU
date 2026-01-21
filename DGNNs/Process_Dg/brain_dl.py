import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import utils as u
import torch
from torch_geometric.utils import stochastic_blockmodel_graph
from tqdm import tqdm
from .sbm_seed import stochastic_blockmodel_graph_with_seed
# 大脑链接图  节点表示小的cube
class Brain_Dataset():
	def __init__(self,args, ood_mode=None):  
		if not isinstance(args.brain_data_args, u.Namespace):
			args.brain_data_args = u.Namespace(args.brain_data_args)
		self.num_classes = 10
		# data = np.load('data/brain/Brain_5000nodes.npz')
		data = np.load(os.path.join(args.brain_data_args.folder,args.brain_data_args.file))
		# print(data.files)#这是npzfile对象  直接用 data["adjs"]访问邻接矩阵  用data["attmats"]问节点标签  用data["labels"]访问标签
		print(data["adjs"].shape, data["attmats"].shape, data["labels"].shape) 
		# (12, 5000, 5000) (5000, 12, 20) (5000, 10)  12是时间步数  20是节点特征维度  节点类别是静态的 

		self.nodes_labels_times = self.load_node_labels(data) # 独热转整数  这个其实是label
		# print(f"self.nodes_labels_times.shape:{self.nodes_labels_times.shape}") #5000*2  nid label
		# print(f"self.nodes_labels_times[:,1].max() & min:{self.nodes_labels_times[:,1].max(), self.nodes_labels_times[:,1].min()}")
		#print(f"self.nodes_labels_times:{self.nodes_labels_times}") 

		self.edges = self.load_transactions(data) # edges是一个字典
		print(f"self.edges['idx'].shape:{self.edges['idx'].shape}")  # ([1955488, 3])
		print(f"self.edges['vals'].shape:{self.edges['vals'].shape}")  # ([1955488])

		# idx 是三列 起点；终点；时间戳   val是权重都是1 别的数据集可能有多次交易等 会多给val  边分类中就不是这样  val是正负边的数量差  一般是1  个别数据集不是
		self.num_nodes, self.nodes_feats = self.load_node_feats(data) # 这个是特征矩阵 12 5000 20 
		print(f"len:self.nodes_feats:{len(self.nodes_feats)}*{self.nodes_feats[0].shape}")
		self.num_non_existing = 12* self.num_nodes ** 2 - len(self.edges)
		
		if ood_mode == "SM":  # SM FI
			n = self.num_nodes
			d = self.edges['idx'].shape[0]/self.num_nodes/(self.num_nodes-1)/(self.max_time-self.min_time+1)
			num_blocks = self.num_classes
			p_ii, p_ij = 0.5 * d, 1.5 * d  
			block_size = n // num_blocks
			block_sizes = [block_size for _ in range(num_blocks-1)] + [block_size + n % block_size] 
			edge_probs = torch.ones((num_blocks, num_blocks)) * p_ij
			edge_probs[torch.arange(num_blocks), torch.arange(num_blocks)] = p_ii 
			
			SM_edges = []
			save_path = f"./DL/data_SM/{args.data}_edges.pt"
			if os.path.exists(save_path):
				self.edges = torch.load(save_path)
				print(f"Loaded existing edges from {save_path}")
			else:	
				for timestep in tqdm(range(self.min_time, int(self.max_time) + 1), desc=f"Generating SM Graphs for {args.data}"):

					edge_index = stochastic_blockmodel_graph_with_seed(block_sizes, edge_probs,time_seed=timestep)  
					cols_time = torch.tensor([timestep] * edge_index.size(1))
					edge_index_time = torch.cat([edge_index.T, cols_time.unsqueeze(1)], dim=1)
					SM_edges.append(edge_index_time)
					
					
				self.edges["idx"] = torch.cat(SM_edges, dim=0)
				self.edges["vals"] = torch.ones(self.edges["idx"].size(0))
				torch.save(self.edges, save_path)
				print(f"Edges saved to {save_path}")

		elif ood_mode == "FI":
			FI_nodes_feats = []
			for timestep in range(self.min_time,int(self.max_time)+1):
				x = self.nodes_feats[timestep] # [5000, 20]
				n = self.num_nodes
				idx = torch.randint(0, n, (n, 2)) 
				weight = torch.rand(n).unsqueeze(1) 
				x_new = x[idx[:, 0]] * weight + x[idx[:, 1]] * (1 - weight)
				FI_nodes_feats.append(x_new)
			print(f"FI_nodes_feats{len(FI_nodes_feats)}*{FI_nodes_feats[0].shape}:::{FI_nodes_feats[0][:3]}")
			self.nodes_feats = FI_nodes_feats


	def load_node_feats(self, data):
		features = data['attmats']
		nodes_feats = []
		for i in range(12):
			nodes_feats.append(torch.FloatTensor(features)[:, i]) #这个操作是 取这个时间的所有特征  相当变成 12 5000 20

		self.num_nodes = 5000
		print(self.num_nodes)
		self.feats_per_node = len(nodes_feats[0])

		return self.num_nodes, nodes_feats  # ？ 返回了两个一样的东西


	def load_node_labels(self, data):
		lcols = u.Namespace({'nid': 0,
							 'label': 1}) #字典变属性


		labels = data['labels']

		nodes_labels_times =[]
		for i in range(len(labels)):
			label = labels[i].tolist().index(1)
			nodes_labels_times.append([i, label])

		nodes_labels_times = torch.LongTensor(nodes_labels_times)

		return nodes_labels_times

	def load_transactions(self, data):
		adj = data['adjs']

		tcols = u.Namespace({'source': 0,
							 'target': 1,
							 'time': 2})
		
		data = []

		t = 0
		for graph in adj:
			for i in range(len(graph)):
				temp = np.concatenate((np.ones(len(np.where(graph[i] == 1)[0])).reshape(-1,1)*i,np.where(graph[i] == 1)[0].reshape(-1,1), np.ones(len(np.where(graph[i] == 1)[0])).reshape(-1,1) * t) , 1).astype(int).tolist()
				data.extend(temp)
			t += 1

		data= torch.LongTensor(data)

		self.max_time = torch.FloatTensor([11])
		self.min_time = 0

		print(data.size(0))

		return {'idx': data, 'vals': torch.ones(data.size(0))}

		# if ood_mode == "SM":  # SM FI LO
		# 	pass
		# elif ood_mode == "FI":
		# 	pass
		# elif ood_mode == "LO":
		# 	pass
		# if ood_mode == "SM":  # SM FI LO
		# 	