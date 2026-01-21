import os
import sys
sys.path.append(os.path.abspath('..'))
import utils as u
from .sbm_seed import stochastic_blockmodel_graph_with_seed
from tqdm import tqdm
import tarfile

import torch


class Uc_Irvine_Message_Dataset():
	def __init__(self,args, ood_mode=None):
		if not isinstance(args.uc_irc_args, u.Namespace):
			args.uc_irc_args = u.Namespace(args.uc_irc_args)
		

		tar_archive = os.path.join(args.uc_irc_args.folder, args.uc_irc_args.edges_file) 		
		print(f"=tar_archive:{tar_archive}")		
		self.edges = self.load_edges(args,tar_archive)

		print(f"self.edges['idx'].shape):{self.edges['idx'].shape}")  
		print(f"self.edges['idx'][:3]:{self.edges['idx'][:3]}")
		print(f"self.edges['vals'].shape):{self.edges['vals'].shape}")
		print(f"self.edges['vals'][:3]):{self.edges['vals'][:10]}")
		print(f"self.max_time:{self.max_time}")
		print(f"self.min_time:{self.min_time}")

		if ood_mode == "SM":  # SM FI 
			n = self.num_nodes
			d = self.edges['idx'].shape[0]/self.num_nodes/(self.num_nodes-1)/(self.max_time-self.min_time+1)
			num_blocks = 2
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

		elif ood_mode == "FI": # to be implemented
			print("UCI FI data will be got in tasker")
			# save_path = f"./DL/data_FI/{args.data}_edges.pt"
			# if os.path.exists(save_path):
			# 	self.edges = torch.load(save_path)
			# 	print(f"Loaded existing edges from {save_path}")
			# else:
			# 	FI_nodes_feats = []
			# 	for timestep in range(self.min_time,int(self.max_time)+1):
			# 		x = self.nodes_feats[timestep] # [5000, 20]
			# 		n = self.num_nodes
			# 		idx = torch.randint(0, n, (n, 2)) 
			# 		weight = torch.rand(n).unsqueeze(1) 
			# 		x_new = x[idx[:, 0]] * weight + x[idx[:, 1]] * (1 - weight)
			# 		FI_nodes_feats.append(x_new)
			# 	print(f"FI_nodes_feats{len(FI_nodes_feats)}*{FI_nodes_feats[0].shape}:::{FI_nodes_feats[0][:3]}")
			# 	self.nodes_feats = FI_nodes_feats	

	def load_edges(self,args,tar_archive):
		# data = u.load_data_from_tar(args.uc_irc_args.edges_file, 
		# 							tar_archive, 
		# 							starting_line=2,
		# 							sep=' ')
		with open(tar_archive, 'r') as f:
			for _ in range(2):
				f.readline()

    		#  直接读取数据并转换为张量
			data = torch.tensor([list(map(int, line.strip().split(','))) for line in f], dtype=torch.float64)
		cols = u.Namespace({'source': 0,
							 'target': 1,
							 'weight': 2,
							 'time': 3})

		data = data.long()
		print(f"=data:{data.size()}") #  (669362, 4)
		print(f"=data:{data[:5]}") #  (669362, 4)

		self.num_nodes = int(data[:,[cols.source,cols.target]].max())

		#first id should be 0 (they are already contiguous)
		data[:,[cols.source,cols.target]] -= 1

		#add edges in the other direction (simmetric)
		data = torch.cat([data,
						   data[:,[cols.target,
						   		   cols.source,
						   		   cols.weight,
						   		   cols.time]]],
						   dim=0)

		data[:,cols.time] = u.aggregate_by_time(data[:,cols.time],
									args.uc_irc_args.aggr_time)

		ids = data[:,cols.source] * self.num_nodes + data[:,cols.target]
		self.num_non_existing = float(self.num_nodes**2 - ids.unique().size(0))

		idx = data[:,[cols.source,
				   	  cols.target,
				   	  cols.time]]

		self.max_time = data[:,cols.time].max()
		self.min_time = data[:,cols.time].min()
		

		return {'idx': idx, 'vals': torch.ones(idx.size(0))}