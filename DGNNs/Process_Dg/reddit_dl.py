import os
import sys
sys.path.append(os.path.abspath('..'))
import utils as u
from datetime import datetime
import torch
from torch_geometric.utils import stochastic_blockmodel_graph
from .sbm_seed import stochastic_blockmodel_graph_with_seed
from tqdm import tqdm
# task: EC  
# ./experiments/parameters_Reddit.yaml
class Reddit_Dataset():
	def __init__(self,args, ood_mode=None):
		if not isinstance(args.reddit_args, u.Namespace):
			args.reddit_args = u.Namespace(args.reddit_args)
		folder = args.reddit_args.folder

		#load nodes
		cols = u.Namespace({'id': 0,
							'feats': 1})
		file = args.reddit_args.nodes_file
		file = os.path.join(folder,file)
		with open(file) as file:
			file = file.read().splitlines()
		
		ids_str_to_int = {}
		id_counter = 0

		feats = []

		for line in file:
			line = line.split(',')
			#node id
			nd_id = line[0]
			if nd_id not in ids_str_to_int.keys():
				ids_str_to_int[nd_id] = id_counter
				id_counter += 1
				nd_feats = [float(r) for r in line[1:]]
				feats.append(nd_feats)
			else:
				print('duplicate id', nd_id)
				raise Exception('duplicate_id')

		feats = torch.tensor(feats,dtype=torch.float)
		num_nodes = feats.size(0)
		
		edges = []
		not_found = 0

		#load edges in title
		edges_tmp, not_found_tmp = self.load_edges_from_file(args.reddit_args.title_edges_file,
															 folder,
															 ids_str_to_int)
		edges.extend(edges_tmp)
		not_found += not_found_tmp
		
		#load edges in bodies

		edges_tmp, not_found_tmp = self.load_edges_from_file(args.reddit_args.body_edges_file,
															 folder,
															 ids_str_to_int)
		edges.extend(edges_tmp)
		not_found += not_found_tmp

		#min time should be 0 and time aggregation
		edges = torch.LongTensor(edges)
		edges[:,2] = u.aggregate_by_time(edges[:,2],args.reddit_args.aggr_time)
		max_time = edges[:,2].max()

		#separate classes
		sp_indices = edges[:,:3].t()
		sp_values = edges[:,3]

		
		pos_mask = sp_values == 1
		neg_mask = sp_values == -1

		neg_sp_indices = sp_indices[:,neg_mask]
		neg_sp_values = sp_values[neg_mask]
		neg_sp_edges = torch.sparse.LongTensor(neg_sp_indices
											  ,neg_sp_values,
											  torch.Size([num_nodes,
											  			  num_nodes,
											  			  max_time+1])).coalesce()

		pos_sp_indices = sp_indices[:,pos_mask]
		pos_sp_values = sp_values[pos_mask]		
		
		pos_sp_edges = torch.sparse.LongTensor(pos_sp_indices
										  	  ,pos_sp_values,
											  torch.Size([num_nodes,
											  			  num_nodes,
											  			  max_time+1])).coalesce()

		#scale positive class to separate after adding
		pos_sp_edges *= 1000
		
		sp_edges = (pos_sp_edges - neg_sp_edges).coalesce()
		
		#separating negs and positive edges per edge/timestamp
		vals = sp_edges._values()
		neg_vals = vals%1000
		pos_vals = vals//1000
		#vals is simply the number of edges between two nodes at the same time_step, regardless of the edge label
		vals = pos_vals - neg_vals

		#creating labels new_vals -> the label of the edges
		new_vals = torch.zeros(vals.size(0),dtype=torch.long)
		new_vals[vals>0] = 1
		new_vals[vals<=0] = 0
		vals = pos_vals + neg_vals
		indices_labels = torch.cat([sp_edges._indices().t(),new_vals.view(-1,1)],dim=1)
		
		self.edges = {'idx': indices_labels, 'vals': vals}
		# idx 是四列 起点；终点；时间戳，边的类别   val是权重 边分类中就不是这样  val是边的值
		# print(self.edges["idx"][:100,3],torch.unique(self.edges["idx"][:, 3]),torch.unique(self.edges["vals"]))
		# print(f"self.edges['idx'].shape{self.edges['idx'].shape}") # [1352694, 4]
		self.num_classes = 2
		self.feats_per_node = feats.size(1)
		self.num_nodes = num_nodes
		self.nodes_feats = feats
		# print(f"self.nodes_feats.shape:{self.nodes_feats.shape}") # [51278, 300]
		self.max_time = max_time
		self.min_time = 0
		print("----- ID edges:",len(self.edges['idx'])) 
		print(f"self.edges['idx'].shape{self.edges['idx'].shape}")
		print(f"self.edges['idx'].shape{self.edges['idx'][:200]}")
		print(f"self.edges['vals'].shape{self.edges['vals'].shape}")
		print(f"self.edges['vals']{self.edges['vals'][:100]}")
		print(f"self.max_time,self.min_time {self.max_time},{self.min_time}")
		print("--------------:",len(self.edges['idx'])) 


		if ood_mode == "NS":  # SM FI 
			
			NS_edges = []
			save_path = f"./DL/data_NS/{args.data}_NS_edges.pt"
			if os.path.exists(save_path):
				self.edges = torch.load(save_path)
				print(f"Loaded existing edges from {save_path}")
			else:	
				for timestep in tqdm(range(self.min_time, int(self.max_time) + 1), desc=f"Generating NS Graphs for {args.data}"):
					print("%"*20)

					edge_index = stochastic_blockmodel_graph_with_seed(block_sizes, edge_probs,time_seed=timestep)  
					cols_time = torch.tensor([timestep] * edge_index.size(1))
					edge_index_time = torch.cat([edge_index.T, cols_time.unsqueeze(1)], dim=1)
					edge_index_time_label = torch.cat([edge_index_time, torch.randint(0, 2, (edge_index_time.shape[0], 1))], dim=1)# Reddit
					NS_edges.append(edge_index_time_label)
					
					
				self.edges["idx"] = torch.cat(SM_edges, dim=0)
				self.edges["vals"] = torch.ones(self.edges["idx"].size(0))
				torch.save(self.edges, save_path)
				print(f"Edges saved to {save_path}")


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
					print(f"edge_index.shape{edge_index.shape}")
					cols_time = torch.tensor([timestep] * edge_index.size(1))
					edge_index_time = torch.cat([edge_index.T, cols_time.unsqueeze(1)], dim=1)
					edge_index_time_label = torch.cat([edge_index_time, torch.randint(0, 2, (edge_index_time.shape[0], 1))], dim=1)# Reddit
					SM_edges.append(edge_index_time_label)
					
					
				self.edges["idx"] = torch.cat(SM_edges, dim=0)
				self.edges["vals"] = torch.ones(self.edges["idx"].size(0))
				torch.save(self.edges, save_path)
				print(f"Edges saved to {save_path}")

		elif ood_mode == "FI": # 不会随时间变化的特征
			FI_nodes_feats = []
			x = self.nodes_feats
			n = self.num_nodes
			idx = torch.randint(0, n, (n, 2)) 
			weight = torch.rand(n).unsqueeze(1) 
			x_new = x[idx[:, 0]] * weight + x[idx[:, 1]] * (1 - weight)

			self.nodes_feats = x_new
			print(f"FI_nodes_feats{x_new.shape}:::{x_new.shape[:3]}")
			



	def prepare_node_feats(self,node_feats):
		node_feats = node_feats[0]
		return node_feats

	
	def load_edges_from_file(self,edges_file,folder,ids_str_to_int):
		edges = []
		not_found = 0

		file = edges_file
		
		file = os.path.join(folder,file)
		with open(file) as file:
			file = file.read().splitlines()

		cols = u.Namespace({'source': 0,
							'target': 1,
							'time': 3,
							'label': 4})

		base_time = datetime.strptime("19800101", '%Y%m%d')

		
		for line in file[1:]:
			fields = line.split('\t')
			sr = fields[cols.source]
			tg = fields[cols.target]

			if sr in ids_str_to_int.keys() and tg in ids_str_to_int.keys():
				sr = ids_str_to_int[sr]
				tg = ids_str_to_int[tg]

				time = fields[cols.time].split(' ')[0]
				time = datetime.strptime(time,'%Y-%m-%d')
				time = (time - base_time).days

				label = int(fields[cols.label])
				edges.append([sr,tg,time,label])
				#add the other edge to make it undirected
				edges.append([tg,sr,time,label])
			else:
				not_found+=1

		return edges, not_found
	

# if ood_mode == "SM":  # SM FI LO
# 			n = self.num_nodes
# 			d = self.edges['idx'].shape[0]/self.num_nodes/(self.num_nodes-1)/(self.max_time-self.min_time+1)
# 			num_blocks = self.num_classes
# 			p_ii, p_ij = 1.5 * d, 0.5 * d  
# 			block_size = n // num_blocks
# 			block_sizes = [block_size for _ in range(num_blocks-1)] + [block_size + n % block_size] 
# 			edge_probs = torch.ones((num_blocks, num_blocks)) * p_ij
# 			edge_probs[torch.arange(num_blocks), torch.arange(num_blocks)] = p_ii 
# 			# unique
# 			unique_labels = torch.unique(self.edges["idx"][:, 3]) 
# 			unique_values = torch.unique(self.edges["vals"])

# 			SM_edges = []

# 			# 完整生成SM
# 			# all_edge_indices = [stochastic_blockmodel_graph(block_sizes, edge_probs) for _ in tqdm(range(self.min_time, self.max_time + 1), desc="Generating SM Graphs", ncols=100)]
# 			# 生成一次图后复制
# 			# edge_index = stochastic_blockmodel_graph(block_sizes, edge_probs)
# 			# all_edge_indices = [edge_index for _ in range(self.min_time, self.max_time + 1)]

# 			# 生成 （本地没有文件时 反注释一种生成方式之后反注释生成）
# 			# for timestep, edge_index in enumerate(all_edge_indices, start=self.min_time):
# 			# 	cols_time = torch.tensor([timestep] * edge_index.size(1))
# 			# 	edge_index_time = torch.cat([all_edge_indices[timestep].T, cols_time.unsqueeze(1)], dim=1)
# 			# 	print(timestep)
# 			# 	SM_edges.append(edge_index_time)
				
# 			# self.edges["idx"] = torch.cat(SM_edges, dim=0)
# 			# print(f"self.edges['idx'].shape{self.edges['idx'].shape}") # [1330404, 3]
# 			# torch.save(self.edges["idx"], "./reddit_SM_edges_idx.pt")
# 			self.edges["idx"] = torch.load("./reddit_SM_edges_idx.pt")
# 			print(len(SM_edges))
			
# 			new_labels = unique_labels[torch.randint(0, len(unique_labels), (self.edges["idx"].size(0),))].unsqueeze(1) 
# 			self.edges["idx"] = torch.cat([self.edges["idx"], new_labels], dim=1)
# 			# print(self.edges["idx"][:,3],self.edges["idx"].shape)

# 			new_values = unique_values[torch.randint(0, len(unique_values), (self.edges["idx"].size(0),))].unsqueeze(1) 
# 			# remember to squeeze
# 			self.edges["vals"] = new_values.squeeze()  
# 			# print(self.edges["vals"][:3],self.edges["vals"].shape)
# 			print("-----OOD edges:",len(self.edges)) 
# 			print(f"self.edges['idx'].shape: {self.edges['idx'].shape}")
# 			print(f"self.edges['vals'].shape: {self.edges['vals'].shape}")
# 			print(f"self.max_time,self.min_time: {self.max_time},{self.min_time}")
# 			print("--------------:",len(self.edges)) 