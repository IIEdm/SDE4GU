import os
import sys
sys.path.append(os.path.abspath('..'))
import utils as u
from .sbm_seed import stochastic_blockmodel_graph_with_seed
from tqdm import tqdm
import tarfile

import torch

from datetime import datetime


class Autonomous_Systems_Dataset():
	def __init__(self,args, ood_mode=None):
		if not isinstance(args.aut_sys_args, u.Namespace):
			args.aut_sys_args = u.Namespace(args.aut_sys_args)

		tar_file = os.path.join(args.aut_sys_args.folder, args.aut_sys_args.tar_file)  
		tar_archive = tarfile.open(tar_file, 'r:gz')

		self.edges = self.load_edges(args,tar_archive)
		print(f"self.edges['idx'].size(0): {self.edges['idx'].size(0)}")
		print(f"self.edges['idx'][:3]: {self.edges['idx'][:3]}")
		print(f"self.edges['vals'][:3]: {self.edges['vals'][:3]}")
		
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
				print(f"self.edges['idx'].size(0): {self.edges['idx'].size(0)}")
				print(f"self.edges['idx'][:3]: {self.edges['idx'][:3]}")
				print(f"self.edges['vals'][:3]: {self.edges['vals'][:3]}")
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
			print("Autonomous Systems FI data will be got in tasker")
			


	def load_edges(self,args,tar_archive):
		files = tar_archive.getnames()

		cont_files2times = self.times_from_names(files)

		edges = []
		cols = u.Namespace({'source': 0,
							 'target': 1,
							 'time': 2})
		for file in files:
			data = u.load_data_from_tar(file, 
									tar_archive, 
									starting_line=4,
									sep='\t',
									type_fn = int,
									tensor_const = torch.LongTensor)

			time_col = torch.zeros(data.size(0),1,dtype=torch.long) + cont_files2times[file]

			data = torch.cat([data,time_col],dim = 1)

			data = torch.cat([data,data[:,[cols.target,
										   cols.source,
										   cols.time]]])
			
			edges.append(data)

		edges = torch.cat(edges)


		_,edges[:,[cols.source,cols.target]] = edges[:,[cols.source,cols.target]].unique(return_inverse = True)


		#use only first X time steps
		indices = edges[:,cols.time] < args.aut_sys_args.steps_accounted
		edges = edges[indices,:]
		
		#time aggregation
		edges[:,cols.time] = u.aggregate_by_time(edges[:,cols.time],args.aut_sys_args.aggr_time)

		self.num_nodes = int(edges[:,[cols.source,cols.target]].max()+1)


		ids = edges[:,cols.source] * self.num_nodes + edges[:,cols.target]
		self.num_non_existing = float(self.num_nodes**2 - ids.unique().size(0))


		self.max_time = edges[:,cols.time].max()
		self.min_time = edges[:,cols.time].min()
		
		return {'idx': edges, 'vals': torch.ones(edges.size(0))}

	def times_from_names(self,files):
		files2times = {}
		times2files = {}

		base = datetime.strptime("19800101", '%Y%m%d')
		for file in files:
			delta =  (datetime.strptime(file[2:-4], '%Y%m%d') - base).days

			files2times[file] = delta
			times2files[delta] = file


		cont_files2times = {}

		sorted_times = sorted(files2times.values())
		new_t = 0

		for t in sorted_times:
			
			file = times2files[t]

			cont_files2times[file] = new_t
			
			new_t += 1
		return cont_files2times