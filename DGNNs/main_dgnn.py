import utils_dg as u
import torch
import torch.distributed as dist  
import numpy as np
import time
import random
import torch_geometric
import dgl
import os
#datasets
from Process_Dg import bitcoin_dl as bc
from Process_Dg import elliptic_temporal_dl as ell_temp
from Process_Dg import uc_irv_mess_dl as ucim
from Process_Dg import auto_syst_dl as aus
from Process_Dg import reddit_dl as rdt
from Process_Dg import brain_dl as brain
import sys

#taskers
import link_pred_tasker as lpt
import edge_cls_tasker as ect
import node_cls_tasker as nct
import brain_node_cls_tasker as brain_nct

#models
import models as mls
import egcn_h
import egcn_o
import model_gnsjd

import splitter as sp
import Cross_Entropy as ce

import trainer as tr

import logger

# detector
import detector
import pickle

def random_param_value(param, param_min, param_max, type='int'):
	if str(param) is None or str(param).lower()=='none':
		if type=='int':
			return random.randrange(param_min, param_max+1)
		elif type=='logscale':
			interval=np.logspace(np.log10(param_min), np.log10(param_max), num=100)
			return np.random.choice(interval,1)[0]
		else:
			return random.uniform(param_min, param_max)
	else:
		return param 

def build_random_hyper_params(args):
	if args.model == 'all':
		model_types = ['gcn', 'egcn_o', 'egcn_h', 'gruA', 'gruB','egcn','lstmA', 'lstmB']
		args.model=model_types[args.rank] 
	args.learning_rate =random_param_value(args.learning_rate, args.learning_rate_min, args.learning_rate_max, type='logscale')

	if args.model == 'gcn':
		args.num_hist_steps = 0
	
		args.num_hist_steps = random_param_value(args.num_hist_steps, args.num_hist_steps_min, args.num_hist_steps_max, type='int')
	
	args.gcn_parameters['feats_per_node'] =random_param_value(args.gcn_parameters['feats_per_node'], args.gcn_parameters['feats_per_node_min'], args.gcn_parameters['feats_per_node_max'], type='int')
	args.gcn_parameters['layer_1_feats'] =random_param_value(args.gcn_parameters['layer_1_feats'], args.gcn_parameters['layer_1_feats_min'], args.gcn_parameters['layer_1_feats_max'], type='int')
	if args.gcn_parameters['layer_2_feats_same_as_l1'] or str(args.gcn_parameters['layer_2_feats_same_as_l1']).lower()=='true':
		args.gcn_parameters['layer_2_feats'] = args.gcn_parameters['layer_1_feats']
	else:
		args.gcn_parameters['layer_2_feats'] =random_param_value(args.gcn_parameters['layer_2_feats'], args.gcn_parameters['layer_1_feats_min'], args.gcn_parameters['layer_1_feats_max'], type='int')
	args.gcn_parameters['lstm_l1_feats'] =random_param_value(args.gcn_parameters['lstm_l1_feats'], args.gcn_parameters['lstm_l1_feats_min'], args.gcn_parameters['lstm_l1_feats_max'], type='int')
	if args.gcn_parameters['lstm_l2_feats_same_as_l1'] or args.gcn_parameters['lstm_l2_feats_same_as_l1'].lower()=='true':
		args.gcn_parameters['lstm_l2_feats'] = args.gcn_parameters['lstm_l1_feats']
	else:
		args.gcn_parameters['lstm_l2_feats'] =random_param_value(args.gcn_parameters['lstm_l2_feats'], args.gcn_parameters['lstm_l1_feats_min'], args.gcn_parameters['lstm_l1_feats_max'], type='int')
	args.gcn_parameters['cls_feats']=random_param_value(args.gcn_parameters['cls_feats'], args.gcn_parameters['cls_feats_min'], args.gcn_parameters['cls_feats_max'], type='int')

	return args


def build_dataset(args, ood_mode=None):
	if args.data == 'bitcoinotc':
		args.bitcoin_args = args.bitcoinotc_args
		return bc.bitcoin_dataset(args, ood_mode)
	elif args.data == 'elliptic':
		return ell.Elliptic_Dataset(args, ood_mode)
	elif args.data == 'uc_irv_mess':
		return ucim.Uc_Irvine_Message_Dataset(args, ood_mode)
	elif args.data == 'autonomous_syst':
		return aus.Autonomous_Systems_Dataset(args, ood_mode)
	elif args.data == 'reddit':
		return rdt.Reddit_Dataset(args, ood_mode)
	elif args.data == 'brain':
		return brain.Brain_Dataset(args, ood_mode) 

def build_tasker(args,dataset,ood_mode=None):
	if args.task == 'link_pred': # SBM UCI AS
		return lpt.Link_Pred_Tasker(args,dataset,ood_mode=ood_mode) 
	elif args.task == 'edge_cls': # BC Reddit
		return ect.Edge_Cls_Tasker(args,dataset,ood_mode=ood_mode)
	elif args.task == 'node_cls': # Barin Elliptic
		if args.data=='brain':
			return brain_nct.Node_Cls_Tasker(args,dataset,ood_mode=ood_mode)
		else:
			return nct.Node_Cls_Tasker(args,dataset,ood_mode=ood_mode)
	elif args.task == 'static_node_cls':
		return nct.Static_Node_Cls_Tasker(args,dataset,ood_mode=ood_mode)

def build_gcn(args,tasker):
	gcn_args = u.Namespace(args.gcn_parameters) 
	gcn_args.feats_per_node = tasker.feats_per_node
	gcn_args.use_2_hot_node_feats = args.use_2_hot_node_feats
	gcn_args.use_1_hot_node_feats = args.use_1_hot_node_feats 

	if 'GNJSD' in args.model:
		args.transformer_parameters['concat_in_skipfeat'] = args.transformer_parameters.get('concat_in_skipfeat', False)
		args.transformer_parameters['skip_in_feat'] = args.transformer_parameters.get('skip_in_feat', True)
		args.transformer_parameters['use_spatial_feat_in_lpe'] = args.transformer_parameters.get('use_spatial_feat_in_lpe', False)
		args.transformer_parameters['use_spectral_in_lpe'] = args.transformer_parameters.get('use_spectral_in_lpe', True)
		args.transformer_parameters['num_filter_subspaces'] = args.transformer_parameters.get('num_filter_subspaces', 1)
		args.transformer_parameters['use_static_spectral_wavelets'] = args.transformer_parameters.get('use_static_spectral_wavelets', False)
		args.transformer_parameters['use_sgnn_dgl'] = args.transformer_parameters.get('use_sgnn_dgl', False)
		args.transformer_parameters['use_pe_module'] = args.transformer_parameters.get('use_pe_module', True)
		transformer_args = u.Namespace(args.transformer_parameters)
		transformer_args.feats_per_node = tasker.feats_per_node
		transformer_args.use_2_hot_node_feats = args.use_2_hot_node_feats
		transformer_args.use_1_hot_node_feats = args.use_1_hot_node_feats 

		transformer_args.aggregator = args.aggregator

	if args.model == 'gcn':
		return mls.Sp_GCN(gcn_args,activation = torch.nn.RReLU()).to(args.device)
	elif args.model == 'skipgcn':
		return mls.Sp_Skip_GCN(gcn_args,activation = torch.nn.RReLU()).to(args.device)
	else: 
		assert args.num_hist_steps > 0, 'more than one step is necessary to train LSTM'	
		if args.model == 'egcn':
			return egcn.EGCN(gcn_args, activation = torch.nn.RReLU()).to(args.device)
		elif args.model == 'GNSJD':
			return model_deft.DEFT(transformer_args, gcn_args, activation = torch.nn.RReLU(), device = args.device, data=args.data)

def add_laplacian_positional_encodings(splitter, pos_enc_dim):

    splitter.train.pos_enc_list = [[u.laplacian_positional_encoding(g['idx'], pos_enc_dim) for g in hist_graphs['hist_adj_list']] for hist_graphs in splitter.train]
    splitter.dev.pos_enc_list = [[u.laplacian_positional_encoding(g['idx'], pos_enc_dim) for g in hist_graphs['hist_adj_list']] for hist_graphs in splitter.dev]
    splitter.test.pos_enc_list = [[u.laplacian_positional_encoding(g['idx'], pos_enc_dim) for g in hist_graphs['hist_adj_list']] for hist_graphs in splitter.test]


def build_classifier(args,tasker):  
	if 'node_cls' == args.task or 'static_node_cls' == args.task:
		mult = 1 
	else:
		mult = 2
	if 'gru' in args.model or 'lstm' in args.model:
		in_feats = args.gcn_parameters['lstm_l2_feats'] * mult
	elif args.model == 'skipfeatsgcn' or args.model == 'skipfeatsegcn_h':
		in_feats = (args.gcn_parameters['layer_2_feats'] + args.gcn_parameters['feats_per_node']) * mult
	elif 'GWNN' in args.model:
		in_feats = args.gcn_parameters['filters'] * mult
	elif 'spatio_temporal_transformer' in args.model:
		in_feats = args.transformer_parameters['out_dim'] * mult
	elif 'DEFT' in args.model:
		in_feats = args.transformer_parameters['out_dim'] * mult
	else:
		in_feats = args.gcn_parameters['layer_2_feats'] * mult

	return mls.Classifier(args,in_features = in_feats, out_features = tasker.num_classes).to(args.device)

def get_model_params(gcn):
	num_parameters = 0
	trainable_parameters = 0
	for param in gcn.parameters():
		if param.requires_grad:
			trainable_parameters += param.numel()
		num_parameters += param.numel()
	return num_parameters, trainable_parameters

if __name__ == '__main__': 

	parser = u.create_parser()
	args = u.parse_args(parser)

	global rank, wsize, use_cuda
	args.use_cuda = (torch.cuda.is_available() and args.use_cuda)
	args.device='cpu'
	if args.use_cuda:
		args.device='cuda'
	print ("use CUDA:", args.use_cuda, "- device:", args.device)
	try:
		dist.init_process_group(backend='mpi') #, world_size=4
		rank = dist.get_rank()   
		wsize = dist.get_world_size() 
		print('Hello from process {} (out of {})'.format(dist.get_rank(), dist.get_world_size()))
		if args.use_cuda:
			torch.cuda.set_device(rank )  # are we sure of the rank+1????
			print('using the device {}'.format(torch.cuda.current_device()))
	except:
		rank = 0
		wsize = 1
		print(('MPI backend not preset. Set process rank to {} (out of {})'.format(rank,wsize)))

	if args.seed is None and args.seed!='None':
		seed = 123+rank#int(time.time())+rank
	else:
		if args.cmd_seed==-1:
			seed=args.seed#+rank
		else:
			seed=args.cmd_seed

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch_geometric.seed_everything(seed)
	dgl.seed(seed)

	args.seed=seed
	args.rank=rank
	args.wsize=wsize
	args = build_random_hyper_params(args)

	#build the dataset 
	dataset = build_dataset(args, ood_mode=None)
	print("==================dataset loaded========================")
	print(f"dataset:{args.data},    times_tamps:{dataset.max_time-dataset.min_time+1}")
	print(f"dataset.num_nodes:{dataset.num_nodes}")       
	if args.use_1_hot_node_feats:
		print("********this data set use degree as feats")
	else:
		print(f"dataset.nodes_feats[:3]:{dataset.nodes_feats[:1]}")
	print("========================================================")

	#build the tasker
	tasker = build_tasker(args,dataset)
	print("==================tasker builded========================")
	print(f"task_type:{args.task}")    
	print("========================================================")

	#build the splitter
	splitter = sp.splitter(args,tasker)
	print("==================splitter builded======================")
	print(f"times_tamps:{dataset.max_time-dataset.min_time+1}")
	print(f"hnum_hist_steps:{args.num_hist_steps}")
	print(f"train_proportion:dev_proportion ={args.train_proportion}:{args.dev_proportion}")
	print(f"train:{len(splitter.train)}")
	print(f"dev:{len(splitter.dev)}")
	print(f"test:{len(splitter.test)}")      
	print(f"num_epochs:{args.num_epochs}")
	print(f"eval_after_epochs:{args.eval_after_epochs}")
	print("========================================================")

	if args.model in ['spatial_transformer','spatio_temporal_transformer','static_spatial_transformer'] and args.transformer_parameters['lap_pos_enc']:
		add_laplacian_positional_encodings(splitter, args.transformer_parameters['pos_enc_dim'])
	


	ood_mode=args.OOD # None or str: SM FI 
	print("\033[38;5;213m" + f"preparing OOD:{ood_mode}" + "=" * 37) 
	# SM for "Structure manipulation" ,FI for "Feature interpolation", LO for "Label leave-out"
	dataset_ood = build_dataset(args, ood_mode=ood_mode) # None or str: SM FI LO 
	tasker_ood = build_tasker(args,dataset_ood, ood_mode=ood_mode)  # FI when use_1_hot_node_feats
	splitter_ood = sp.splitter(args,tasker_ood)
	print(f"*****EDL:{args.EDL}")
	print("=" * 40 + "\033[0m")

	#build the models
	gcn = build_gcn(args, tasker)
	classifier = build_classifier(args,tasker)

	# build a loss
	
	cross_entropy = ce.Cross_Entropy(args,dataset).to(args.device)
	edl_cross_entropy = ce.EDL_CE(args,dataset,rho=1,mode="exp").to(args.device)
	edl_cross_entropy = ce.EDL_CE(args,dataset,rho=1,mode="relu").to(args.device)
		
	# trainer
	trainer = tr.Trainer(args,
						 splitter = splitter,
						 splitter_ood = splitter_ood,
						 gcn = gcn,
						 classifier = classifier,
						 comp_loss = cross_entropy if args.EDL == "none" else edl_cross_entropy,
						 dataset = dataset,
						 num_classes = tasker.num_classes)

	trainer.train()

	#detect
	eval_test, nodes_embs, logits_test, labels_test = trainer.run_epoch(splitter.test, 10001, 'TEST', grad = False)
	eval_test_ood, nodes_embs_ood, logits_test_ood , _= trainer.run_epoch(splitter_ood.test, 10002, 'TEST', grad = False)
