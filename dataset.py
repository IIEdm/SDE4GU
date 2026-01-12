
import numpy as np
import torch
import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


def load_dataset(args):
    
    if args.dataset in ('arxiv', 'cora', 'citeseer', 'pubmed', 'amazon-computer'):
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_graph_dataset(args.data_dir, args.dataset, args.ood_type)

    else:
        raise ValueError('Invalid dataname')
    return dataset_ind, dataset_ood_tr, dataset_ood_te



def create_feat_noise_dataset(data):

    x = data.x
    n = data.num_nodes
    idx = torch.randint(0, n, (n, 2))
    weight = torch.rand(n).unsqueeze(1)
    x_new = x[idx[:, 0]] * weight + x[idx[:, 1]] * (1 - weight)

    dataset = Data(x=x_new, edge_index=data.edge_index, y=data.y)
    dataset.node_idx = torch.arange(n)

    # if hasattr(data, 'train_mask'):
    #     tensor_split_idx = {}
    #     idx = torch.arange(data.num_nodes)
    #     tensor_split_idx['train'] = idx[data.train_mask]
    #     tensor_split_idx['valid'] = idx[data.val_mask]
    #     tensor_split_idx['test'] = idx[data.test_mask]
    #
    #     dataset.splits = tensor_split_idx

    return dataset


def load_graph_dataset(data_dir, dataname, ood_type):
    transform = T.NormalizeFeatures()
    if dataname in ('cora', 'citeseer', 'pubmed'):
        torch_dataset = Planetoid(root=f'{data_dir}Planetoid', split='public', name=dataname)
        dataset = torch_dataset[0]
        tensor_split_idx = {}
        idx = torch.arange(dataset.num_nodes)
        tensor_split_idx['train'] = idx[dataset.train_mask]
        
        tensor_split_idx['valid'] = idx[dataset.val_mask]
        
        tensor_split_idx['test'] = idx[dataset.test_mask]
        dataset.splits = tensor_split_idx
    
    elif dataname == 'amazon-computer':
        torch_dataset = Amazon(root=f'{data_dir}Amazon',
                               name='Computers', transform=transform)
        dataset = torch_dataset[0]
    
    elif dataname == 'arxiv':
        torch_dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=f'{data_dir}/ogb')
        dataset = torch_dataset[0]
        dataset.edge_index = to_undirected(dataset.edge_index)
        dataset.splits = torch_dataset.get_idx_split()
    else:
        raise NotImplementedError

    dataset.node_idx = torch.arange(dataset.num_nodes)
    dataset_ind = dataset

    if ood_type == 'feature':
        dataset_ood_tr = create_feat_noise_dataset(dataset)
        dataset_ood_te = create_feat_noise_dataset(dataset)
    elif ood_type == 'label':
        if dataname == 'cora':
            class_t = 3
        elif dataname == 'citeseer':
            class_t = 2
        elif dataname == 'pubmed':
            class_t = 1
        elif dataname == 'amazon-computer':
            class_t = 5
        elif dataname == 'arxiv':
            class_t = 20
        label = dataset.y

        if dataname == 'pubmed':
            center_node_mask_ind = (label >= class_t)
        else:
            center_node_mask_ind = (label > class_t).squeeze()
        idx = torch.arange(label.size(0))
        
        dataset_ind.node_idx = idx[center_node_mask_ind]

        if dataname in ('cora', 'citeseer', 'pubmed'):
            split_idx = dataset.splits
        elif dataname == 'arxiv':
            split_idx = torch_dataset.get_idx_split()
        if dataname in ('cora', 'citeseer', 'pubmed', 'arxiv'):
            tensor_split_idx = {}
            idx = torch.arange(label.size(0))
            for key in split_idx:
                mask = torch.zeros(label.size(0), dtype=torch.bool)
                mask[torch.as_tensor(split_idx[key])] = True
                tensor_split_idx[key] = idx[mask * center_node_mask_ind]
            dataset_ind.splits = tensor_split_idx

        dataset_ood_tr = Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y)
        dataset_ood_te = Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y)

        
        center_node_mask_ood_tr = (label == class_t).squeeze()
        
        center_node_mask_ood_te = (label < class_t).squeeze()
        dataset_ood_tr.node_idx = idx[center_node_mask_ood_tr]
        dataset_ood_te.node_idx = idx[center_node_mask_ood_te]
    else:
        raise NotImplementedError

    return dataset_ind, dataset_ood_tr, dataset_ood_te