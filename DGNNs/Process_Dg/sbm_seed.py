import torch
from torch_geometric.utils import stochastic_blockmodel_graph
import random
def stochastic_blockmodel_graph_with_seed(block_sizes, edge_probs, time_seed=None):
    
    seed = time_seed  
    torch.manual_seed(seed)  
    random.seed(seed)  

    return stochastic_blockmodel_graph(block_sizes, edge_probs)