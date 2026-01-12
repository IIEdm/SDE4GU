import warnings 
warnings.filterwarnings('ignore')
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from logger import Logger_ood
from data_utils import  eval_acc, rand_splits, fpr_and_fdr_at_recall
from dataset import load_dataset
from parse import parser_add_main_args

from model import *
from sklearn.metrics import roc_auc_score, average_precision_score


def cal_max_score(id_preds_l, ood_preds_l):
    id_score, ood_score = id_preds_l[0], ood_preds_l[0]
    for id_preds, ood_preds in zip(id_preds_l[1:], ood_preds_l[1:]):
        id_score += id_preds
        ood_score += ood_preds
    id_score = (torch.max(id_score/len(id_preds_l), dim=1)).values.detach().cpu()
    ood_score = (torch.max(ood_score/len(ood_preds_l), dim=1)).values.detach().cpu()
    return id_score, ood_score
def corr_err_label(y_true, logits):
    y_true = y_true.detach().cpu().numpy()
    y_pred = logits.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    correct = y_true == y_pred
    return correct.flatten().astype(int)

def err_measure(cor_label, id_max_score, recall_level=0.95):
    id_max_score = np.array(id_max_score[:]).flatten()
    max_auroc = roc_auc_score(cor_label, id_max_score)
    max_aupr_cor = average_precision_score(cor_label, id_max_score)
    max_aupr_err = average_precision_score(1-cor_label, -id_max_score)
    max_fpr, max_threshould = fpr_and_fdr_at_recall(cor_label, id_max_score, recall_level)
    return [max_auroc, max_aupr_cor, max_aupr_err, max_fpr]
@torch.no_grad()
def misclassification(model, dataset_ind, dataset_ood, criterion, eval_func, args, device):
    model.eval()
    acc_res = []
    tms = args.samples
    id_predss, ood_predss = [], []
    edge_index_in, edge_weight_in = get_rw_adj(dataset_ind.edge_index, edge_weight=dataset_ind.edge_attr, norm_dim=1,
                                  fill_value=args.self_loop_weight,
                                  num_nodes=dataset_ind.num_nodes,
                                  dtype=dataset_ind.x.dtype)
    #x_in, edge_index_in = dataset_ind.x.to(device), edge_index_in.to(device)
    edge_index_ood, edge_weight_ood = get_rw_adj(dataset_ood.edge_index, edge_weight=dataset_ood.edge_attr, norm_dim=1,
                                   fill_value=args.self_loop_weight,
                                   num_nodes=dataset_ood.num_nodes,
                                   dtype=dataset_ood.x.dtype)
    edge_index_in, edge_index_ood, edge_weight_in, edge_weight_ood = edge_index_in.to(device), \
        edge_index_ood.to(device), edge_weight_in.to(device), edge_weight_ood.to(device)
    id_idx, ood_idx = dataset_ind.splits['test'], dataset_ood.node_idx
    train_idx, valid_idx, test_idx = dataset_ind.splits['train'], dataset_ind.splits['valid'], dataset_ind.splits['test']
    y = dataset_ind.y.to(device)
    model.encoder.ind_edge_index = edge_index_in
    model.encoder.ood_edge_index = edge_index_ood
    model.encoder.ind_edge_weight = edge_weight_in
    model.encoder.ood_edge_weight = edge_weight_ood
    # st = time.time()
    for t in range(tms):
        flag = True
        id_logits = model(dataset_ind, flag, device)
        id_preds = torch.softmax(id_logits, dim=1)[id_idx]
        id_predss.append(id_preds)
        if t == 0:
            # ACC
            train_score = eval_func(y[train_idx], id_logits[train_idx])
            valid_score = eval_func(y[valid_idx], id_logits[valid_idx])
            test_score = eval_func(y[test_idx], id_logits[test_idx])
            valid_out = F.log_softmax(id_logits[valid_idx], dim=1)
            valid_loss = criterion(valid_out, y[valid_idx].squeeze(1))
            test_correct = corr_err_label(y[test_idx], id_logits[test_idx])
            acc_res = [train_score, valid_score, test_score, valid_loss.item()]
            
        
        flag = False
        ood_logits = model(dataset_ood, flag, device)
        ood_preds = torch.softmax(ood_logits, dim=1)[ood_idx]
        ood_preds = torch.cat([id_preds, ood_preds], dim=0)
        ood_predss.append(ood_preds)
        
    id_max_score, ood_max_score = cal_max_score(id_predss, ood_predss)
    max_auroc, max_aupr, max_fpr, _ = get_measures(id_max_score, ood_max_score)
    misclass_res = err_measure(test_correct, id_max_score)
    return acc_res, misclass_res


def detection(pos, neg):
    #calculate the minimum detection error
    Y1 = neg
    X1 = pos
    end = np.max([np.max(X1), np.max(Y1)])
    start = np.min([np.min(X1),np.min(Y1)])
    gap = (end- start)/10

    errorBase = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / float(len(Y1))
        errorBase = np.minimum(errorBase, (tpr+error2)/2.0)

    return errorBase

def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1
    labels_neg = 1-labels
    n_pos = -pos
    n_neg = -neg
    n_examples = np.squeeze(np.vstack((n_pos, n_neg)))

    auroc = roc_auc_score(labels, examples)
    aupr = average_precision_score(labels, examples)
    aupr_n = average_precision_score(labels_neg, n_examples)
    detection_err = detection(pos, neg)
    detection_acc = 1-detection_err
    fpr, threshould = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, aupr_n, fpr, detection_acc



@torch.no_grad()
def evaluate_detection(model, dataset_ind, dataset_ood, criterion, eval_func, args, device):
    model.eval()
    
    edge_index_in, edge_weight_in = get_rw_adj(dataset_ind.edge_index, edge_weight=dataset_ind.edge_attr, norm_dim=1,
                                  fill_value=args.self_loop_weight,
                                  num_nodes=dataset_ind.num_nodes,
                                  dtype=dataset_ind.x.dtype)
    edge_index_ood, edge_weight_ood = get_rw_adj(dataset_ood.edge_index, edge_weight=dataset_ood.edge_attr, norm_dim=1,
                                   fill_value=args.self_loop_weight,
                                   num_nodes=dataset_ood.num_nodes,
                                   dtype=dataset_ood.x.dtype)
    edge_index_in, edge_index_ood, edge_weight_in, edge_weight_ood = edge_index_in.to(device), \
        edge_index_ood.to(device), edge_weight_in.to(device), edge_weight_ood.to(device)
    id_idx, ood_idx = dataset_ind.splits['test'], dataset_ood.node_idx
    train_idx, valid_idx, test_idx = dataset_ind.splits['train'], dataset_ind.splits['valid'], dataset_ind.splits['test']
    y = dataset_ind.y.to(device)
    model.encoder.ind_edge_index = edge_index_in
    model.encoder.ood_edge_index = edge_index_ood
    model.encoder.ind_edge_weight = edge_weight_in
    model.encoder.ood_edge_weight = edge_weight_ood
    id_scores, ood_scores = 0,0
    for i in range(args.samples):
        id_logits = model(dataset_ind, True, device)
        train_score = eval_func(y[train_idx], id_logits[train_idx])
        valid_score = eval_func(y[valid_idx], id_logits[valid_idx])
        test_score = eval_func(y[test_idx], id_logits[test_idx])
        
        train_out = F.log_softmax(id_logits[train_idx], dim=1)
        train_loss = criterion(train_out, y[train_idx].squeeze(1))
        valid_out = F.log_softmax(id_logits[valid_idx], dim=1)
        valid_loss = criterion(valid_out, y[valid_idx].squeeze(1))
        test_out = F.log_softmax(id_logits[test_idx], dim=1)
        test_loss = criterion(test_out, y[test_idx].squeeze(1))
        id_score = torch.logsumexp(id_logits, dim=-1)
        id_score = model.propagation(id_score, edge_index_in, 4, 0.3)
        id_score = id_score[id_idx].detach().cpu()
        id_scores += id_score
        
        ood_logits = model(dataset_ood, False, device)
        ood_score = torch.logsumexp(ood_logits, dim=-1)
        ood_score = model.propagation(ood_score, edge_index_ood, 4, 0.3)
        ood_score = ood_score[ood_idx].detach().cpu()
        ood_scores += ood_score
    id_score = id_scores/args.samples
    ood_score = ood_scores/args.samples
    auroc, aupr_in, aupr_out, fpr, detection_acc = get_measures(id_score, ood_score)
    return [train_score, valid_score, test_score], [train_loss, valid_loss, test_loss],\
    [auroc, aupr_in, aupr_out, fpr, detection_acc] 


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset_ind, dataset_ood_tr, dataset_ood_te = load_dataset(args)

if len(dataset_ind.y.shape) == 1:
    dataset_ind.y = dataset_ind.y.unsqueeze(1)
if len(dataset_ood_tr.y.shape) == 1:
    dataset_ood_tr.y = dataset_ood_tr.y.unsqueeze(1)
if isinstance(dataset_ood_te, list):
    for data in dataset_ood_te:
        if len(data.y.shape) == 1:
            data.y = data.y.unsqueeze(1)
else:
    if len(dataset_ood_te.y.shape) == 1:
        dataset_ood_te.y = dataset_ood_te.y.unsqueeze(1)

### get splits for all runs ###
if args.dataset in ['cora', 'citeseer', 'pubmed',  'arxiv']:
    pass
else:
    dataset_ind.splits = rand_splits(dataset_ind.node_idx, train_prop=args.train_prop, valid_prop=args.valid_prop)

### print dataset info ###
c = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])
d = dataset_ind.x.shape[1]

print(f"ind dataset {args.dataset}: all nodes {dataset_ind.num_nodes} | centered nodes {dataset_ind.node_idx.shape[0]} | edges {dataset_ind.edge_index.size(1)} | "
      + f"classes {c} | feats {d}")
print(f"ood tr dataset {args.dataset}: all nodes {dataset_ood_tr.num_nodes} | centered nodes {dataset_ood_tr.node_idx.shape[0]} | edges {dataset_ood_tr.edge_index.size(1)}")
if isinstance(dataset_ood_te, list):
    for i, data in enumerate(dataset_ood_te):
        print(f"ood te dataset {i} {args.dataset}: all nodes {data.num_nodes} | centered nodes {data.node_idx.shape[0]} | edges {data.edge_index.size(1)}")
else:
    print(f"ood te dataset {args.dataset}: all nodes {dataset_ood_te.num_nodes} | centered nodes {dataset_ood_te.node_idx.shape[0]} | edges {dataset_ood_te.edge_index.size(1)}")

### load method ###
model = GNSD(d, c, args)

### loss function ###
criterion = nn.NLLLoss()

### metric for classification ###
eval_func = eval_acc

### logger for result report ###
logger = Logger_ood(args.runs, args)
model.train()
print('MODEL:', model)

### Training loop ###
for run in range(args.runs):
    model.reset_parameters()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = model.loss_compute(dataset_ind, dataset_ood_tr, criterion, device, args)
        loss.backward()
        optimizer.step()

        if epoch % args.display_step == 0:
            
            acc_res, loss_res, ood_res = evaluate_detection(model, dataset_ind, dataset_ood_te, criterion, eval_func, args, device)
            print(f'Epoch: {epoch:03d}, '
                      f'Train Loss: {loss.item():.4f}, '
                      f'Valid Loss: {loss_res[1]:.4f}, '
                      f'Test Loss: {loss_res[2]:.4f}, '

                      f'ACCURACY: '
                      f'Train: {100 * acc_res[0]:.2f}%, '
                      f'Valid: {100 * acc_res[1]:.2f}%, '
                      f'Test: {100 * acc_res[2]:.2f}%'
                      f'    Detection Task: '
                      f'AUROC: {100 * ood_res[0]:.2f}%, '
                      f'AUPR_in: {100 * ood_res[1]:.2f}%, '
                      f'AUPR_out: {100 * ood_res[2]:.2f}%, '
                      f'FPR95: {100 * ood_res[3]:.2f}%, '
                      f'DETECTION_acc: {100 * ood_res[4]:.2f}%, ')
                      
            logger.add_result(run, loss_res+acc_res+ood_res)
        
    logger.print_statistics(run)

results = logger.print_statistics()

