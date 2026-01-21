import torch
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import torch.nn.functional as F
from tabulate import tabulate
import numpy as np

def compute_auroc(ind_scores, labels):
    return roc_auc_score(labels, ind_scores)

def compute_fpr95(ind_scores, labels):

    fpr, tpr, thresholds = roc_curve(labels, ind_scores)
    fpr95 = fpr[tpr >= 0.95][0]  
    return fpr95

def compute_aupr(ind_scores, labels):
    precision, recall, thresholds = precision_recall_curve(labels, ind_scores)
    labels_0 = (labels == 0)
    aupr_0 = average_precision_score(labels_0, 1 - ind_scores)
    labels_1 = (labels == 1)
    aupr_1 = average_precision_score(labels_1, ind_scores)
    return average_precision_score(labels, ind_scores), [aupr_0, aupr_1]

def display_metric_OD(method,ood_mode, ind_scores, labels, aupr_classes = False):
    auroc = compute_auroc(ind_scores, labels)
    aupr,aupr_classes = compute_aupr(ind_scores, labels)
    fpr95 = compute_fpr95(ind_scores, labels)
    print(f"\033[94m++++++OOD_mode:{ood_mode}, OOD detect method: {method} ++++++\033[0m")
    print(f"AUROC: {auroc:10.4f}  |  AUPR: {aupr:10.4f}  |  fpr95: {fpr95:10.4f}")
    if aupr_classes:    
        print(f"AUROC_OOD:{aupr_classes[0]} AUROC_ID:{aupr_classes[1]}")

def tabular_out(ind_scores_test, correct, exp_name="EXPNAME", method="method"):
    ind_scores_test = ind_scores_test.cpu().detach().numpy()
    correct = correct.cpu().detach().numpy()
    auroc = roc_auc_score(correct, ind_scores_test)
    precision_1, recall_1, _ = precision_recall_curve(correct, ind_scores_test)
    aupr_1 = auc(recall_1, precision_1)
    precision_0, recall_0, _ = precision_recall_curve(1 - correct, 1 - ind_scores_test)
    aupr_0 = auc(recall_0, precision_0)
    fpr, tpr, thresholds = roc_curve(correct, ind_scores_test)
    fpr95 = fpr[tpr >= 0.95][0]  

    accuracies = []
    for threshold in thresholds:
        predictions = (ind_scores_test >= threshold).astype(int)
        accuracy = (predictions == correct).mean()
        accuracies.append(accuracy)
    best_threshold = thresholds[np.argmax(accuracies)]
    best_accuracy = max(accuracies)
    accuracy = best_accuracy

    results = [
        ["Method","Experiment", "AUROC", "AUPR_1", "AUPR_0", "FPR95", "DET_ACC"],
        [method, exp_name, f"{auroc:.4f}", f"{aupr_1:.4f}", f"{aupr_0:.4f}", f"{fpr95:.4f}", f"{accuracy:.4f}"]
    ]

    print(tabulate(results, headers="firstrow", tablefmt="grid"))

def evaluate(logits_test, labels_test, logits_test_ood, method = "Confidence", ood_mode = "UNKONE"): 
    
    logits_all = torch.cat([logits_test, logits_test_ood], dim=0)
    labels = torch.cat([torch.ones(logits_test.shape[0]), torch.zeros(logits_test_ood.shape[0])], dim=0) 
    softmax_probs = F.softmax(logits_all, dim=1) 
    
    if method == "confidence":     
        ind_scores, _ = torch.max(softmax_probs, dim=1)  
    elif method == "energy":
        T = 1 
        neg_energy = T * torch.logsumexp(logits_all / T, dim=-1) 
        ind_scores = neg_energy
    elif method == "entropy":     
        p = F.softmax(logits_all, dim=1)
        logp = F.log_softmax(logits_all, dim=1)
        total_unc = -torch.sum(p * logp, dim=1)   
        ind_scores = -total_unc  
    elif method == "aleatoric":
        p_star = F.softmax(logits_all, dim = 1)    
        alpha_star = torch.clamp(torch.exp(logits_all), max=1e10)  
        # alpha_star = torch.exp(logits_all)
        alpha0_star = torch.sum(alpha_star, dim = 1) 
        a = torch.digamma(alpha_star + 1) - torch.digamma(alpha0_star + 1).reshape(-1,1)   
        alea_unc = -torch.sum(p_star * a, dim =1) 
        ind_scores = -alea_unc  
    elif method == "epistemic":  
        p = F.softmax(logits_all, dim=1)
        logp = F.log_softmax(logits_all, dim=1)
        total_unc = -torch.sum(p * logp, dim=1)
        p_star = F.softmax(logits_all, dim=1)  
        alpha_star = torch.clamp(torch.exp(logits_all), max=1e10) 
        # alpha_star = torch.exp(logits_all)
        alpha0_star = torch.sum(alpha_star, dim=1)
        a = torch.digamma(alpha_star + 1) - torch.digamma(alpha0_star + 1).reshape(-1, 1)
        alea_unc = -torch.sum(p_star * a, dim=1)
        epi_unc = total_unc - alea_unc
        ind_scores = -epi_unc
    elif method == "evisec":
        alpha = torch.exp(logits_all) + 1 
        alpha_sum = torch.sum(alpha, dim = 1) 
        u = len(alpha[0])/ alpha_sum
        print(f"+++++++++++++++++shape:{u.shape}")
        ind_scores = -u
    elif method == "EDL":
        alpha = torch.exp(logits_all) + 1 
        alpha_sum = torch.sum(alpha, dim = 1) 
        u = len(alpha[0])/ alpha_sum
        ind_scores = -u
    elif method == "DAEDL":
        alpha = torch.exp(logits_all) + 1 
        alpha_sum = torch.sum(alpha, dim = 1) 
        u = len(alpha[0])/alpha_sum
        ind_scores = -u
    # display_metric_OD(method, ood_mode, ind_scores.cpu(), labels.cpu())
    print(f"\033[94m+++++++++++++OOD_mode:{ood_mode}, method: {method} ++++++++++++\033[0m")
    tabular_out(ind_scores, labels, exp_name="OOD_detection", method=method)
    ind_scores_test = ind_scores[:logits_test.shape[0]]   
    predicted_classes = torch.argmax(logits_test, axis=1)

    correct = (predicted_classes == labels_test).float()
    # print(correct.shape,correct[:])
    tabular_out(ind_scores_test, correct, exp_name="MC_detection", method=method)
    

