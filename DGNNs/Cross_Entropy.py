import torch
import utils_dg as u
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence 

class Cross_Entropy(torch.nn.Module):
    """docstring for Cross_Entropy"""
    def __init__(self, args, dataset):
        super().__init__()
        weights = torch.tensor(args.class_weights).to(args.device)

        self.weights = self.dyn_scale(args.task, dataset, weights) 
        
    
    def dyn_scale(self,task,dataset,weights):

        def scale(labels):
            return weights
        return scale
    

    def logsumexp(self,logits):  
        m,_ = torch.max(logits,dim=1)
        m = m.view(-1,1)
        sum_exp = torch.sum(torch.exp(logits-m),dim=1, keepdim=True)
        return m + torch.log(sum_exp)
    
    def forward(self,logits,labels): 
        '''
        logits is a matrix M by C where m is the number of classifications and C are the number of classes   logits行M表示样本  每行有C个logits
        labels is a integer tensor of size M where each element corresponds to the class that prediction i  就是M个标签
        should be matching to
        '''
        labels = labels.view(-1,1)  
        alpha = self.weights(labels)[labels].view(-1,1)
        loss = alpha * (- logits.gather(-1,labels) + self.logsumexp(logits)) 
        return loss.mean()

class EDL_CE(torch.nn.Module):
    """docstring for Cross_Entropy"""
    def __init__(self, args, dataset, rho=1, mode="exp"):
        super().__init__()
        weights = torch.tensor(args.class_weights).to(args.device)

        self.weights = self.dyn_scale(args.task, dataset, weights) #
        self.rho = rho
        self.mode = mode
    
    def dyn_scale(self,task,dataset,weights):

        def scale(labels):
            return weights
        return scale
    

    def logsumexp(self,logits):  
        m,_ = torch.max(logits,dim=1)
        m = m.view(-1,1)
        sum_exp = torch.sum(torch.exp(logits-m),dim=1, keepdim=True)
        return m + torch.log(sum_exp)
    
    def forward(self,logits,labels):  
        if self.mode == "exp":
            alphas = torch.exp(logits)    
        elif self.mode == "relu":
            alphas = torch.relu(logits) + 1
        alpha_sum = alphas.sum(dim=1, keepdim=True)
        digamma_alpha = torch.digamma(alphas)
        digamma_alpha_sum = torch.digamma(alpha_sum)

        labels = labels.view(-1,1)  
        loss = labels * (digamma_alpha_sum - digamma_alpha)
        
        alphas_hat = alphas.clone()  
        batch_indices = torch.arange(alphas.shape[0]).unsqueeze(1)  
        alphas_hat[batch_indices, labels] = 1 
        alphas_tilde = alphas_hat + 1e-6
        dirichlet_1 = Dirichlet(alphas_tilde) 
        dirichlet_2 = Dirichlet(torch.ones_like(alphas_tilde))  
        kl_loss = kl_divergence(dirichlet_1, dirichlet_2)       
        loss = loss.mean(dim=1) + self.rho * kl_loss
        return loss.mean()

