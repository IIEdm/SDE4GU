import torch
import torchsde
import torch.nn as nn
import torch.nn.functional as F

from backbone import *
from SGNN import *


# used for decoding embedding to classes
class decoder_MLP(nn.Module):
  def __init__(self, c, args):
    super().__init__()
    self.args = args
    self.m21 = nn.Linear(args.hidden_channels, args.hidden_channels)
    self.m22 = nn.Linear(args.hidden_channels, c)

  def forward(self, x):
    x = F.dropout(x, self.args.dropout, training=self.training)
    x = F.dropout(x + self.m21(torch.tanh(x)), self.args.dropout, training=self.training)
    x = F.dropout(self.m22(torch.tanh(x)), self.args.dropout, training=self.training)
    return x


class decoder_MLP_simple(nn.Module):
  def __init__(self, c, args):
    super().__init__()
    self.args = args

    self.m22 = nn.Linear(args.hidden_channels, c)

  def forward(self, x):
    x = self.m22(torch.tanh(x))
    return x

# used for encoding feature to embedding
class encoder_MLP(nn.Module):
  def __init__(self, d, args):
    super().__init__()
    self.args = args
    self.m11 = nn.Linear(d, args.hidden_channels)


  def forward(self, x):

      x = x.to(self.args.device)
      x = F.dropout(x, self.args.input_dropout, training=self.training)

      x = self.m11(x)

      return x



class SGNN(torchsde.SDEIto):
    def __init__(self, d, c, args):
        super(SGNN, self).__init__(noise_type="diagonal")
        self.input_encoder = encoder_MLP(d, args)
        self.bnin = nn.BatchNorm1d(args.hidden_channels)
        self.f_encoder = Drift(args.hidden_channels, args.hidden_channels, args.hidden_channels, num_layers=1, dropout=args.dropout, use_bn=args.use_bn)
        # SFN includes the graph information to model  the dependency of noises.
        self.g_encoder = SFN(args.hidden_channels, args.hidden_channels, args.hidden_channels, num_layers=1, dropout=args.dropout, use_bn=args.use_bn)
        
        self.bng = nn.BatchNorm1d(args.hidden_channels)
        

        self.output_decoder = decoder_MLP_simple(c, args)

        self.args = args
        self.time = self.args.time

        self.N = self.args.N
        self.ts = torch.tensor([0, self.time])
        self.device = self.args.device

        self.ind_flag = True
        self.ind_edge_index = None
        self.ood_edge_index = None
        self.ind_edge_weight = None
        self.ood_edge_weight = None
        self.c_size = c
        self.sdeint_fn = torchsde.sdeint_adjoint if self.args.adjoint else torchsde.sdeint

    def reset_parameters(self):
        self.f_encoder.reset_parameters()
        self.g_encoder.reset_parameters()

    def f_net(self, t, y):
        if self.ind_flag == True:
            edge_index = self.ind_edge_index.to(self.device)
            ax = self.f_encoder(y, edge_index)
            return ax - y
        else:

            edge_index = self.ood_edge_index.to(self.device)
            ax = self.f_encoder(y, edge_index)
            return ax - y

    def g_net(self, t, y):
        if self.ind_flag == True:
            edge_index = self.ind_edge_index.to(self.device)
            g_output = self.g_encoder(y, edge_index)

            return y-g_output
        else:
            edge_index = self.ood_edge_index.to(self.device)
            g_output = self.g_encoder(y, edge_index)

            return y-g_output
    

    def forward(self, x, flag, device):
        self.ind_flag = flag
        node_embeddings = self.input_encoder(x)
        if self.args.use_bn:
            node_embeddings = self.bnin(node_embeddings)
        ts = torch.linspace(0, self.time, self.N).to(device)


        z = self.sdeint_fn(
            sde=self,
            y0=node_embeddings,
            ts=ts,
            method=self.args.method,
            dt=self.args.dt,
            adaptive=self.args.adaptive,
            rtol=self.args.rtol,
            atol=self.args.atol,
            names={'drift': 'f_net', 'diffusion': 'g_net'}
        )

        hidden_embedding = z[-1]
        logits = self.output_decoder(hidden_embedding)

        return logits

