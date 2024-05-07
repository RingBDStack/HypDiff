"""Base model class."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.layers import FermiDiracDecoder
import manifolds
import models.encoders as encoders
from torch_geometric.nn import GCNConv

class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        self.num_timesteps=1000
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)
    def exists(self,x):
        return x is not None

    def default(self,val, d):
        if self.exists(val):
            return val
        return d() if callable(d) else d
    def cal_u0(self,x):
       x=self.manifold.logmap0(x,c=1.0)
       u0=torch.mean(x,dim=0)
       u0=self.manifold.expmap0(u0,c=1.0)
       return u0
    def extract(self, a, t, x_shape):
       b, *_ = t.shape
       out = a.gather(-1, t)
       return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def tran_direction(self,direction_vector, gaussian_point):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       
        transformed_vector = torch.sign(direction_vector)
        transformed_point= gaussian_point*transformed_vector
        return transformed_point
    
    def get_alphas(self,timesteps):
        def linear_beta_schedule(timesteps):
            scale = 1000 / timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)
        betas = linear_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_minus=torch.sqrt(1. - alphas_cumprod)
        return torch.sqrt(alphas_cumprod),alphas_minus
    def q_sample(self, x_start, t,direction, noise=None):
        noise = self.default(noise, lambda: torch.randn_like(x_start))
        alphas_cumprod, minus_alphas_cumprod = self.get_alphas(1000)
        alphas_cumprod=alphas_cumprod.cuda()
        minus_alphas_cumprod=minus_alphas_cumprod.cuda()
        return (
            self.extract(alphas_cumprod, t, x_start.shape) * x_start +
            self.extract(minus_alphas_cumprod, t, x_start.shape) * noise
        )
    def encode(self, x, adj):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode(x, adj)
        # print(h)
        b,  n = h.shape
        t=1
        x=h
        h0 = h
        return x,t,h0,adj
    

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLP, self).__init__()
        input=input_dim+1
        self.fc1 = nn.Linear(input, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, output_dim)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.tanh=nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, t):
        # t=[ : ,None]
        # result = torch.cat((x, t), dim=2)
        x = torch.cat((x, t.unsqueeze(1)), dim=1).float()
        #x = self.sigmoid(self.fc1(x))
        x=self.tanh(self.fc1(x))
        #x = self.sigmoid(self.fc2(x))
        x=self.fc2(x)
        #x = self.relu(self.fc3(x))
        #x=self.fc3(x)
        return x
    
class gcndec(nn.Module):
    def __init__(self,input_dim ):
        super(gcndec, self).__init__()
        self.conv1 = GCNConv(input_dim, 32)  
        self.conv2 = GCNConv(32,input_dim)
        
        
    def forward(self, x, adj):
        x=x.to(torch.float32)
        #print(adj.shape)
        x=self.conv1(x,adj)
        #x=F.relu(x)
        x=self.conv2(x,adj)
        x=F.relu(x)
        return x

class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """

    def __init__(self, args):
        super(LPModel, self).__init__(args)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges
        self.mlp=MLP(args.dim, args.hid1, args.hid2, args.dim)
        self.gcn=gcndec(args.dim)
    def decode(self, h, idx):
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        probs = self.dc.forward(sqdist)
        return probs
    def sample(self,x,adj):
        img=torch.randn_like(x)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch=len(img)
        t = torch.full((batch,), 1000, device=device).long()
        h=self.gcn(img,adj)

        return h

    def compute_metrics(self, embeddings, data,t,h0, adj,split):
        if split == 'train' or split == 'all':
            sample_edges_false = data[f'{split}_edges_false']
            edges_false = sample_edges_false[np.random.randint(0, len(data[f'{split}_edges_false']), self.nb_edges)]
        else:
            edges_false = data[f'{split}_edges_false']

        # print('------')
        pos_scores = self.decode(h0, data[f'{split}_edges'])
        neg_scores = self.decode(h0, edges_false)

        # print('------')

        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))

        loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        #loss=loss+loss_diff
        # print(loss)
        #print(loss_diff)
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)

        losses = loss 
        metrics = {'loss': losses, 'roc': roc, 'ap': ap}
        return metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])

    