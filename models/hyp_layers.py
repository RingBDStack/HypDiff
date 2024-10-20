"""Hyperbolic layers."""
import math
import torch
import torch.nn as nn
import torch.nn.init as init
# from geoopt import PoincareBall
# from geoopt import Lorentz
# from manifolds import Lorentz
from torch.nn.modules.module import Module

# from layers.att_layers import DenseAtt
# from manifolds.utils import proj_tan


def get_dim_act_curv(config, num_layers, enc=True):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    model_config = config.model
    act = getattr(nn, model_config.act)
    if isinstance(act(),nn.LeakyReLU):
        acts = [act(0.5)] * (num_layers)
    else:
        acts = [act()] * (num_layers)  # len=args.num_layers
    if enc:
        dims = [model_config.hidden_dim] * (num_layers+1)  # len=args.num_layers+1
    else:
        dims = [model_config.dim]+[model_config.hidden_dim] * (num_layers)  # len=args.num_layers+1

    manifold_class = {'PoincareBall': PoincareBall, 'Lorentz': Lorentz}

    if enc:
        manifolds = [manifold_class[model_config.manifold](model_config.c, learnable=model_config.learnable_c)
                     for _ in range(num_layers)]+[manifold_class[model_config.manifold](model_config.c, learnable=model_config.learnable_c)]
    else:
        manifolds = [manifold_class[model_config.manifold](model_config.c, learnable=model_config.learnable_c)]+\
                    [manifold_class[model_config.manifold](model_config.c, learnable=model_config.learnable_c) for _ in
                    range(num_layers)]

    return dims, acts, manifolds


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, in_dim, out_dim, manifold_in, manifold_out, dropout=0., act=nn.ReLU(),use_norm=True):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(in_dim, out_dim, manifold_in, dropout)
        self.hyp_act = HypAct(manifold_in, manifold_out, act)
        self.norm = HypNorm(out_dim, manifold_in)
        self.use_norm = use_norm
    def forward(self, x):
        x = self.linear(x)
        if self.use_norm:
            x = self.norm(x)
        x = self.hyp_act(x)
        return x


class HGCLayer(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, in_dim, out_dim, manifold_in, manifold_out, dropout=0., act=nn.ReLU(), edge_dim=1, normalization_factor=1,
                 aggregation_method='sum', msg_transform=True, sum_transform=True,use_norm='ln'):
        super(HGCLayer, self).__init__()
        self.linear = HypLinear(in_dim, out_dim, manifold_in, dropout)
        self.agg = HypAgg(
            out_dim, manifold_in, dropout, edge_dim, normalization_factor, aggregation_method, act, msg_transform,
            sum_transform
        )
        self.use_norm = use_norm
        if use_norm != 'none':
            self.norm = HypNorm(out_dim, manifold_in,use_norm)
        self.hyp_act = HypAct(manifold_in, manifold_out, act)

    def forward(self, input):
        x, adj = input
        # print('in:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        x = self.linear(x)
        # print('linear:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        # if self.use_norm != 'none':
        #     x = self.norm(x)
        x = self.agg(x, adj)
        # print('agg:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        if self.use_norm != 'none':
            x = self.norm(x)
            # print('norm:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        x = self.hyp_act(x)
        # print('HypAct:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        return x,adj


class HGATLayer(nn.Module):

    """https://github.com/gordicaleksa/pytorch-GAT"""
    def __init__(self, in_dim, out_dim, manifold_in, manifold_out, dropout=0., act=nn.LeakyReLU(0.5), edge_dim=2, normalization_factor=1,
                 aggregation_method='sum', msg_transform=True, sum_transform=True,use_norm='ln',num_of_heads=4,local_agg=True,use_act=True,
                 return_multihead=False):
        super(HGATLayer, self).__init__()
        self.manifold_in = manifold_in
        self.manifold_out = manifold_out
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)
        self.linear_proj = nn.Linear(in_dim, out_dim,bias=False)
        # self.scoring_fn_target = nn.Parameter(torch.Tensor(1,1, num_of_heads, out_dim//num_of_heads))
        # self.scoring_fn_source = nn.Parameter(torch.Tensor(1,1, num_of_heads, out_dim//num_of_heads))
        self.scoring_fn = nn.Linear(2*out_dim//num_of_heads+1,1,bias=False)
        # self.att_net = DenseAtt(out_dim, dropout=dropout, edge_dim=edge_dim)
        self.leakyReLU = nn.LeakyReLU(0.2)
        self.use_act = use_act
        self.act = act
        self.num_of_heads = num_of_heads
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        if in_dim != out_dim:
            self.skip_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.skip_proj = None
        self.local_agg = local_agg
        self.return_multihead = return_multihead
        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input):
        x, adj = input
        b, n, _ = x.size()
        x = self.manifold_in.logmap0(x)  # (b,n,dim_in)
        x = self.dropout(x)
        nodes_features_proj = self.linear_proj(x).view(b,n,self.num_of_heads,-1)  # (b,n,n_head,dim_out/n_head)
        nodes_features_proj = self.dropout(nodes_features_proj)
        # score_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)  # (b,n,n_head)
        # score_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)  # (b,n,n_head)
        # score = self.leakyReLU(score_source.unsqueeze(1) + score_target.unsqueeze(2))
        x_left = torch.unsqueeze(nodes_features_proj, 2)
        x_left = x_left.expand(-1, -1, n, -1, -1)
        x_right = torch.unsqueeze(nodes_features_proj, 1)
        x_right = x_right.expand(-1, n, -1, -1, -1)  # (b,n,n,n_head,dim_out/n_head)
        score = self.scoring_fn(torch.cat([x_left,x_right,adj[...,None,None].expand(-1,-1,-1,self.num_of_heads,-1)],dim=-1)).squeeze()
        score = self.leakyReLU(score)  # (b,n,n,n_head)

        if self.local_agg:
            edge_mask = (adj > 1e-5).float()
            pad_mask = 1 - edge_mask
            connectivity_mask = -9e15 * pad_mask  # (b,n,n)
            score = score + connectivity_mask.unsqueeze(-1).expand(-1, -1,-1, self.num_of_heads)  # (b,n,n,n_head) padding的地方会-9e15

        att = torch.softmax(score,dim=2).transpose(2,3)  # (b,n,n_head,n)
        att = self.dropout(att).transpose(2,3).unsqueeze(-1)  # (b,n,n,n_head,1)
        msg = x_right * att  # (b,n,n,n_head,dim_out/n_head)
        msg = torch.sum(msg,dim=2)  # ->(b,n,n_head,dim_out/n_head)
        if self.return_multihead:
            return self.manifold_out.expmap0(msg)
        if self.in_dim != self.out_dim:
            x = self.skip_proj(x)  # (b,n,dim_out)
        x = self.manifold_out.expmap0(x)
        addend = msg.view(b,n,-1)+self.bias
        addend = self.manifold_out.transp0(x,addend)
        x = self.manifold_out.expmap(x,addend)
        if self.use_act:
            x = self.act(x)     # Todo 是否需要 可能encoder会有问题
        return x,adj
class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    input in manifold
    output in manifold
    """

    def __init__(self, in_dim, out_dim, manifold_in, dropout):
        super(HypLinear, self).__init__()
        self.manifold = manifold_in
        self.bias = nn.Parameter(torch.Tensor(1, out_dim))
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.dp = nn.Dropout(dropout)
        self.c = 1.0
        if out_dim > in_dim and self.manifold.name == 'Lorentz':
            self.scale = nn.Parameter(torch.tensor([1 / out_dim]).sqrt_())
        else:
            self.scale = 1.
        self.reset_parameters()

    def reset_parameters(self):
        # init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
       # x = self.manifold.logmap0(x,c=self.c)
        x = self.linear(x) * self.scale
        x = self.dp(x)
        # x = proj_tan0(x, self.manifold)
        x = self.manifold.proj_tan0(x,c=self.c)
        x = self.manifold.expmap0(x, c = self.c)
        bias = self.manifold.proj_tan0(self.bias.view(1, -1),c=self.c)
        bias = self.manifold.ptransp0(x, bias ,c=self.c)
        x = self.manifold.expmap(x, bias, c =self.c)
        return x


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, out_dim, manifold_in, dropout, edge_dim, normalization_factor=1, aggregation_method='sum',
                 act=nn.ReLU(), msg_transform=True, sum_transform=True):
        super(HypAgg, self).__init__()
        self.manifold = manifold_in
        self.dim = out_dim
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.att_net = DenseAtt(out_dim, dropout=dropout, edge_dim=edge_dim)
        self.msg_transform = msg_transform
        self.sum_transform = sum_transform
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        if msg_transform:
            self.msg_net = nn.Sequential(
                nn.Linear(out_dim+edge_dim-1, out_dim),
                act,
                # nn.LayerNorm(out_dim),
                nn.Linear(out_dim, out_dim)
            )

        if sum_transform:
            self.out_net = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                act,
                # nn.LayerNorm(out_dim),
                nn.Linear(out_dim, out_dim)
            )


    def forward(self, x, adj):
        b,n,_ = x.size()
        # # b x n x 1 x d     0,0,...0,1,1,...1...
        # x_left = torch.unsqueeze(x, 2)
        # x_left = x_left.expand(-1,-1, n, -1)
        # # b x 1 x n x d     0,1,...n-1,0,1,...n-1...
        # x_right = torch.unsqueeze(x, 1)
        # x_right = x_right.expand(-1,n, -1, -1)
        x_left = x[:,:, None,:].expand(-1,-1, n, -1)
        x_right = x[:,None, :,:].expand(-1,n, -1, -1)
        edge_attr = None
        if self.edge_dim == 1:
            edge_attr = None
        elif self.edge_dim == 2:
            edge_attr = self.manifold.dist(x_left, x_right, keepdim=True)  # (b,n,n,1)

        att = self.att_net(self.manifold.logmap0(x_left), self.manifold.logmap0(x_right),adj.unsqueeze(-1),edge_attr)  # (b,n_node,n_node,dim)
        if self.msg_transform:
            msg = self.manifold.logmap0(x_right)
            if edge_attr is not None:
                msg = torch.cat([msg,edge_attr],dim=-1)
            msg = self.msg_net(msg)
        else:
            msg = self.manifold.logmap(x_left, x_right)# (b,n_node,n_node,dim)  x_col落在x_row的切空间
        msg = msg * att
        msg = torch.sum(msg,dim=2)  # (b,n_node,dim)
        if self.sum_transform:
            msg = self.out_net(msg)
        if self.msg_transform:
            msg = proj_tan0(msg, self.manifold)
            msg = self.manifold.transp0(x, msg)
        else:
            msg = proj_tan(x, msg,self.manifold)
        output = self.manifold.expmap(x, msg)
        return output


class HypAct(Module):
    """
    Hyperbolic activation layer.
    input in manifold
    output in manifold
    """

    def __init__(self, manifold_in, manifold_out, act):
        super(HypAct, self).__init__()
        self.manifold_in = manifold_in
        self.manifold_out = manifold_out
        self.act = act
        self.c = 1.0

    def forward(self, x):
        x = self.act(self.manifold_in.logmap0(x,c=self.c))
        x = self.manifold_in.proj_tan0(x, c=self.c)
        x = self.manifold_out.expmap0(x,c=self.c)
        return x


class HypNorm(nn.Module):

    def __init__(self, in_features, manifold, method='ln'):
        super(HypNorm, self).__init__()
        self.manifold = manifold
        if self.manifold.name == 'Lorentz':
            in_features = in_features - 1
        if method == 'ln':
            self.norm = nn.LayerNorm(in_features)
        self.c = 1.0

    def forward(self, h):
        h = self.manifold.logmap0(h,c=self.c)
        if self.manifold.name == 'Lorentz':
            h[..., 1:] = self.norm(h[..., 1:].clone())
        else:
            h = self.norm(h)
        h = self.manifold.expmap0(h,c=self.c)
        return h


def proj_tan0(u, manifold):
    if manifold.name == 'Lorentz':
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals
    else:
        return u


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result
