import torch as torch

# import dgl
# from dgl.nn.pytorch import GraphConv as GraphConv
import scipy.sparse as sp
import numpy as np
from torch.nn import init


# torch.seed()
# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# torch._set_deterministic(True)
#************************************************************
"""some utiliees including some mudules to apply neural net blocks 
on the matrixes(relationl data)"""

#************************************************************

class node_mlp(torch.nn.Module):
    """
    This layer apply a chain of mlp on each node of tthe graph.
    thr input is a matric matrrix with n rows whixh n is the nide number.
    """
    def __init__(self, input, layers= [16, 16], normalize = False, dropout_rate = 0):
        """

        :param input: the feture size of input matrix; Number of the columns
        :param normalize: either use the normalizer layer or not
        :param layers: a list which shows the ouyput feature size of each layer; Note the number of layer is len(layers)
        """
        super(node_mlp, self).__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(input, layers[0])])

        for i in range(len(layers)-1):
            self.layers.append(torch.nn.Linear(layers[i],layers[i+1]))

        self.norm_layers = None
        if normalize:
            self.norm_layers =  torch.nn.ModuleList([torch.nn.BatchNorm1d(c) for c in [input]+layers])
        self.dropout = torch.nn.Dropout(dropout_rate)
        # self.reset_parameters()

    def forward(self, in_tensor, activation = torch.tanh, applyActOnTheLastLyr=True):
        h = in_tensor
        for i in range(len(self.layers)):
            if self.norm_layers!=None:
                if len(h.shape)==2:
                    h = self.norm_layers[i](h)
                else:
                    shape = h.shape
                    h= h.reshape(-1, h.shape[-1])
                    h = self.norm_layers[i](h)
                    h=h.reshape(shape)
            h = self.dropout(h)
            h = self.layers[i](h)
            if i != (len(self.layers)-1) or applyActOnTheLastLyr:
                h = activation(h)
        return h

class Graph_mlp(torch.nn.Module):
        """
        This layer apply a chain of mlp on each node of tthe graph.
        thr input is a matric matrrix with n rows whixh n is the nide number.
        """

        def __init__(self, input,  layers=[1024], normalize=False, dropout_rate=0):
            """

            :param input: the feture size of input matrix; Number of the columns
            :param normalize: either use the normalizer layer or not
            :param layers: a list which shows the ouyput feature size of each layer; Note the number of layer is len(layers)
            """
            super(Graph_mlp, self).__init__()

            layers = [input] + layers
            self.Each_neuron = torch.nn.ModuleList([torch.nn.Linear(layers[i],layers[i+1]) for i in range(len(layers)-1)])

        def forward(self, in_tensor, activation=torch.tanh):
            z = in_tensor
            for i in range(len(self.Each_neuron)):
                z = self.Each_neuron[i](z)
                if i !=(len(self.Each_neuron)-1):
                    z = activation(z)
            z = torch.mean(z,1)
            z = activation(z)
            return z

class poolingLayer_average(torch.nn.Module):
    """
    This layer apply a chain of mlp on each node of tthe graph.
    thr input is a matric matrrix with n rows whixh n is the nide number.
    """

    def __init__(self, input,):
        super(Graph_mlp, self).__init__()

    def forward(self, in_tensor, activation=torch.tanh):
        in_tensor = torch.mean(in_tensor,1)
        in_tensor = activation(in_tensor)
        return in_tensor


# class node_mlp(torch.nn.Module):
#     """
#     This laye applt a chain of mlp on each node of tthe graph.
#     This layer apply a chain of mlp on each node of tthe graph.
#     thr input is a matric matrrix with n rows whixh n is the nide number.
#     """
#     def __init__(self, input, layers= [16, 16]):
#         """
#         :param input: the feture size of input matrix; Number of the columns
#         :param layers: a list which shows the ouyput feature size of each layer; Note the number of layer is len(layers)
#         """
#         super(node_mlp, self).__init__()
#         self.layers = torch.nn.ParameterList([torch.nn.Parameter(torch.Tensor(input, layers[0]))])
#         for i in range(len(layers)-1):
#             self.layers.append(torch.nn.Parameter(torch.Tensor(layers[i],layers[i+1])))
#         self.reset_parameters()
#     def forward(self, in_tensor,activation = torch.tanh ):
#         h = in_tensor
#
#         for layer in self.layers:
#             torch.matmul(h, layer)
#             h = activation(h)
#         return h
#     def reset_parameters(self):
#         for i, weight in enumerate(self.layers):
#             self.layers[i] = init.xavier_uniform_(weight)



class edge_mlp(torch.nn.Module):
    """
    this layer applies Multi layer perceptron on each edge of the graph.
    the input of the layer is a 3 dimentional tensor in which
    the third dimention is feature vector of each mode.
    """
    def __init__(self, input, layers = [8, 4], activation = torch.tanh, last_layer_activation= torch.sigmoid):
        super(edge_mlp, self).__init__()
        """
        Construct the graph mlp
        Args:
            layer: a list whcih determine the number of layers and
            the number of neurons in each layer.
            input: The size of the third dimention of Tensor
        """
        self.activation = activation
        self.last_layer_activation = last_layer_activation
        self.mlp_layers = torch.nn.ParameterList([torch.nn.Parameter(torch.Tensor(input, layers[0]))])
        for i in range(len(layers)-1):
            self.mlp_layers.append(torch.nn.Parameter(torch.Tensor(layers[i],layers[i+1])))

        self.reset_parameters()

    def forward(self, in_tensor) :
        h = in_tensor
        for index,layer in enumerate(self.mlp_layers):
            h = torch.matmul(h, layer)
            if index<(len(self.mlp_layers)-1): h= self.activation(h)
            else: h = self.last_layer_activation(h)
        return torch.squeeze(h)

    def reset_parameters(self):
        for i, weight in enumerate(self.mlp_layers):
            self.mlp_layers[i] = init.xavier_uniform_(weight)

# GCN basic operation
# class GraphConvNN(torch.nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(GraphConvNN, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.weight = torch.nn.Parameter(torch.FloatTensor(input_dim, output_dim))
#         self.reset_parameters()
#
#     def forward(self,adj, x, sparse= False):
#         """
#         :param adj: normalized adjacency matrix of graph
#         :param x: normalized node feature matrix
#         :param sparse: either the adj is a sparse matrix or not
#         :return:
#         """
#         y = torch.matmul( adj, x)
#         y = torch.spmm(y,self.weight) if sparse else torch.matmul(y,self.weight)
#         return y
#
#     def reset_parameters(self):
#         self.weight = init.xavier_uniform_(self.weight)

class GraphConvNN(torch.nn.Module):
    r"""Apply graph convolution over an input signal.

    Graph convolution is introduced in `GCN <https://arxiv.org/abs/1609.02907>`__
    and can be described as below:

    .. math::
      h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ij}}h_j^{(l)}W^{(l)})

    where :math:`\mathcal{N}(i)` is the neighbor set of node :math:`i`. :math:`c_{ij}` is equal
    to the product of the square root of node degrees:
    :math:`\sqrt{|\mathcal{N}(i)|}\sqrt{|\mathcal{N}(j)|}`. :math:`\sigma` is an activation
    function.

    The model parameters are initialized as in the
    `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__ where
    the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
    and the bias is initialized to be zero.

    Notes
    -----
    Zero in degree nodes could lead to invalid normalizer. A common practice
    to avoid this is to add a self-loop for each node in the graph, which
    can be achieved by:

    >>> g = ... # some DGLGraph
    >>> g.add_edges(g.nodes(), g.nodes())


    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    norm : str, optional
        How to apply the normalizer. If is `'right'`, divide the aggregated messages
        by each node's in-degrees, which is equivalent to averaging the received messages.
        If is `'none'`, no normalization is applied. Default is `'both'`,
        where the :math:`c_{ij}` in the paper is applied.
    weight : bool, optional
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Attributes
    ----------
    weight : torch.Tensor
        The learnable weight tensor.
    bias : torch.Tensor
        The learnable bias tensor.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=False,
                 activation=None):
        super(GraphConvNN, self).__init__()
        if norm not in ('none', 'both', 'right'):
            raise ('Invalid norm value. Must be either "none", "both" or "right".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm

        if weight:
            self.weight = torch.nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, graph, feat, weight=None):
        r"""Compute graph convolution.

        Notes
        -----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Weight shape: "math:`(\text{in_feats}, \text{out_feats})`.

        Parameters
        ----------
        graph : DGLGraph
            The adg of graph. It should include self loop
        feat : torch.Tensor
            The input feature
        weight : torch.Tensor, optional
            Optional external weight tensor.

        Returns
        -------
        torch.Tensor
            The output feature
        """


        if self._norm == 'both':
            degs = graph.sum(-2).float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,)
            norm = torch.reshape(norm, shp)
            feat = feat * norm

        if weight is not None:
            if self.weight is not None:
                raise ('External weight is provided while at the same time the'
                               ' module has defined its own weight parameter. Please'
                               ' create the module with flag weight=False.')
        else:
            weight = self.weight

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            if weight is not None:
                feat = torch.matmul(feat, weight)
            # graph.srcdata['h'] = feat
            # graph.update_all(fn.copy_src(src='h', out='m'),
            #                  fn.sum(msg='m', out='h'))
            rst = torch.matmul(graph, feat)
        else:
            # aggregate first then mult W
            # graph.srcdata['h'] = feat
            # graph.update_all(fn.copy_src(src='h', out='m'),
            #                  fn.sum(msg='m', out='h'))
            # rst = graph.dstdata['h']
            rst = torch.matmul(graph, feat)
            if weight is not None:
                rst = torch.matmul(rst, weight)

        if self._norm != 'none':
            degs = graph.sum(-1).float().clamp(min=1)
            if self._norm == 'both':
                norm = torch.pow(degs, -0.5)
            else:
                norm = 1.0 / degs
            shp = norm.shape + (1,)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def preprocess_graph(adj):
    rowsum = np.array(adj.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return adj_normalized

class Learnable_Histogram(torch.nn.Module):
    def __init__(self, bin_num):
        super(Learnable_Histogram, self).__init__()
        self.bin_num = bin_num
        self.bin_width = torch.nn.Parameter(torch.Tensor(bin_num,1))
        self.bin_center = torch.nn.Parameter(torch.Tensor(bin_num,1))
        self.reset_parameters()

    def forward(self, vec):
        score_vec = vec-self.bin_center
        score_vec = 1-torch.abs(score_vec)*torch.abs(self.bin_width)
        score_vec = torch.relu(score_vec)
        score_vec
        return score_vec

    def reset_parameters(self):
        self.bin_width = torch.nn.init.xavier_uniform_(self.bin_width)
        self.bin_center = torch.nn.init.xavier_uniform_(self.bin_center)

