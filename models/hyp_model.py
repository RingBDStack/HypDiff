import models.Aggregation as Aggregation
import dgl
from utils.util import *
from models.hyp_layers import HNNLayer
from manifolds import PoincareBall
class HAveEncoder(torch.nn.Module):
    def __init__(self, in_feature_dim, hiddenLayers=[256, 256, 256], GraphLatntDim=1024):
        super(HAveEncoder, self).__init__()
        self.manifold = PoincareBall()
        hiddenLayers = [in_feature_dim] + hiddenLayers + [GraphLatntDim]
        self.normLayers = torch.nn.ModuleList(
            [torch.nn.LayerNorm(hiddenLayers[i + 1], elementwise_affine=False) for i in range(len(hiddenLayers) - 1)])
        self.normLayers.append(torch.nn.LayerNorm(hiddenLayers[-1], elementwise_affine=False))
        self.GCNlayers = torch.nn.ModuleList([dgl.nn.pytorch.conv.GraphConv(hiddenLayers[i], hiddenLayers[i + 1],
                                                                            activation=None, bias=True, weight=True) for
                                              i in range(len(hiddenLayers) - 1)])

        self.poolingLayer = Aggregation.AvePool()
        self.HNNlayers= HNNLayer(GraphLatntDim, GraphLatntDim, self.manifold,self.manifold)
        self.stochastic_mean_layer = node_mlp(GraphLatntDim, [GraphLatntDim])
        self.stochastic_log_std_layer = node_mlp(GraphLatntDim, [GraphLatntDim])

    def forward(self, graph, features, batchSize, activation=torch.nn.LeakyReLU(0.01)):
        h = features
        for i in range(len(self.GCNlayers)):
            h = self.GCNlayers[i](graph, h)
            h = activation(h)
            # if((i<len(self.GCNlayers)-1)):
            h = self.normLayers[i](h)

        h = h.reshape(*batchSize, -1)

        h = self.poolingLayer(h)

        h = self.normLayers[-1](h)
        # print('h1')
        # print(h)
        h = self.HNNlayers(h)
        # print('h2')
        # print(h)
        h = self.manifold.logmap0(h,c=1.0)
        # print('h3')
        # print(h)
        mean = self.stochastic_mean_layer(h, activation=lambda x: x)
        log_std = self.stochastic_log_std_layer(h, activation=lambda x: x)

        return mean, log_std, h


class GraphTransformerDecoder_FC(torch.nn.Module):
    def __init__(self, input, lambdaDim, SubGraphNodeNum, directed=True):
        super(GraphTransformerDecoder_FC, self).__init__()
        self.SubGraphNodeNum = SubGraphNodeNum
        self.directed = directed
        layers = [input] + [1024, 1024, 1024]

        if directed:
            layers = layers + [SubGraphNodeNum * SubGraphNodeNum]
        else:
            layers = layers + [int((SubGraphNodeNum - 1) * SubGraphNodeNum / 2) + SubGraphNodeNum]
        self.normLayers = torch.nn.ModuleList(
            [torch.nn.LayerNorm(layers[i + 1], elementwise_affine=False) for i in range(len(layers) - 2)])
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(layers[i], layers[i + 1], torch.float32) for i in range(len(layers) - 1)])

    def forward(self, in_tensor, subgraphs_indexes=None, activation=torch.nn.LeakyReLU(0.001)):

        for i in range(len(self.layers)):
            # in_tensor = self.normLayers[i](in_tensor)
            in_tensor = self.layers[i](in_tensor)
            if i != len((self.layers)) - 1:
                in_tensor = activation(in_tensor)
                in_tensor = self.normLayers[i](in_tensor)
        if self.directed:
            ADJ = in_tensor.reshape(in_tensor.shape[0], self.SubGraphNodeNum, -1)
        else:
            ADJ = torch.zeros((in_tensor.shape[0], self.SubGraphNodeNum, self.SubGraphNodeNum)).to(in_tensor.device)
            ADJ[:, torch.tril_indices(self.SubGraphNodeNum, self.SubGraphNodeNum, -1)[0],
            torch.tril_indices(self.SubGraphNodeNum, self.SubGraphNodeNum, -1)[1]] = in_tensor[:,
                                                                           :(in_tensor.shape[-1]) - self.SubGraphNodeNum]
            ADJ = ADJ + ADJ.permute(0, 2, 1)
            ind = np.diag_indices(ADJ.shape[-1])
            ADJ[:, ind[0], ind[1]] = in_tensor[:, -self.SubGraphNodeNum:]  # torch.ones(ADJ.shape[-1]).to(ADJ.device)
        # adj_list= torch.matmul(torch.matmul(in_tensor, self.lamda),in_tensor.permute(0,2,1))
        # return adj_list
        # if subgraphs_indexes==None:
        # adj_list= torch.matmul(in_tensor,in_tensor.permute(0,2,1))
        return ADJ
        # else:
        #     adj_list = []
        #     for i in range(in_tensor.shape[0]):
        #         adj_list.append(torch.matmul(in_tensor[i][subgraphs_indexes[i]].to(device), in_tensor[i][subgraphs_indexes[i]].permute(0,2,1)).to(device))
        #     return torch.stack(adj_list)


class kernelGVAE(torch.nn.Module):
    def __init__(self, ker, encoder, decoder, AutoEncoder, graphEmDim=4096):
        super(kernelGVAE, self).__init__()
        self.embeding_dim = graphEmDim
        self.kernel = ker  # TODO: bin and width whould be determined if kernel is his
        self.AutoEncoder = AutoEncoder
        self.decode = decoder
        self.encode = encoder

        self.stochastic_mean_layer = node_mlp(self.embeding_dim, [self.embeding_dim])
        self.stochastic_log_std_layer = node_mlp(self.embeding_dim, [self.embeding_dim])

    def forward(self, graph, features, batchSize, subgraphs_indexes):
        """
        :param graph: normalized adjacency matrix of graph
        :param features: normalized node feature matrix
        :return:
        """
        mean, log_std, h = self.encode(graph, features, batchSize)
        print('h')
        print(h.size())
        samples = self.reparameterize(mean, log_std)
        reconstructed_adj_logit = self.decode(samples, subgraphs_indexes)
        reconstructed_adj = torch.sigmoid(reconstructed_adj_logit)

        kernel_value = self.kernel(reconstructed_adj)
        return reconstructed_adj, samples, mean, log_std, kernel_value, reconstructed_adj_logit

    def reparameterize(self, mean, log_std):
        if self.AutoEncoder == True:
            return mean
        var = torch.exp(log_std).pow(2)
        eps = torch.randn_like(var)
        sample = eps.mul(var).add(mean)

        return sample