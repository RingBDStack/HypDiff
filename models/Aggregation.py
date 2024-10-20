
import torch




class AvePool(torch.nn.Module):
    def __init__(self):
        super(AvePool, self).__init__()

    def forward(self, in_tensor):
        return torch.sum(in_tensor,1)

class GcnPool(torch.nn.Module):
    """
    This layer apply a chain of mlp on each node of tthe graph.
    thr input is a matric matrrix with n rows whixh n is the nide number.
    """

    def __init__(self, inputDim, OutDim=1024):
        """

        :param inputDim: the feture size of input matrix; Number of the columns or dim of feature matrix
        :param normalize: either use the normalizer layer or not
        :param layers: the graph  feature size or the size of fature matrix before aggregation
        """
        super(GcnPool, self).__init__()
        self.featureTrnsfr = torch.nn.Linear(inputDim, OutDim)

    def forward(self, in_tensor, activation=torch.tanh):
        z = self.featureTrnsfr(in_tensor)
        z = torch.mean(z, 1)
        z = activation(z)
        return z

