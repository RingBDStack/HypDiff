import torch

class kernel(torch.nn.Module):
    """
     this class return a list of kernel ordered by keywords in kernel_type
    """
    def __init__(self, **ker):
        """
        :param ker:
        kernel_type; a list of string which determine needed kernels
        """
        self.device = ker.get("device")
        super(kernel, self).__init__()
        self.kernel_type = ker.get("kernel_type")
        kernel_set = set(self.kernel_type)

        if "in_degree_dist" in kernel_set or "out_degree_dist" in kernel_set:
            self.degree_hist = Histogram(self.device, ker.get("degree_bin_width").to(self.device), ker.get("degree_bin_center").to(self.device))

        if "RPF" in kernel_set:
            self.num_of_steps = ker.get("step_num")
            self.hist = Histogram(self.device, ker.get("bin_width"), ker.get("bin_center"))

        if "trans_matrix" in kernel_set:
            self.num_of_steps = ker.get("step_num")



    def forward(self,adj):
        vec = self.kernel_function(adj)
        # return self.hist(vec)
        return vec

    def kernel_function(self, adj): # TODO: another var for keeping the number of moments
        # ToDo: here we assumed the matrix is symetrix(undirected) which might not
        vec = []  # feature vector
        for kernel in self.kernel_type:
            if "TotalNumberOfTriangles" == kernel:
                vec.append(self.TotalNumberOfTriangles(adj))
            if "in_degree_dist" == kernel:
                degree_hit = []
                for i in range(adj.shape[0]):
                    # degree = adj[i, subgraph_indexes[i]][:, subgraph_indexes[i]].sum(1).view(1, -1)
                    degree = adj[i].sum(1).view(1, -1)
                    degree_hit.append(self.degree_hist(degree.to(self.device)))
                vec.append(torch.cat(degree_hit))
            if "out_degree_dist" == kernel:
                degree_hit = []
                for i in range(adj.shape[0]):
                    degree = adj[i].sum(0).view(1, -1)
                    degree_hit.append(self.degree_hist(degree))
                vec.append(torch.cat(degree_hit))
            if "RPF" == kernel:
                raise("should be changed") #ToDo: need to be fixed
                tr_p = self.S_step_trasition_probablity(adj, self.num_of_steps)
                for i in range(len(tr_p)):
                    vec.append(self.hist(torch.diag(tr_p[i])))

            if "trans_matrix" == kernel:
                vec.extend(self.S_step_trasition_probablity(adj, self.num_of_steps))
                # vec = torch.cat(vec,1)

            if "tri" == kernel:  # compare the nodes degree in the given order
                tri, square = self.tri_square_count(adj)
                vec.append(tri), vec.append(square)

            if "TrianglesOfEachNode" == kernel: # this kernel returns a verctor, element i of this vector is the number of triangeles which are centered at node i
                vec.append(self.TrianglesOfEachNode(adj))

            if "ThreeStepPath" == kernel:
                vec.append(self.TreeStepPathes(adj))
        return vec

    def tri_square_count(self, adj):
        ind = torch.eye(adj[0].shape[0]).to(self.device)
        adj = adj - ind
        two__ = torch.matmul(adj, adj)
        tri_ = torch.matmul(two__, adj)
        squares = torch.matmul(two__, two__)
        return (torch.diagonal(tri_, dim1=1, dim2=2), torch.diagonal(squares, dim1=1, dim2=2))

    def S_step_trasition_probablity(self, adj, s=4, dataset_scale=None ):
        """
         this method take an adjacency matrix and return its j<s adjacency matrix, sorted, in a list
        :param s: maximum step; the function will return s-step matrixes from 1 to s in a list
        :param Adj: adjacency matrix of the graph
        :return: a list in whcih the i-th elemnt is the i step transition probablity
        """
        # mask = torch.zeros(adj.shape).to(device)

        p1 = adj.to(self.device)
        # p1 = p1 * mask
        # ind = torch.eye(adj[0].shape[0])
        # p1 = p1 - ind
        TP_list = []
        # to save memory Use ineficient loop
        if dataset_scale=="large":
            p = []
            for i in range(adj.shape[0]):
                p.append(p1[i] * (p1[i].sum(1).float().clamp(min=1) ** -1))
            p1 = torch.stack(p)
        else:
            p1 = p1*(p1.sum(2).float().clamp(min=1) ** -1).view(adj.shape[0],adj.shape[1], 1)

        # p1[p1!=p1] = 0
        # p1 = p1 * mask

        if s>0:
            # TP_list.append(torch.matmul(p1,p1))
            TP_list.append( p1)
        for i in range(s-1):
            TP_list.append(torch.matmul(p1, TP_list[-1] ))
        return TP_list

    def TrianglesOfEachNode(self, adj,  ):
        """
         this method take an adjacency matrix and count the number of triangles centered at each node; this method return a vector for each graph
        """

        p1 = adj.to(self.device)
        p1 = p1 * (1 - torch.eye(adj.shape[-1], adj.shape[-1])).to(self.device)

        # to save memory Use ineficient loop
        tri = torch.diagonal(torch.matmul(p1, torch.matmul(p1, p1)),dim1=-2, dim2=-1)/6
        return tri

    def TreeStepPathes(self, adj,  ):
        """
         this method take an adjacency matrix and count the number of pathes between each two node with lenght 3; this method return a matrix for each graph
        """

        p1 = adj.to(self.device)
        p1 = p1 * (1 - torch.eye(adj.shape[-1], adj.shape[-1])).to(self.device)

        # to save memory Use ineficient loop
        # tri = torch.diagonal(torch.matmul(p1, torch.matmul(p1, p1)),dim1=-2, dim2=-1)/6
        tri = torch.matmul(p1, torch.matmul(p1, p1))
        return tri

    def TotalNumberOfTriangles(self, adj):
        """
         this method take an adjacency matrix and count the number of triangles in it the corresponding graph
        """
        p1 = adj.to(self.device)
        p1 = p1 * (1 - torch.eye(adj.shape[-1], adj.shape[-1])).to(self.device)

        # to save memory Use ineficient loop
        tri = torch.diagonal(torch.matmul(p1, torch.matmul(p1, p1)),dim1=-2, dim2=-1)/6
        return tri.sum(-1)

class Histogram(torch.nn.Module):
    # this is a soft histograam Function.
    #for deails check section "3.2. The Learnable Histogram Layer" of
    # "Learnable Histogram: Statistical Context Features for Deep Neural Networks"
    def __init__(self, device, bin_width = None, bin_centers = None):
        super(Histogram, self).__init__()
        self.device = device
        self.bin_width = bin_width.to(self.device)
        self.bin_center = bin_centers.to(self.device)
        if self.bin_width == None:
            self.prism()
        else:
            self.bin_num = self.bin_width.shape[0]

    def forward(self, vec):
        #REceive a vector and return the soft histogram

        #comparing each element with each of the bin center
        score_vec = vec.view(vec.shape[0],1, vec.shape[1], ) - self.bin_center
        # score_vec = vec-self.bin_center
        score_vec = 1-torch.abs(score_vec)*self.bin_width
        score_vec = torch.relu(score_vec)
        return score_vec.sum(2)

    def prism(self):
        pass
