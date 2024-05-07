from __future__ import division
from __future__ import print_function

import os
import numpy as np

import torch.optim as optim
import torch
import torch.nn as nn
from models.direction_diffusion import diffusion
from models.direction_diffusionset import diffusionset
from utils.data_utils import load_data
from hyperbolic_learning.hyperkmeans import hkmeanscom as Community_cluster
from hyperbolic_learning.hyperkmeans import graph_hkmeanscom as graph_Community_cluster
from tqdm import tqdm
import os
import pickle
import time
import networkx as nx
from torch.nn.modules.module import Module
import torch.nn.functional as F

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, graphs: list, max_num_nodes: int, features="id"):

        self.max_num_nodes = max_num_nodes
        self.adj = []
        self.features = []
        self.lens = []
        self.graphs = graphs
        for g in graphs:
            adj_ = nx.adjacency_matrix(g).todense()
            self.adj.append(np.asarray(adj_) + np.identity(g.number_of_nodes()))

        if features == "id":
            self.features.append(np.identity(max_num_nodes))

    def __len__(self):
        return len(self.adj)

    def __getitem__(self, idx):

        adj = self.adj[idx]
        num_nodes = adj.shape[0]
        adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj

        adj_decoded = np.zeros(self.max_num_nodes * (self.max_num_nodes + 1) // 2)
        node_idx = 0

        adj_vectorized = adj_padded[
            np.triu(np.ones((self.max_num_nodes, self.max_num_nodes))) == 1
        ]

        features = self.features[0]

        return {"adj": adj_padded, "adj_decoded": adj_vectorized, "features": features}


class FermiDiracDecoder(Module):
    """Fermi Dirac to compute edge probabilities based on distances."""

    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
        return probs

def hyperdiff(args): 
    #  
    Community_cluster(args) 
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = os.path.join(os.environ['LOG_DIR'], args.dataset)
    data = load_data(args, os.path.join(os.environ['DATAPATH'], args.dataset))
    adj=data['adj_train_norm']

    file_path = os.path.join(save_dir, 'embeddings.npy')
    numpy_array = np.load(file_path) 
    sparse_tensor=adj
    row_indices = sparse_tensor._indices()[0] 
    col_indices = sparse_tensor._indices()[1] 

    edge_index = torch.stack([row_indices, col_indices], dim=0)
    edge_index = edge_index.to(torch.long)
    h0 = torch.from_numpy(numpy_array)
    lable_path=os.path.join(save_dir,'label.npy')
    center_path=os.path.join(save_dir,'center.npy')

    label=np.load(lable_path)
    # print(label)
    # print()
    center=np.load(center_path)
    # print(center)
    center=torch.tensor(center)
    # print(center[label[0]])
    print('dim:', numpy_array.shape[1])
    diff = diffusion(args, dim=numpy_array.shape[1])
    h0=h0.to(device)
    optimizer = optim.Adam(diff.parameters(), lr=0.001)
    diff.to(device)
    center=center.to(device)
    epochs=args.diff_epoc
    adj=adj.to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = diff(h0,adj,label,center,restrict=True,graphset=False)
        print("Epoch: ", epoch + 1, "Loss:", loss.item())
        loss.backward()
        optimizer.step()
    file_path = os.path.join(save_dir, 'model.pth')
    diff_path=os.path.join(save_dir,'sample.pt')
    torch.save(diff.state_dict(),diff_path)
    diff.load_state_dict(torch.load(diff_path))
    diff.to(device)
    #sa=diff.sample(x=h0,adj=adj,labels=label,data=dataset)
    diff.eval()
    with torch.no_grad():
        sa=diff.p_sample_loop(h0,adj,label,center) 

    t=1
    embeddings=1

    # 加载模型参数
    from models.base_models import LPModel
    args.device = device
    args.n_nodes, args.feat_dim = data['features'].shape
    args.nb_false_edges = len(data['train_edges_false'])
    args.nb_edges = len(data['train_edges'])

    model = LPModel(args)
    model.load_state_dict(torch.load(file_path))
    model.eval()
    best_test_metrics = model.compute_metrics(embeddings, data,t,sa, adj,'test')
    print(best_test_metrics['roc'])
    print(best_test_metrics['ap'])

    sample_path=os.path.join(save_dir, 'result.pt')
    torch.save(sa,sample_path)


def test(args):

    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    save_dir = os.path.join(os.environ['LOG_DIR'], args.dataset)
    data = load_data(args, os.path.join(os.environ['DATAPATH'], args.dataset))

    adj=data['adj_train_norm']

    file_path = os.path.join(save_dir, 'embeddings.npy')
    numpy_array = np.load(file_path)
    sparse_tensor=adj
            #values = sparse_tensor._values()
    row_indices = sparse_tensor._indices()[0]  
    col_indices = sparse_tensor._indices()[1]  

    edge_index = torch.stack([row_indices, col_indices], dim=0)
    edge_index = edge_index.to(torch.long)

    h0 = torch.from_numpy(numpy_array)
    label_path=os.path.join(save_dir, 'label.npy')
    center_path=os.path.join(save_dir, 'center.npy')

    label=np.load(label_path)
    center=np.load(center_path)
    center=torch.tensor(center)
    diff=diffusion(args, dim=args.dim)
    file_path = os.path.join(save_dir, 'model.pth')
    #torch.save(diff.state_dict(),'sample.pt')
    diff_path=os.path.join(save_dir, 'sample.pt')
    diff.load_state_dict(torch.load(diff_path))
    diff.to(device)
    h0=h0.to(device)
    adj=adj.to(device)
    center=center.to(device)
    #sa=diff.sample(x=h0,adj=adj,labels=label,data=dataset)
    diff.eval()
    with torch.no_grad():
        sa=diff.p_sample_loop(h0,adj,label,center)
    # print(sa)
    #sa=h0
    t=1
    embeddings=1

    # 加载模型参数
    from models.base_models import LPModel
    args.device = device
    args.n_nodes, args.feat_dim = data['features'].shape
    args.nb_false_edges = len(data['train_edges_false'])
    args.nb_edges = len(data['train_edges'])

    model = LPModel(args)
    model.load_state_dict(torch.load(file_path))
    model.eval()
    best_test_metrics = model.compute_metrics(embeddings, data,t,sa, adj,'test')
    print(best_test_metrics['roc'])
    print(best_test_metrics['ap'])

    sample_path=os.path.join(save_dir, 'result.pt')
    torch.save(sa,sample_path)

# def hyperdiff_graphset(args):
def hyperdiff_graphset(args, tensor_array, dataloader):
    h0 = tensor_array
    graph_Community_cluster(args, tensor_array, dataloader)
    save_dir = os.path.join(os.environ['LOG_DIR'], args.dataset)
    # file_path = os.path.join(save_dir, 'embeddings.npy')
    # with open(file_path, "rb") as f:
    #     h0 = pickle.load(f)
    # tensor_array = h0
    diff=diffusionset(args, in_dim=h0.shape[1], out_dim=h0.shape[1])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    h0=h0.to(device)
    diff.to(device)
    epochs=args.diff_epoc
    optimizer = optim.Adam(diff.parameters(), lr=0.001)
    # losses = 0
    adj=1
    for epoch in tqdm(range(epochs)):

        f2=open('logs/timecp/' +args.dataset +'_time.txt', 'a+')
        f2.flush()
        t_total = time.time()

            # label加载
            # label_path=os.path.join(save_dir, 'label'+str(batch_idx)+'.npy')
            # center_path=os.path.join(save_dir, 'center'+str(batch_idx)+'.npy')
        label_path=os.path.join(save_dir, 'label'+'.npy')
        center_path=os.path.join(save_dir, 'center'+'.npy')
        label=np.load(label_path)
            #     print(label)
            #     print()
        center=np.load(center_path)
            #     print(center)
        center=torch.tensor(center)
            #     print(center[label[0]])
        center=center.to(device)
        # print(label.shape)
        # print(center.shape)
        
        optimizer.zero_grad()
        loss = diff(h0,adj,label,center,restrict=True,graphset=True)
        loss.backward()
        optimizer.step()

        print("Epoch: ", epoch + 1, "Loss:", loss.item())
        t_end = time.time()
        f2.write('{0:4} {1:4} \n'.format('diff_time',t_end - t_total))
        f2.close()
    diff_path=os.path.join(save_dir, 'sample.pt')
    torch.save(diff.state_dict(),diff_path)

    diff.load_state_dict(torch.load(diff_path))
    diff.to(device)
    #sa=diff.sample(x=h0,adj=adj,labels=label,data=dataset)
    diff.eval()
    with torch.no_grad():
        sa=diff.p_sample_loop(h0,adj,label,center,graphset=True)
    sample_path=os.path.join(save_dir, 'result'+'.pt')
    torch.save(sa,sample_path)

    g_path = os.path.join(save_dir, 'graph_test.dat')
    with open(g_path, "rb") as f:
        graphs = pickle.load(f)
    # generate(args, torch.stack(sas), sa.shape[1])
    generate(args,sa,sa.shape[2],graphs)

def test_graphset(args): 
    save_dir = os.path.join(os.environ['LOG_DIR'], args.dataset)
    file_path = os.path.join(save_dir, 'embeddings.npy')
    with open(file_path, "rb") as f:
        h0 = pickle.load(f)

    g_path = os.path.join(save_dir, 'graph_test.dat')
    with open(g_path, "rb") as f:
        graphs = pickle.load(f)[:100]
    print(len(graphs))
    with open(g_path, "wb") as f:
        pickle.dump(graphs, f)
    max_num_nodes = max(
            [graphs[i].number_of_nodes() for i in range(len(graphs))]
        )
    dataset = GraphDataset(graphs, max_num_nodes, "id")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    diff=diffusionset(args, in_dim=h0.shape[1], out_dim=h0.shape[1])
    diff.to(device)
    h0=h0.to(device)
    optimizer = optim.Adam(diff.parameters(), lr=0.001)
    diff_path=os.path.join(save_dir, 'sample.pt')
    diff.load_state_dict(torch.load(diff_path))
    diff.to(device)
    sas = []

    label_path=os.path.join(save_dir, 'label'+'.npy')
    label=np.load(label_path)
    center_path=os.path.join(save_dir, 'center'+'.npy')
    center=np.load(center_path)
    center=torch.tensor(center)
    center=center.to(device)
    adj=1
    diff.eval() 
    with torch.no_grad():
        sa = diff.p_sample_loop(h0, adj, label, center, graphset=True)
    sample_path = os.path.join(save_dir, 'result' + '.pt')
    torch.save(sa, sample_path)

    # generate(args, torch.stack(sas), sa.shape[1])
    #generate(args, sa, sa.shape[2])
    generate(args, sa, max_num_nodes, graphs)


def generate(args, h, max_num_nodes, graphs):
    save_dir = os.path.join(os.environ['LOG_DIR'], args.dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    h = torch.tensor(h).to(device)
    # Fermi
    # dc = FermiDiracDecoder(r=2.0, t=1.0)
    # # output_dim = max_num_nodes * (max_num_nodes + 1) // 2
    # # dim = max_num_nodes
    # # linear_dc = nn.Linear(dim,output_dim)
    # embeddings = dc.forward(h)

    #adj decode
    embeddings = h.squeeze(dim=1)
    embeddings = (-torch.log((1 / (embeddings + 1e-8)) - 1))
    graph_num = len(embeddings)
    max_num_node = int(max_num_nodes)
    y_pred_long = torch.zeros(graph_num, max_num_node, max_num_node).cuda() 
    G_pred_list = []
    for i in tqdm(range(graph_num)):
        # y = F.softmax(embeddings[i])
        y = (embeddings[i] - embeddings[i].min()) / (embeddings[i].max() - embeddings[i].min())
        y_thresh = (torch.ones(y.size(0), y.size(1))*0.5).cuda()
        y_result = torch.gt(y, y_thresh).float()
        if not torch.any(y_result):
            G_pred = graphs[i]
        else:
            y_pred_long[i,:,:] = y_result 
            adj_pred = decode_adj(y_pred_long[i].cpu().numpy())
            G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)
        
    fname = os.path.join(save_dir, 'diff_embeddings.npy')
    with open(fname, "wb") as f:
        pickle.dump(G_pred_list, f) 


def decode_adj(adj_output):
    '''
        recover to adj from adj_output
        note: here adj_output have shape (n-1)*m
    '''
    max_prev_node = adj_output.shape[1]
    adj = np.zeros((adj_output.shape[0], adj_output.shape[0]))
    for i in range(adj_output.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + max(0, i - max_prev_node + 1) - (i + 1)
        output_end = max_prev_node
        adj[i, input_start:input_end] = adj_output[i,::-1][output_start:output_end] # reverse order
    adj_full = np.zeros((adj_output.shape[0]+1, adj_output.shape[0]+1))
    n = adj_full.shape[0]
    adj_full[1:n, 0:n-1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full

def get_graph(adj):
    '''
    get a graph from zero-padded adj
    :param adj:
    :return:
    '''
    # remove all zeros rows and columns
    adj = adj[~np.all(adj == 0, axis=1)]
    adj = adj[:, ~np.all(adj == 0, axis=0)]
    adj = np.asmatrix(adj)
    G = nx.from_numpy_matrix(adj)
    return G