import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pickle
import optimizers
from torch.nn.modules.module import Module
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
import numpy as np
import networkx as nx
from tqdm import tqdm
from utils.data_utils import load_data
import os
import time
from models.base_models import LPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

class FermiDiracDecoder(Module):
    """Fermi Dirac to compute edge probabilities based on distances."""

    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
        return probs


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

def train(args):
    real_graphs = []
    path = './data/'
    datasets = [args.dataset]
    # datasets = ["BA","COLLAB","SynCommunity","SynER","SynEgo","PROTEINS","IMDB-BINARY",]#"MUTAG","Grid",
    for dataset in datasets:
        if dataset == 'SynER':
            with open(f"./data/{dataset}_origin.pkl", "rb") as f:
                real_graphs = pickle.load(f)
        elif dataset == 'SynCommunity':
            with open(f"./data/{dataset}1000_origin.pkl", "rb") as f:
                real_graphs = pickle.load(f)
        elif dataset == 'SynEgo':
            with open(f"./data/{dataset}1000_origin.pkl", "rb") as f:
                real_graphs = pickle.load(f)
        elif dataset in ["CL100","CL500","CL1000","CL5000"]:
            with open(f"./data/{dataset}.pkl", "rb") as f:
                real_graphs = pickle.load(f)
        elif dataset in ["BA","Grid"]:
            with open(f"./data/{dataset}.pkl", "rb") as f:
                real_graphs = pickle.load(f)
        elif dataset in ["MUTAG","IMDB-BINARY","COLLAB","PROTEINS"]:
            origin_dataset = TUDataset(name = dataset, root = path)
            # if len(origin_dataset) > 100:
            for graph in origin_dataset: 
                if graph.num_nodes < 50:
                    temp_graph = to_networkx(graph)
                    # if temp_graph.number_of_nodes() < 20: 
                        # origin_dataset = origin_dataset[:100]
                    real_graphs.append(temp_graph)


        graphs = real_graphs
        print('graph_len:', len(graphs))
        save_dir = os.path.join(os.environ['LOG_DIR'], args.dataset)

        print(len(graphs))
        max_num_nodes = max(
                [graphs[i].number_of_nodes() for i in range(len(graphs))]
            )
        args.dim = max_num_nodes
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        name = dataset
        dataset = GraphDataset(graphs, max_num_nodes, "id")
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1
        )
        for batch_idx, in_data in enumerate(dataloader):
            inti_data = load_data(args, in_data)
            args.n_nodes, args.feat_dim = inti_data['features'].shape
            args.nb_false_edges = len(inti_data['train_edges_false'])
            args.nb_edges = len(inti_data['train_edges'])
            break
        
        args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
        # No validation for reconstruction task
        args.eval_freq = args.epochs + 1
        if not args.lr_reduce_freq:
            args.lr_reduce_freq = args.epochs
        model = LPModel(args)
        model = model.to(args.device)
        optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                        weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(args.lr_reduce_freq),
            gamma=float(args.gamma)
        )
        tot_params = sum([np.prod(p.size()) for p in model.parameters()])
        # if args.cuda is not None and int(args.cuda) >= 0 :
        #     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        #     model = model.to(args.device)
        #     for x, val in data.items():
        #         if torch.is_tensor(data[x]):
        #             data[x] = data[x].to(args.device)
        # Train model
        # t_total = time.time()
        counter = 0
        best_val_metrics = model.init_metric_dict()
        best_test_metrics = None
        best_emb = None
        ######### adj_list #########
        embedding_list = []
        graphs_new = []
        for epoch in tqdm(range(args.epochs)): 
            f2=open('logs/timecp/' +args.dataset +'_time.txt', 'a+') 
            f2.flush()
            t_total = time.time()

            for batch_idx, in_data in tqdm(enumerate(dataloader)):
                data = load_data(args, in_data)
                for x, val in data.items():
                    if torch.is_tensor(data[x]):
                        data[x] = data[x].to(args.device)
                # adj_sig = torch.sigmoid(data['adj'][0])
                # adj_list.append(adj_sig)

                # embeddings = train_encoder(args, p_data, save_dir)
                model.train()
                optimizer.zero_grad()
                embeddings,t,h0,adj= model.encode(data['features'], data['adj_train_norm'])
                train_metrics = model.compute_metrics(embeddings, data, t,h0,adj,'train')
                train_metrics['loss'].backward()
                if args.grad_clip is not None:
                    max_norm = float(args.grad_clip)
                    all_params = list(model.parameters())
                    for param in all_params:
                        torch.nn.utils.clip_grad_norm_(param, max_norm)
                optimizer.step()
                lr_scheduler.step()
                
                if (epoch + 1) % args.eval_freq == 0:
                    model.eval()
                    embeddings,t,h0,adj = model.encode(data['features'], data['adj_train_norm'])
                    val_metrics = model.compute_metrics(embeddings, data, t,h0,adj,'val')

                if epoch == args.epochs-1:
                    model.eval()
                    best_emb,t,h0,adj = model.encode(data['features'], data['adj_train_norm'])
                    best_test_metrics = model.compute_metrics(best_emb, data, t,h0,adj,'test')
                    # if best_test_metrics['roc'] > 0.85:
                    embedding_list.append(best_emb.detach().cpu())
                    print('best_test_metrics:', best_test_metrics)

                    graphs_new.append(graphs[batch_idx])
            
            t_end = time.time()
            f2.write('{0:4} {1:4} \n'.format('hgcn_time',t_end - t_total))
            f2.close()

        fname_test = os.path.join(save_dir, 'graph_test.dat')
        with open(fname_test, "wb") as f:
            pickle.dump(graphs_new, f)
        print(len(graphs_new))
        fname = os.path.join(save_dir, 'embeddings.npy')
        with open(fname, "wb") as f:
            pickle.dump(torch.stack(embedding_list), f)
        print(len(embedding_list))
        print("end!")

        return torch.stack(embedding_list), dataloader
        

def train_encoder(args, data, save_dir):
    args.n_nodes, args.feat_dim = data['features'].shape
    args.nb_false_edges = len(data['train_edges_false'])
    args.nb_edges = len(data['train_edges'])
    # No validation for reconstruction task
    args.eval_freq = args.epochs + 1
    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs
    model = LPModel(args)

    optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    if args.cuda is not None and int(args.cuda) >= 0 :
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)
    # Train model
    # t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None

    for epoch in tqdm(range(args.epochs)):
        # tt = time.time()
        model.train()
        optimizer.zero_grad()
        embeddings,t,h0,adj= model.encode(data['features'], data['adj_train_norm'])
        # print(h0)

        train_metrics = model.compute_metrics(embeddings, data, t,h0,adj,'train')
        # print(train_metrics['loss'])
        train_metrics['loss'].backward()
        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            all_params = list(model.parameters())
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm)
        optimizer.step()
        lr_scheduler.step()
        
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            embeddings,t,h0,adj = model.encode(data['features'], data['adj_train_norm'])
            val_metrics = model.compute_metrics(embeddings, data, t,h0,adj,'val')

    model.eval()

    best_test_metrics = model.compute_metrics(embeddings, data,t,h0, adj,'test')

    if not best_test_metrics:
        model.eval()
        best_emb= model.encode(data['features'], data['adj_train_norm'])
        best_test_metrics = model.compute_metrics(best_emb, data, 'test')
    
    return best_emb
    # print(best_test_metrics)
    # print('End encoding!')
    # np.save(os.path.join(save_dir, 'embeddings.npy'), h0.cpu().detach().numpy())
    # if hasattr(model.encoder, 'att_adj'): 
    #     filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
    #     pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
    #     print('Dumped attention adj: ' + filename)

    # json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))
    # torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))