import torch
from ddpm import Model
import torch.optim as optim
from ddpm_utils import DDPMSampler
from stat_rnn import mmd_eval
import dgl
import random
import logging
import networkx as nx
from data import *
from GlobalProperties import *


def get_subGraph_features(org_adj, subgraphs_indexes, kernel_model):
    device = device = "cuda" if torch.cuda.is_available() else "cpu"
    subgraphs = []
    target_kelrnel_val = None

    for i in range(len(org_adj)):
        subGraph = org_adj[i]
        if subgraphs_indexes != None:
            subGraph = subGraph[:, subgraphs_indexes[i]]
            subGraph = subGraph[subgraphs_indexes[i], :]
        # Converting sparse matrix to sparse tensor
        subGraph = torch.tensor(subGraph.todense())
        subgraphs.append(subGraph)
    subgraphs = torch.stack(subgraphs).to(device)

    if kernel_model != None:
        target_kelrnel_val = kernel_model(subgraphs)
        target_kelrnel_val = [val.to("cpu") for val in target_kelrnel_val]
    subgraphs = subgraphs.to("cpu")
    torch.cuda.empty_cache()
    return target_kelrnel_val, subgraphs


def load_data(args):
    dataset = args.dataset
    list_adj, list_x, list_label = list_graph_loader(dataset, return_labels=True)  # , _max_list_size=80

    if args.bfsOrdering == True:
        list_adj = BFS(list_adj)

    self_for_none = True

    if len(list_adj) == 1:
        test_list_adj = list_adj.copy()
        list_graphs = Datasets(list_adj, self_for_none, list_x, None)
    else:
        max_size = None
        # list_label = None
        list_adj, test_list_adj, list_x_train, list_x_test, _, list_label_test = data_split(list_adj, list_x,
                                                                                            list_label)
        val_adj = list_adj[:int(len(test_list_adj))]
        list_graphs = Datasets(list_adj, self_for_none, list_x_train, list_label, Max_num=max_size,
                               set_diag_of_isol_Zer=False)
        list_test_graphs = Datasets(test_list_adj, self_for_none, list_x_test, list_label_test,
                                    Max_num=list_graphs.max_num_nodes,
                                    set_diag_of_isol_Zer=False)

    return list_graphs, list_test_graphs, val_adj, test_list_adj


def train(diff,args,feat):
    epochs = args.epoch_diff
    optimizer = optim.Adam(diff.parameters(), lr=args.lr_diff)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diff.to(device)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = diff.loss_fn(feat)
        print("Epoch: ", epoch + 1, "Loss:", loss.item())
        loss.backward()
        optimizer.step()
    torch.save(diff, 'diff.pt')


def sample(feat):
    diff = torch.load('diff.pt')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sampler = DDPMSampler(beta_1=1e-4, beta_T=0.02, T=1000, diffusion_fn=diff, device=device, shape=feat.shape)
    samples = sampler.sampling(1, feat, only_final=True)
    print(samples)
    torch.save(samples, 'samples.pt')


def test(args):
    list_graphs, list_test_graphs, val_adj, test_list_adj = load_data(args)
    dataset = args.dataset
    graph_save_path = './log/' + dataset + '/'
    model_save_path = graph_save_path + 'model.pt'
    model = torch.load(model_save_path)
    epoch = args.epoch_number-1
    batch = 1
    model.load_state_dict(torch.load(graph_save_path + "model_hyp_" + str(epoch) + "_" + str(batch)))
    list_graphs.shuffle()
    batch = 0
    mini_batch_size = args.batchSize
    self_for_none = True
    list_graphs.processALL(self_for_none=self_for_none)
    adj_list = list_graphs.get_adj_list()
    device = torch.device(args.device if torch.cuda.is_available() and args.UseGPU else "cpu")
    kernel_model = torch.load('kernel.pt')
    graphFeatures, _ = get_subGraph_features(adj_list, None, kernel_model)
    list_graphs.set_features(graphFeatures)
    for iter in range(0, max(int(len(list_graphs.list_adjs) / mini_batch_size), 1) * mini_batch_size, mini_batch_size):
        from_ = iter
        to_ = mini_batch_size * (batch + 1)
        # for iter in range(0, len(list_graphs.list_adjs), mini_batch_size):
        #     from_ = iter
        #     to_= mini_batch_size*(batch+1) if mini_batch_size*(batch+2)<len(list_graphs.list_adjs) else len(list_graphs.list_adjs)
        print(list_graphs)
        print(from_)
        print(to_)
        org_adj, x_s, node_num, subgraphs_indexes, target_kelrnel_val = list_graphs.get__(from_, to_, self_for_none, bfs=None)

        #
        node_num = len(node_num) * [list_graphs.max_num_nodes]

        x_s = torch.cat(x_s)
        x_s = x_s.reshape(-1, x_s.shape[-1])

        model.train()

        _, subgraphs = get_subGraph_features(org_adj, None, None)

        # target_kelrnel_val = kernel_model(org_adj, node_num)

        # batchSize = [org_adj.shape[0], org_adj.shape[1]]

        batchSize = [len(org_adj), org_adj[0].shape[0]]
        print(len(org_adj))
        # org_adj_dgl = [dgl.from_scipy(sp.csr_matrix(graph.cpu().detach().numpy())) for graph in org_adj]
        [graph.setdiag(1) for graph in org_adj]
        org_adj_dgl = [dgl.from_scipy(graph) for graph in org_adj]
        org_adj_dgl = dgl.batch(org_adj_dgl).to(device)
        print('org_adj_dgl')
        print(org_adj_dgl)
        pos_wight = torch.true_divide(sum([x.shape[-1] ** 2 for x in subgraphs]) - subgraphs.sum(), subgraphs.sum())

        reconstructed_adj, prior_samples, post_mean, post_log_std, generated_kernel_val, reconstructed_adj_logit = model(
            org_adj_dgl.to(device), x_s.to(device), batchSize, subgraphs_indexes)
        mu, std, h = model.encode(org_adj_dgl.to(device), x_s.to(device), batchSize)
        print(h.size())
        torch.save(h, 'feat.pt')
        h_sample = torch.load('samples.pt')
        mean = model.stochastic_mean_layer(h_sample)
        log_std = model.stochastic_log_std_layer(h_sample)
        samples = model.reparameterize(mean, log_std)
        reconstructed_adj_logit = model.decode(samples, subgraphs_indexes)
        reconstructed_adj = torch.sigmoid(reconstructed_adj_logit)

        rnd_indx = random.randint(0, len(node_num) - 1)
        sample_graph = reconstructed_adj[rnd_indx].cpu().detach().numpy()
        sample_graph = sample_graph[:node_num[rnd_indx], :node_num[rnd_indx]]
        sample_graph[sample_graph >= 0.5] = 1
        sample_graph[sample_graph < 0.5] = 0

        G = nx.from_numpy_matrix(sample_graph)
        # plotter.plotG(G, "generated" + dataset,
        #               file_name=graph_save_path + "generatedSample_At_epoch" + str(epoch))
        print("reconstructed graph vs Validation:")
        logging.info("reconstructed graph vs Validation:")
        reconstructed_adj = reconstructed_adj.cpu().detach().numpy()
        reconstructed_adj[reconstructed_adj >= 0.5] = 1
        reconstructed_adj[reconstructed_adj < 0.5] = 0
        reconstructed_adj = [nx.from_numpy_matrix(reconstructed_adj[i]) for i in range(reconstructed_adj.shape[0])]
        reconstructed_adj = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in
                             reconstructed_adj if not nx.is_empty(G)]

        target_set = [nx.from_numpy_matrix(val_adj[i].toarray()) for i in range(len(val_adj))]
        target_set = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in target_set if
                      not nx.is_empty(G)]
        reconstruc_MMD_loss = mmd_eval(reconstructed_adj, target_set[:len(reconstructed_adj)], diam=True)
        logging.info(reconstruc_MMD_loss)


from config import parser

args = parser.parse_args()

def diff_train(args):
    dataset = args.dataset
    number = args.epoch_number - 1
    graph_save_path = './log/' + dataset + '/'
    feat_save_path = graph_save_path + f'{number}_feat.pt'
    feat = torch.load(feat_save_path)
    print(feat)
    dim = feat.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diff = Model(dim_in=dim, dim_hidden=512, num_layer=10, T=1000, beta_1=1e-4, beta_T=0.02)
    # train(diff=diff,args=args,feat=feat)
    sample(feat)
    test(args)
# diff_train(args)