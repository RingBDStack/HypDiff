import logging
import torch.nn.functional as F  # type: ignore
import argparse
from models.model import *
from models.hyp_model import *
from data import *
import pickle
import random as random
from GlobalProperties import *
from stat_rnn import mmd_eval
import time
import timeit
import dgl  # type: ignore


class NodeUpsampling(torch.nn.Module):
    def __init__(self, InNode_num, outNode_num, InLatent_dim=None):
        super(NodeUpsampling, self).__init__()
        self.Node_mlp = torch.nn.Linear(InNode_num * InLatent_dim, InLatent_dim * outNode_num)

    def forward(self, inTensor, activation=torch.nn.LeakyReLU(0.001)):
        Z = self.Node_mlp(inTensor.reshpe(inTensor.shape[0], -1))
        Z = torch.matmul(Z.reshpe(inTensor.shape[0], -1).permute(0, 2, 1), inTensor)

        return activation(Z)


class LatentMtrixTransformer(torch.nn.Module):
    def __init__(self, InNode_num, InLatent_dim=None, OutLatentDim=None):
        super(NodeUpsampling, self).__init__()
        self.Node_mlp = torch.nn.Linear(InNode_num * InLatent_dim, InNode_num * OutLatentDim)

    def forward(self, inTensor, activation=torch.nn.LeakyReLU(0.001)):
        Z = self.Node_mlp(inTensor.reshpe(inTensor.shape[0], -1))
        Z = torch.matmul(inTensor, Z.reshpe(inTensor.shape[-1], -1))

        return activation(Z)


# ============================================================================

def test_(number_of_samples, model, graph_size, path_to_save_g, device,remove_self=True, save_graphs=True):
    import os
    if not os.path.exists(path_to_save_g):
        os.makedirs(path_to_save_g)
    # model.eval()
    generated_graph_list = []
    if not os.path.isdir(path_to_save_g):
        os.makedirs(path_to_save_g)
    k = 0
    for g_size in graph_size:
        for j in range(number_of_samples):
            z = torch.tensor(numpy.random.normal(size=[1, model.embeding_dim]))
            z = torch.randn_like(z)
            start_time = time.time()

            adj_logit = model.decode(z.to(device).float())
            print("--- %s seconds ---" % (time.time() - start_time))
            logging.info("--- %s seconds ---" % (time.time() - start_time))
            reconstructed_adj = torch.sigmoid(adj_logit)
            sample_graph = reconstructed_adj[0].cpu().detach().numpy()
            # sample_graph = sample_graph[:g_size,:g_size]
            sample_graph[sample_graph >= 0.5] = 1
            sample_graph[sample_graph < 0.5] = 0
            G = nx.from_numpy_matrix(sample_graph)
            # generated_graph_list.append(G)
            f_name = path_to_save_g + str(k) + str(g_size) + str(j) + args.dataset
            k += 1

            if remove_self:
                G.remove_edges_from(nx.selfloop_edges(G))

            G.remove_nodes_from(list(nx.isolates(G)))
            generated_graph_list.append(G)

    # ======================================================
    # save nx files
    if save_graphs:
        nx_f_name = path_to_save_g + "_" + args.dataset + "_" + args.decoder + "_" + args.model_vae + "_" + args.task
        with open(nx_f_name, 'wb') as f:
            pickle.dump(generated_graph_list, f)
    # # ======================================================
    return generated_graph_list


def EvalTwoSet(model, test_list_adj, graph_save_path, device, Save_generated=True, _f_name=None, onlyTheBigestConCom=True):
    generated_graphs = test_(1, model, [x.shape[0] for x in test_list_adj], graph_save_path, device, save_graphs=Save_generated)
    graphs_to_writeOnDisk = [nx.to_numpy_array(G) for G in generated_graphs]
    if (onlyTheBigestConCom == False):
        if Save_generated:
            np.save(graph_save_path + 'generatedGraphs_adj_' + str(_f_name) + '.npy', graphs_to_writeOnDisk,
                    allow_pickle=True)

            logging.info(mmd_eval(generated_graphs, [nx.from_numpy_matrix(graph.toarray()) for graph in test_list_adj]))
    print("====================================================")
    logging.info("====================================================")

    print("result for subgraph with maximum connected componnent")
    logging.info("result for subgraph with maximum connected componnent")
    generated_graphs = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in generated_graphs if
                        not nx.is_empty(G)]

    statistic_ = mmd_eval(generated_graphs, [nx.from_numpy_matrix(graph.toarray()) for graph in test_list_adj],
                          diam=True)
    # if writeThem_in!=None:
    #     with open(writeThem_in+'MMD.log', 'w') as f:
    #         f.write(statistic_)
    logging.info(statistic_)
    if Save_generated:
        graphs_to_writeOnDisk = [nx.to_numpy_array(G) for G in generated_graphs]
        np.save(graph_save_path + 'Single_comp_generatedGraphs_adj_' + str(_f_name) + '.npy', graphs_to_writeOnDisk,
                allow_pickle=True)

        graphs_to_writeOnDisk = [G.toarray() for G in test_list_adj]
        np.save(graph_save_path + 'testGraphs_adj_.npy', graphs_to_writeOnDisk, allow_pickle=True)
    return statistic_


def get_subGraph_features(args,org_adj, subgraphs_indexes, kernel_model):
    device = args.device
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


# the code is a hard copy of https://github.com/orybkin/sigma-vae-pytorch
def log_guss(mean, log_std, samples):
    return 0.5 * torch.pow((samples - mean) / log_std.exp(), 2) + log_std + 0.5 * np.log(2 * np.pi)


def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor


def OptimizerVAE(reconstructed_adj, reconstructed_kernel_val, targert_adj, target_kernel_val, log_std, mean, alpha,
                 reconstructed_adj_logit, pos_wight, norm):
    loss = norm * torch.nn.functional.binary_cross_entropy_with_logits(reconstructed_adj_logit.float(),
                                                                       targert_adj.float(), pos_weight=pos_wight)

    norm = mean.shape[0] * mean.shape[1]
    kl = (1 / norm) * -0.5 * torch.sum(1 + 2 * log_std - mean.pow(2) - torch.exp(log_std).pow(2))

    acc = (reconstructed_adj.round() == targert_adj).sum() / float(
        reconstructed_adj.shape[0] * reconstructed_adj.shape[1] * reconstructed_adj.shape[2])
    kernel_diff = 0
    each_kernel_loss = []
    log_sigma_values = []
    for i in range(len(target_kernel_val)):
        log_sigma = ((reconstructed_kernel_val[i] - target_kernel_val[i]) ** 2).mean().sqrt().log()
        log_sigma = softclip(log_sigma, -6)
        log_sigma_values.append(log_sigma.detach().cpu().item())
        step_loss = log_guss(target_kernel_val[i], log_sigma, reconstructed_kernel_val[i]).mean()
        each_kernel_loss.append(step_loss.cpu().detach().numpy() * alpha[i])
        kernel_diff += step_loss * alpha[i]

    kernel_diff += loss * alpha[-2]
    kernel_diff += kl * alpha[-1]
    each_kernel_loss.append((loss * alpha[-2]).item())
    each_kernel_loss.append((kl * alpha[-1]).item())
    return kl, loss, acc, kernel_diff, each_kernel_loss, log_sigma_values


def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])


# ========================================================================

synthesis_graphs = {"wheel_graph", "star", "triangular_grid", "DD", "ogbg-molbbbp", "grid", "small_lobster",
                    "small_grid", "community", "lobster", "ego", "one_grid", "IMDBBINARY", ""}


def train(args, list_graphs, alpha, self_for_none, decoder, device, model, optimizer,val_adj):
    epoch_number = args.epoch_number
    mini_batch_size = args.batchSize
    visulizer_step = args.Vis_step
    step = 0
    graph_save_path = args.graph_save_path
    keepThebest = True
    min_loss = float('inf')
    for epoch in range(epoch_number):

        list_graphs.shuffle()
        batch = 0
        for iter in range(0, max(int(len(list_graphs.list_adjs) / mini_batch_size), 1) * mini_batch_size,
                          mini_batch_size):
            from_ = iter
            to_ = mini_batch_size * (batch + 1)
            # for iter in range(0, len(list_graphs.list_adjs), mini_batch_size):
            #     from_ = iter
            #     to_= mini_batch_size*(batch+1) if mini_batch_size*(batch+2)<len(list_graphs.list_adjs) else len(list_graphs.list_adjs)

            org_adj, x_s, node_num, subgraphs_indexes, target_kelrnel_val = list_graphs.get__(from_, to_, self_for_none,
                                                                                              bfs=None)

            if (type(decoder)) in [GraphTransformerDecoder_FC]:  #
                node_num = len(node_num) * [list_graphs.max_num_nodes]

            x_s = torch.cat(x_s)
            x_s = x_s.reshape(-1, x_s.shape[-1])

            model.train()
            _, subgraphs = get_subGraph_features(args,org_adj, None, None)

            # target_kelrnel_val = kernel_model(org_adj, node_num)

            # batchSize = [org_adj.shape[0], org_adj.shape[1]]

            batchSize = [len(org_adj), org_adj[0].shape[0]]

            # org_adj_dgl = [dgl.from_scipy(sp.csr_matrix(graph.cpu().detach().numpy())) for graph in org_adj]
            [graph.setdiag(1) for graph in org_adj]
            org_adj_dgl = [dgl.from_scipy(graph) for graph in org_adj]
            org_adj_dgl = dgl.batch(org_adj_dgl).to(device)
            pos_wight = torch.true_divide(sum([x.shape[-1] ** 2 for x in subgraphs]) - subgraphs.sum(), subgraphs.sum())

            reconstructed_adj, prior_samples, post_mean, post_log_std, generated_kernel_val, reconstructed_adj_logit = model(
                org_adj_dgl.to(device), x_s.to(device), batchSize, subgraphs_indexes)
            kl_loss, reconstruction_loss, acc, kernel_cost, each_kernel_loss, log_sigma_values = OptimizerVAE(
                reconstructed_adj,
                generated_kernel_val,
                subgraphs.to(device),
                [val.to(device) for val in
                 target_kelrnel_val],
                post_log_std, post_mean, alpha,
                reconstructed_adj_logit,
                pos_wight, 2)

            loss = kernel_cost
            step += 1
            optimizer.zero_grad()
            loss.backward()

            if keepThebest and min_loss > loss:
                min_loss = loss.item()
                torch.save(model.state_dict(), "model")
            # torch.nn.utils.clip_grad_norm(model.parameters(),  1.0044e-05)
            optimizer.step()

            if (step + 1) % visulizer_step == 0 or epoch_number == epoch + 1:
                model.eval()
                mu, std, h = model.encode(org_adj_dgl.to(device), x_s.to(device), batchSize)
                save_path = f"{graph_save_path}{epoch}_feat.pt"
                torch.save(h, save_path)
                if True:
                    dir_generated_in_train = "generated_graph_train/"
                    if not os.path.isdir(dir_generated_in_train):
                        os.makedirs(dir_generated_in_train)
                    rnd_indx = random.randint(0, len(node_num) - 1)
                    sample_graph = reconstructed_adj[rnd_indx].cpu().detach().numpy()
                    sample_graph = sample_graph[:node_num[rnd_indx], :node_num[rnd_indx]]
                    sample_graph[sample_graph >= 0.5] = 1
                    sample_graph[sample_graph < 0.5] = 0

                    G = nx.from_numpy_matrix(sample_graph)

                    print("reconstructed graph vs Validation:")
                    logging.info("reconstructed graph vs Validation:")
                    reconstructed_adj = reconstructed_adj.cpu().detach().numpy()
                    reconstructed_adj[reconstructed_adj >= 0.5] = 1
                    reconstructed_adj[reconstructed_adj < 0.5] = 0
                    reconstructed_adj = [nx.from_numpy_matrix(reconstructed_adj[i]) for i in
                                         range(reconstructed_adj.shape[0])]
                    reconstructed_adj = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in
                                         reconstructed_adj if not nx.is_empty(G)]

                    target_set = [nx.from_numpy_matrix(val_adj[i].toarray()) for i in range(len(val_adj))]
                    target_set = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in target_set if
                                  not nx.is_empty(G)]
                    reconstruc_MMD_loss = mmd_eval(reconstructed_adj, target_set[:len(reconstructed_adj)], diam=True)
                    logging.info(reconstruc_MMD_loss)

                # todo: instead of printing diffrent level of logging shoud be used
                model.eval()
                if args.task == "graphGeneration":
                    # print("generated vs Validation:")
                    mmd_res = EvalTwoSet(model, val_adj[:1000], graph_save_path, device,Save_generated=True, _f_name=epoch)
                    with open(graph_save_path + '_MMD.log', 'a') as f:
                        f.write(str(step) + " @ loss @ , " + str(
                            loss.item()) + " , @ Reconstruction @ , " + reconstruc_MMD_loss + " , @ Val @ , " + mmd_res + "\n")

                    if ((step + 1) % visulizer_step * 2):
                        torch.save(model.state_dict(), graph_save_path + "model_" + str(epoch) + "_" + str(batch))
                stop = timeit.default_timer()
                # print("trainning time at this epoch:", str(stop - start))
                model.train()
            # if reconstruction_loss.item()<0.051276 and not swith:
            #     alpha[-1] *=2
            #     swith = True
            k_loss_str = ""
            for indx, l in enumerate(each_kernel_loss):
                k_loss_str += str(l) + ".   "

            print(
                "Epoch: {:03d} |Batch: {:03d} | loss: {:05f} | reconstruction_loss: {:05f} | z_kl_loss: {:05f} | accu: {:03f}".format(
                    epoch + 1, batch, loss.item(), reconstruction_loss.item(), kl_loss.item(), acc), k_loss_str)
            logging.info(
                "Epoch: {:03d} |Batch: {:03d} | loss: {:05f} | reconstruction_loss: {:05f} | z_kl_loss: {:05f} | accu: {:03f}".format(
                    epoch + 1, batch, loss.item(), reconstruction_loss.item(), kl_loss.item(), acc) + " " + str(
                    k_loss_str))
            # print(log_sigma_values)
            log_std = ""
            for indx, l in enumerate(log_sigma_values):
                log_std += "log_std "
                log_std += str(l) + ".   "
            print(log_std)
            logging.info(log_std)
            batch += 1
        # scheduler.step()
    model.eval()
    torch.save(model.state_dict(), graph_save_path + "model_hyp_" + str(epoch) + "_" + str(batch))


def load_data(args):
    dataset = args.dataset
    list_adj, list_x, list_label = list_graph_loader(dataset, return_labels=True)  # , _max_list_size=80

    if args.bfsOrdering == True:
        list_adj = BFS(list_adj)

    self_for_none = True
    if (args.decoder) in ("FCdecoder"):  # ,"FC_InnerDOTdecoder"
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


def model_parameter_preprocess(args):
    dataset = args.dataset
    kernl_type = []
    if args.model_vae == "graphHVAE" or args.model_vae == "graphVAE":
        alpha = [1, 1]
        step_num = 0

    AutoEncoder = False

    if AutoEncoder == True:
        alpha[-1] = 0

    if args.beta != None:
        alpha[-1] = args.beta

    print("kernl_type:" + str(kernl_type))
    print("alpha: " + str(alpha) + " num_step:" + str(step_num))
    return alpha, step_num, kernl_type


def graph_train(args):
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    keepThebest = False

    

    graph_save_path = args.graph_save_path
    graph_save_path = args.graph_save_path

    if graph_save_path == None:
        graph_save_path = './log/' + args.dataset + "/"
        args.graph_save_path = graph_save_path
    from pathlib import Path

    Path(graph_save_path).mkdir(parents=True, exist_ok=True)

# maybe to the beest way
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=graph_save_path + 'log.log', filemode='w', level=logging.INFO)

# **********************************************************************
# setting; general setting and hyper-parameters for each dataset
    print("KernelVGAE SETING: " + str(args))
    logging.info("KernelVGAE SETING: " + str(args))
    PATH = args.PATH  # the dir to save the with the best performance on validation data
    device = torch.device(args.device if torch.cuda.is_available() and args.UseGPU else "cpu")

    alpha, step_num, kernel_type = model_parameter_preprocess(args)
    list_graphs, list_test_graphs, val_adj, test_list_adj = load_data(args)

    SubGraphNodeNum = list_graphs.max_num_nodes
    in_feature_dim = list_graphs.feature_size  # ToDo: consider none Synthasis data
    nodeNum = list_graphs.max_num_nodes

    degree_center = torch.tensor([[x] for x in range(0, SubGraphNodeNum, 1)])
    degree_width = torch.tensor([[.1] for x in range(0, SubGraphNodeNum,
                                                 1)])  # ToDo: both bin's center and widtg also maximum value of it should be determinde auomaticly
# ToDo: both bin's center and widtg also maximum value of it should be determinde auomaticly

    bin_center = torch.tensor([[x] for x in range(0, SubGraphNodeNum, 1)])
    bin_width = torch.tensor([[1] for x in range(0, SubGraphNodeNum, 1)])

    kernel_model = kernel(device=device, kernel_type=kernel_type, step_num=step_num,
                      bin_width=bin_width, bin_center=bin_center, degree_bin_center=degree_center,
                      degree_bin_width=degree_width)
    torch.save(kernel_model,'kernel.pt')
    if args.encoder_type == "AvePool":
        encoder = AveEncoder(in_feature_dim, [256], args.graphEmDim)
    elif args.encoder_type == "HAvePool":
        encoder = HAveEncoder(in_feature_dim, [256], args.graphEmDim)
    else:
        print("requested encoder is not implemented")
        exit(1)

    if args.decoder == "FC":
        decoder = GraphTransformerDecoder_FC(args.graphEmDim, 256, nodeNum, args.directed)
    else:
        print("requested decoder is not implemented")
        exit(1)
    AutoEncoder = False
    model = kernelGVAE(kernel_model, encoder, decoder, AutoEncoder,
                   graphEmDim=args.graphEmDim)  # parameter namimng, it should be dimentionality of distriburion
    model.to(device)
    model_save_path = graph_save_path + 'model.pt'
    torch.save(model, model_save_path)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    num_nodes = list_graphs.max_num_nodes
# ToDo Check the effect of norm and pos weight

# target_kelrnel_val = kernel_model(target_adj)

    list_graphs.shuffle()
    start = timeit.default_timer()
# Parameters

    swith = False
    print(model)
    logging.info(model.__str__())
    
    self_for_none = True

    list_graphs.processALL(self_for_none=self_for_none)
    adj_list = list_graphs.get_adj_list()
    graphFeatures, _ = get_subGraph_features(args,adj_list, None, kernel_model)
    list_graphs.set_features(graphFeatures)
    train(args, list_graphs, alpha,self_for_none, decoder, device, model, optimizer,val_adj)
    # if args.task == "graphGeneration":
    #     EvalTwoSet(model, test_list_adj, graph_save_path, Save_generated=True, _f_name="final_eval")

# from config import parser

# args = parser.parse_args()
# graph_train(args)