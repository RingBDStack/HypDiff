import concurrent.futures
from datetime import datetime
from functools import partial
import numpy as np
import networkx as nx
import os
import pickle as pkl
import subprocess
import time
import sys
import mmd_rnn as mmd
import pickle
from scipy.linalg import eigvalsh
PRINT_TIME = False


def degree_worker(G):
    return np.array(nx.degree_histogram(G))


def add_tensor(x, y):
    support_size = max(len(x), len(y))
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))
    return x + y


def degree_stats(graph_ref_list, graph_pred_list, is_parallel=False):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)

    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)
    print(len(sample_ref), len(sample_pred))
    # mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.gaussian_emd)
    mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.gaussian_tv)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist

def MMD_diam(graph_ref_list, graph_pred_list, is_parallel=False):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]


    for i in range(len(graph_ref_list)):
        try:
            degree_temp = np.array([nx.diameter(graph_ref_list[i])])
            sample_ref.append(degree_temp)
        except:
            print("An exception occurred; disconnected graph in ref set")
    for i in range(len(graph_pred_list_remove_empty)):
            try:
                degree_temp = np.array([nx.diameter(graph_pred_list_remove_empty[i])])
                sample_pred.append(degree_temp)
            except:
                print("An exception occurred; disconnected graph in gen set")
    mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.gaussian_tv, is_hist=False)
    return mmd_dist

def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist


def clustering_stats(graph_ref_list, graph_pred_list, bins=100, is_parallel=False):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)
        # check non-zero elements in hist
        # total = 0
        # for i in range(len(sample_pred)):
        #    nz = np.nonzero(sample_pred[i])[0].shape[0]
        #    total += nz
        # print(total)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)
    #
    # mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.gaussian_emd,
    #                            sigma=1.0 / 10, distance_scaling=bins)
    mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.gaussian_tv,
                               sigma=1.0 / 10)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing clustering mmd: ', elapsed)
    return mmd_dist


# maps motif/orbit name string to its corresponding list of indices from orca output
motif_to_indices = {
    '3path': [1, 2],
    '4cycle': [8],
}
COUNT_START_STR = 'orbit counts: \n'


def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):
    tmp_fname = 'eval/orca/tmp.txt'
    f = open(tmp_fname, 'w')
    f.write(str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
    for (u, v) in edge_list_reindexed(graph):
        f.write(str(u) + ' ' + str(v) + '\n')
    f.close()

    output = subprocess.check_output(['./eval/orca/orca', 'node', '4', 'eval/orca/tmp.txt', 'std'])
    output = output.decode('utf8').strip()

    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR)
    output = output[idx:]
    node_orbit_counts = np.array([list(map(int, node_cnts.strip().split(' ')))
                                  for node_cnts in output.strip('\n').split('\n')])

    try:
        os.remove(tmp_fname)
    except OSError:
        pass

    return node_orbit_counts


def motif_stats(graph_ref_list, graph_pred_list, motif_type='4cycle', ground_truth_match=None, bins=100):
    # graph motif counts (int for each graph)
    # normalized by graph size
    total_counts_ref = []
    total_counts_pred = []

    num_matches_ref = []
    num_matches_pred = []

    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]
    indices = motif_to_indices[motif_type]
    for G in graph_ref_list:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)

        if ground_truth_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == ground_truth_match:
                    match_cnt += 1
            num_matches_ref.append(match_cnt / G.number_of_nodes())

        # hist, _ = np.histogram(
        #        motif_counts, bins=bins, density=False)
        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_ref.append(motif_temp)

    for G in graph_pred_list_remove_empty:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)

        if ground_truth_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == ground_truth_match:
                    match_cnt += 1
            num_matches_pred.append(match_cnt / G.number_of_nodes())

        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_pred.append(motif_temp)

    mmd_dist = mmd.compute_mmd(total_counts_ref, total_counts_pred, kernel=mmd.gaussian,
                               is_hist=False)
    # print('-------------------------')
    # print(np.sum(total_counts_ref) / len(total_counts_ref))
    # print('...')
    # print(np.sum(total_counts_pred) / len(total_counts_pred))
    # print('-------------------------')
    return mmd_dist


# this functione is used to calculate some of the famous graph properties
def MMD_triangles(graph_ref_list, graph_pred_list ):
    """

    :param list_of_adj: list of nx arrays
    :return:
    """
    total_counts_pred = []
    for graph in graph_pred_list:
        total_counts_pred.append([np.sum(list(nx.triangles(graph).values()))/graph.number_of_nodes()])

    total_counts_ref = []
    for graph in graph_ref_list:
        total_counts_ref.append([np.sum(list(nx.triangles(graph).values()))/graph.number_of_nodes()])

    total_counts_pred = np.array(total_counts_pred)
    total_counts_ref = np.array(total_counts_ref)
    mmd_dist = mmd.compute_mmd(total_counts_ref, total_counts_pred, kernel=mmd.gaussian_tv,
                               is_hist=False, sigma=30.0)
    print("averrage number of tri in ref/ test: ", str(np.average(total_counts_pred)), str(np.average(total_counts_ref)))
    return mmd_dist



def sparsity_stats_all(graph_ref_list, graph_pred_list):
    def sparsity(G):
        return (G.number_of_nodes()**2-len(G.edges))/ G.number_of_nodes()**2

    def edge_num(G):
        return len(G.edges)

    total_counts_ref = []
    total_counts_pred = []

    edge_num_ref = []
    edge_num_pre = []
    for G in graph_ref_list:
        sp = sparsity(G)
        total_counts_ref.append([sp])
        edge_num_ref.append(edge_num(G))

    for G in graph_pred_list:
        sp = sparsity(G)
        total_counts_pred.append([sp])
        edge_num_pre.append(edge_num(G))

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)
    mmd_dist = mmd.compute_mmd(total_counts_ref, total_counts_pred, kernel=mmd.gaussian_tv,
                               is_hist=False, sigma=30.0)

    print('-------------------------')
    print(np.sum(total_counts_ref, axis=0) / len(total_counts_ref))
    print('...')
    print(np.sum(total_counts_pred, axis=0) / len(total_counts_pred))
    print('-------------------------')
    print("average edge # in test set:")
    print(np.average(edge_num_ref))
    print("average edge # in generated set:")
    print(np.average(edge_num_pre))
    print('-------------------------')

    return mmd_dist, np.average(edge_num_ref), np.average(edge_num_pre)

def orbit_stats_all(graph_ref_list, graph_pred_list):
    total_counts_ref = []
    total_counts_pred = []

    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    for G in graph_ref_list:
        try:
            orbit_counts = orca(G)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list:
        try:
            orbit_counts = orca(G)
        except:
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)
    mmd_dist = mmd.compute_mmd(total_counts_ref, total_counts_pred, kernel=mmd.gaussian_tv,
                               is_hist=False, sigma=30.0)
    # mmd_dist = mmd.compute_mmd(total_counts_ref, total_counts_pred, kernel=mmd.gaussian,
    #                            is_hist=False, sigma=30.0)
    print('-------------------------')
    print(np.sum(total_counts_ref, axis=0) / len(total_counts_ref))
    print('...')
    print(np.sum(total_counts_pred, axis=0) / len(total_counts_pred))
    print('-------------------------')
    return mmd_dist

# This function takes two list of networkx2 objects and compare their mmd for preddefined statistics
def mmd_eval(generated_graph_list, original_graph_list, diam = False):
    generated_graph_list = [G for G in generated_graph_list if not G.number_of_nodes() == 0]
    for G in generated_graph_list:
        G.remove_edges_from(nx.selfloop_edges(G))

    for G in original_graph_list:
        G.remove_edges_from(nx.selfloop_edges(G))
    # removing emty graph
    tmp_generated_graph_list = []
    for G in generated_graph_list:
        if G.number_of_nodes()>0:
            tmp_generated_graph_list.append(G)
    generated_graph_list = tmp_generated_graph_list
    mmd_degree = degree_stats(original_graph_list, generated_graph_list)
    # try:
    #     mmd_4orbits = orbit_stats_all(original_graph_list, generated_graph_list)
    # except :
    #     print("Unexpected error:", sys.exc_info()[0])
    #     mmd_4orbits = -1
    mmd_clustering = clustering_stats(original_graph_list, generated_graph_list)
    mmd_sparsity, degree1, degree2 = sparsity_stats_all(original_graph_list, generated_graph_list)
    mmd_spectral = spectral_stats(original_graph_list, generated_graph_list)

    print('degree', mmd_degree, 'clustering', mmd_clustering, "Spec:", mmd_spectral)
    return('degree: '+str(mmd_degree) +' clustering: ' + str(mmd_clustering) +" Spec: "+str( mmd_spectral) + " average edge # in test set: "+ str(degree1) + " average edge # in generated set: "+ str(degree2))
# load a list of graphs
def load_graph_list(fname,remove_self=True):
    with open(fname, "rb") as f:
        glist =  np.load(f, allow_pickle=True)
    # np.save(fname+'Lobster_adj.npy', glist, allow_pickle=True)
    graph_list =[]
    for G in glist:
        if type(G) == np.ndarray:
            graph = nx.from_numpy_matrix(G)
        elif type(G)==nx.classes.graph.Graph:
            graph = G
        else:
            graph = nx.Graph()
            if len(G[0])>0:
                graph.add_nodes_from(G[0])
                graph.add_edges_from(G[1])
            else:
                continue

        if remove_self:
            graph.remove_edges_from(nx.selfloop_edges(graph))
        graph.remove_nodes_from(list(nx.isolates(graph)))
        Gcc = sorted(nx.connected_components(graph), key=len, reverse=True)
        graph = graph.subgraph(Gcc[0])
        graph = nx.Graph(graph)
        graph_list.append(graph)
    return graph_list


def spectral_worker(G):
  # eigs = nx.laplacian_spectrum(G)
  eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
  spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
  spectral_pmf = spectral_pmf / spectral_pmf.sum()
  # from scipy import stats
  # kernel = stats.gaussian_kde(eigs)
  # positions = np.arange(0.0, 2.0, 0.1)
  # spectral_density = kernel(positions)

  # import pdb; pdb.set_trace()
  return spectral_pmf

def spectral_stats(graph_ref_list, graph_pred_list, is_parallel=False):
  ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
  sample_ref = []
  sample_pred = []
  # in case an empty graph is generated
  graph_pred_list_remove_empty = [
      G for G in graph_pred_list if not G.number_of_nodes() == 0
  ]

  prev = datetime.now()
  if is_parallel:
    with concurrent.futures.ThreadPoolExecutor() as executor:
      for spectral_density in executor.map(spectral_worker, graph_ref_list):
        sample_ref.append(spectral_density)
    with concurrent.futures.ThreadPoolExecutor() as executor:
      for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
        sample_pred.append(spectral_density)

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #   for spectral_density in executor.map(spectral_worker, graph_ref_list):
    #     sample_ref.append(spectral_density)
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #   for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
    #     sample_pred.append(spectral_density)
  else:
    for i in range(len(graph_ref_list)):
      spectral_temp = spectral_worker(graph_ref_list[i])
      sample_ref.append(spectral_temp)
    for i in range(len(graph_pred_list_remove_empty)):
      spectral_temp = spectral_worker(graph_pred_list_remove_empty[i])
      sample_pred.append(spectral_temp)
  # print(len(sample_ref), len(sample_pred))

  # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
  # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
  # mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.gaussian_emd,
  #                 sigma=1.0 / 10, distance_scaling=bins)
  mmd_dist =  mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.gaussian_tv)
  # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

  elapsed = datetime.now() - prev
  if PRINT_TIME:
    print('Time computing degree mmd: ', elapsed)
  return mmd_dist


if __name__ == '__main__':
    # # dataset = "cora"
    # # model = ["graphit","arga"]
    # # for m in model:
    # #     print(m)
    # #     adj_orig = np.load('parmis/'+dataset+'_'+m+'/adj_orig.npy')
    # #     generated_graphs = np.load('parmis/'+dataset+'_'+m+'/adj_rec.npy')
    # #     mmd_eval([nx.from_numpy_matrix(generated_graphs)], [nx.from_numpy_matrix(adj_orig)])
    #
    from  data import  *
    # generated_graph = np.load("result_final/NetGan/adj_lobster_3000")

    # p1 = 0.7
    # p2 = 0.7
    # mean_node = 80
    # G = nx.random_lobster(mean_node, p1, p2)
    #
    # mmd_eval([nx.from_numpy_matrix(generated_graph)], [G])
    #
    # #==========================
    # G = grid(random.randint(15, 15), random.randint(15, 15))
    # generated_graph = np.load("result_final/NetGan/adj_grid_3000")
    # mmd_eval([nx.from_numpy_matrix(generated_graph)], [G])

    # C:\git\Graph - Generative - Models\results\GRAPHRNN
    # pred_fname = "/local-scratch/kiarash/GraphENN_remote_c/Reprted/GRIDRNN-rnn/crossEntropy_bestLR001_GraphRNN_RNN_grid_4_128_pred_3000_1.dat_nx22_"
    # fname = "/local-scratch/kiarash/GraphENN_remote_c/Reprted/GRIDRNN-rnn/crossEntropy_bestLR001_GraphRNN_RNN_grid_4_128_pred_3000_1.dat_nx22_"
    # pred_fname = "/local-scratch/kiarash/GraphENN_remote_c/Reprted/GraphRNN_RNN_grid_4_128_pred_3000_1.dat_nx22_"
    # dataset = "grid"
    # pred_fname = "/local-scratch/kiarash/GGM-metrics/Kernel_paper_Results/GRAN_grid__gen_adj3000.npy"
    #lobster rnn-rnn
    # pred_fname = "/local-scratch/kiarash/GraphENN_remote_c/Reprted/GRIDRNN/crossEntropy_bestLR001_GraphRNN_MLP_grid_4_128_pred_3000_3.dat_nx22_"
    # # lobster rnn-mlp
    # pred_fname = "/local-scratch/kiarash/GraphENN_remote_c/Reprted/lobsterMLP/crossEntropy_bestLR001_GraphRNN_MLP_lobster_4_128_pred_3000_3.dat_nx22_"
    dataset = "lobster"
    #------------------------------------
    #OGB
    # pred_fname = "/local-scratch/kiarash/GraphENN_remote_c/Reprted/crossEntropy_bestLR001_GraphRNN_MLP_ogbg-molbbbp_4_128_pred_3000_3.dat_nx22_"
    # pred_fname = "/local-scratch/kiarash/GraphENN_remote_c/Reprted/OGB-RNN-RNN/crossEntropy_bestLR001_GraphRNN_RNN_ogbg-molbbbp_4_128_pred_3000_1.dat_nx22_"
    # pred_fname = "/local-scratch/kiarash/GGM-metrics/Kernel_paper_Results/GRAN_molbbbp__gen_adj3000.npy"
    # dataset = "DD"

    #
    #
    # # Lobster
    # pred_fname = "/local-scratch/kiarash/Baseline_GRAN/GRAN/exp/Reported/GRANMixtureBernoulli_grid_2022-Mar-01-12-46-22_263643/GRAN_grid__test_test_adj3000.npy"
    #
    # dataset = "grid"
    # # Lobster
    # pred_fname = "/local-scratch/kiarash/GGM-metrics/Kernel_paper_Results/crossEntropy_bestLR001_GraphRNN_MLP_lobster_4_128_pred_3000_3.dat_nx22_"
    #
    # wheel_graph
    # pred_fname = "/local-scratch/kiarash/AAAI/Graph-Generative-Models/MMD_Nips2022/MMD_AvePool_FC_wheel_graph_graphGeneration_TotalNumberOfTrianglesBFSTrue200001649443496.1055677/generatedGraphs_adj.npy"
    # dataset = "wheel_graph"
    plot_the_graphs = False

    models = []
    # rnn DD
    test_fname = "/local-scratch/kiarash/google-research/bigg/data/DD/test.npy"
    pred_fname = "/local-scratch/kiarash/GraphENN_remote_c/Reprted/DD-RNN/crossEntropy_bestLR001_GraphRNN_RNN_DD_4_128_pred_600_1.dat_nx22_"
    models.append([test_fname, pred_fname, None])

    # DD BiGG
    test_fname = "/local-scratch/kiarash/google-research/bigg/data/DD/test.npy"
    pred_fname = "/local-scratch/kiarash/google-research/bigg/data/DD/epoch-1000.ckpt.graphs-0"
    models.append([test_fname, pred_fname, None])

    test_fname = "/local-scratch/kiarash/google-research/bigg/data/tri/test.npy"
    pred_fname = "/local-scratch/kiarash/GraphENN_remote_c/Reprted/tri-RNN/crossEntropy_bestLR001_GraphRNN_RNN_tri-grid_4_128_pred_3000_1.dat_nx22_"
    models.append([test_fname, pred_fname, None])

    test_fname = "/local-scratch/kiarash/google-research/bigg/data/tri/test.npy"
    pred_fname = "/local-scratch/kiarash/GraphENN_remote_c/Reprted/tri-MLP/crossEntropy_bestLR001_GraphRNN_MLP_tri-grid_4_128_pred_3000_3.dat_nx22_"
    models.append([test_fname, pred_fname, None])


    test_fname = "/local-scratch/kiarash/google-research/bigg/data/tri/test.npy"
    pred_fname = "/local-scratch/kiarash/google-research/bigg/data/tri/epoch-1000.ckpt.graphs-0"
    models.append([test_fname, pred_fname, None])


    test_fname = "/local-scratch/kiarash/AAAI/Graph-Generative-Models/FinalResultHopefully/MMD_AvePool_FC_DD_graphGeneration_kipfBFSTrue200001647138192.79155/completeView/testGraphs_adj.npy"
    pred_fname = "/local-scratch/kiarash/AAAI/Graph-Generative-Models/FinalResultHopefully/MMD_AvePool_FC_DD_graphGeneration_kipfBFSTrue200001647138192.79155/completeView/Single_comp_generatedGraphs_adj.npy"
    models.append([test_fname, pred_fname, None])

    test_fname = "/local-scratch/kiarash/Baseline_GRAN/GRAN/exp/Reported/GRANMixtureBernoulli_triangular_grid_2022-May-08-15-01-34_348873/GRAN_triangular_grid__test_test_adj.npy"
    pred_fname = "/local-scratch/kiarash/GraphENN_remote_c/Reprted/tri-RNN/crossEntropy_bestLR001_GraphRNN_RNN_tri-grid_4_128_pred_3000_1.dat_nx22_"
    models.append([test_fname, pred_fname, None])

    test_fname = "/local-scratch/kiarash/Baseline_GRAN/GRAN/exp/Reported/GRANMixtureBernoulli_triangular_grid_2022-May-08-15-01-34_348873/GRAN_triangular_grid__test_test_adj.npy"
    pred_fname = "/local-scratch/kiarash/GraphENN_remote_c/graphs/crossEntropy_bestLR001_GraphRNN_MLP_tri-grid_4_128_pred_3000_3.dat_nx22_"
    models.append([test_fname, pred_fname, None])

    test_fname = "/local-scratch/kiarash/Baseline_GRAN/GRAN/exp/Reported/GRANMixtureBernoulli_triangular_grid_2022-May-08-15-01-34_348873/GRAN_triangular_grid__test_test_adj.npy"
    pred_fname = "/local-scratch/kiarash/Baseline_GRAN/GRAN/exp/Reported/GRANMixtureBernoulli_triangular_grid_2022-May-08-15-01-34_348873/GRAN_triangular_grid__gen_adj.npy"
    models.append([test_fname, pred_fname, None])
    
    test_fname = "/local-scratch/kiarash/AAAI/Graph-Generative-Models/FinalResultHopefully/MMD_AvePool_FC_triangular_grid_graphGeneration_kipfBFSTrue200001651972897.5996404/testGraphs_adj_.npy"
    pred_fname = "/local-scratch/kiarash/AAAI/Graph-Generative-Models/FinalResultHopefully/MMD_AvePool_FC_triangular_grid_graphGeneration_kipfBFSTrue200001651972897.5996404/_triangular_grid_FC_kipf_graphGeneration"
    models.append([test_fname, pred_fname, None])

    # grid kernel tri
    test_fname = "/local-scratch/kiarash/MMD_AvePool_FC_lobster_graphGeneration_KernelAugmentedWithTotalNumberOfTrianglesBFSTrue200001651430752.5701604/testGraphs_adj_.npy"
    pred_fname = "/local-scratch/kiarash/MMD_AvePool_FC_lobster_graphGeneration_KernelAugmentedWithTotalNumberOfTrianglesBFSTrue200001651430752.5701604/Single_comp_generatedGraphs_adj_17998.npy"
    # pred_fname = "/local-scratch/kiarash/MMD_AvePool_FC_grid_graphGeneration_KernelAugmentedWithTotalNumberOfTrianglesBFSTrue200001651363794.9637518/Single_comp_generatedGraphs_adj_15998.npy"
    models.append([test_fname, pred_fname, None])

    # grid kernel tri
    test_fname = "/local-scratch/kiarash/AAAI/Graph-Generative-Models/MMD_AvePool_FC_grid_graphGeneration_KernelAugmentedWithTotalNumberOfTrianglesBFSTrue100001651273589.400573/testGraphs_adj_.npy"
    pred_fname = "/local-scratch/kiarash/MMD_AvePool_FC_grid_graphGeneration_KernelAugmentedWithTotalNumberOfTrianglesBFSTrue200001651363794.9637518/Single_comp_generatedGraphs_adj_18998.npy"
    # pred_fname = "/local-scratch/kiarash/MMD_AvePool_FC_grid_graphGeneration_KernelAugmentedWithTotalNumberOfTrianglesBFSTrue200001651363794.9637518/Single_comp_generatedGraphs_adj_15998.npy"
    models.append([test_fname, pred_fname, None])

# GRAN DD
    pred_fname = "/local-scratch/kiarash/Baseline_GRAN/GRAN/GRAN_DD__gen_adj.npy"
    test_fname = "/local-scratch/kiarash/Baseline_GRAN/GRAN/GRAN_DD__test_test_adj.npy"
    models.append([test_fname, pred_fname, None])

    # GRAN DD
    pred_fname  = "/local-scratch/kiarash/Baseline_GRAN/GRAN/exp/Reported/GRANMixtureBernoulli_DD_2022-Mar-04-17-42-36_40882/GRAN_DD__gen_adj.npy"
    test_fname= "/local-scratch/kiarash/Baseline_GRAN/GRAN/exp/Reported/GRANMixtureBernoulli_DD_2022-Mar-04-17-42-36_40882/GRAN_DD__test_test_adj.npy"
    models.append([test_fname,pred_fname,  None])


    # GRAN DD
    pred_fname = "/local-scratch/kiarash/AAAI/Graph-Generative-Models/MMD_Reported/MMD_AvePool_FC_DD_graphGeneration_kernelBFSTrue200001650675057.502729/Single_comp_generatedGraphs_adj_final_eval.npy"
    models.append([None ,pred_fname,  "DD"])



    # graphrnnMLP DD
    test_fname = "/local-scratch/kiarash/GraphENN_remote_c/Reprted/DD-MLP/crossEntropy_bestLR001_GraphRNN_MLP_DD_4_128_test_0.dat_nx22_"
    pred_fname = "/local-scratch/kiarash/GraphENN_remote_c/Reprted/DD-MLP/crossEntropy_bestLR001_GraphRNN_MLP_DD_4_128_pred_3000_3.dat_nx22_"
    models.append([None,pred_fname, "DD"])


    # lobster bigg
    test_fname = "/local-scratch/kiarash/google-research/bigg/data/lobster_Kernel/test.npy"
    pred_fname = "/local-scratch/kiarash/google-research/bigg/data/lobster_Kernel/epoch-1000.ckpt.graphs-0.npy"
    models.append([test_fname, pred_fname, None])

    # grid bigg
    test_fname = "/local-scratch/kiarash/google-research/bigg/data/grid-BFS/test.npy"
    pred_fname = "/local-scratch/kiarash/google-research/bigg/data/grid-BFS/generated.npy"
    models.append([test_fname, pred_fname, None])

    # grid kernel
    pred_fname = "/local-scratch/kiarash/AAAI/Graph-Generative-Models/MMD_AvePool_FC_grid_graphGeneration_kernelBFSTrue200001650422195.1629238Candidate/Single_comp_generatedGraphs_adj_19998.npy"
    models.append([None, pred_fname, "grid"])

    # big obg
    test_fname = "/local-scratch/kiarash/google-research/bigg/data/ogbg-molbbbp/test.npy"
    pred_fname = "/local-scratch/kiarash/google-research/bigg/data/ogbg-molbbbp/generated.npy"
    models.append([test_fname,pred_fname, None])



    pred_fname = "/local-scratch/kiarash/AAAI/Graph-Generative-Models/MMD_AvePool_FC_lobster_graphGeneration_kernelBFSTrue200001650320718.923034/Single_comp_generatedGraphs_adj_15998.npy"
    models.append([None, pred_fname, "lobster"])


#ReportedResult\PTC_result\BIGG\PTC_lattice_graph\test-graphs.pkl
    models = []
    test_fname = "ReportedResult/PTC_result/BIGG/PTC_lattice_graph/test.npy"
    pred_fname = "ReportedResult/PTC_result/GraphRNN/crossEntropy_bestLR001_GraphRNN_RNN_PTC_4_128_pred_300_1.dat_nx22_"

    models = []
    test_fname = "ReportedResult/IMDb/bigg/test.npy"
    pred_fname = "ReportedResult/IMDb/bigg/generated.npy"
    models = []
    test_fname = "ReportedResult/IMDb/bigg/test.npy"
    pred_fname = "ReportedResult/IMDb/GraphRNN/crossEntropy_bestLR001_GraphRNN_RNN_IMDBBINARY_4_128_pred_3000_1.dat_nx22_"
    models = []
    pred_fname = "ReportedResult/MUTAG_lattice_graph/generated.npy"
    test_fname = "ReportedResult/MUTAG_lattice_graph/test.npy"

    models = []
    pred_fname = "ReportedResult/MUTAG/GRAPHRNN/crossEntropy_bestLR001_GraphRNN_MLP_MUTAG_4_128_pred_3000_1.dat_nx22_"
    test_fname = "ReportedResult/MUTAG/BIG_MUTAG_lattice_graph/test.npy"

    models = []
    pred_fname = "ReportedResult/MUTAG/GRAPHRNN/crossEntropy_bestLR001_GraphRNN_RNN_MUTAG_4_128_pred_3000_1.dat_nx22_"
    test_fname = "ReportedResult/MUTAG/BIG_MUTAG_lattice_graph/test.npy"

    models = []
    pred_fname = "ReportedResult/MUTAG/GRAPHRNN/crossEntropy_bestLR001_GraphRNN_RNN_MUTAG_4_128_pred_3000_1.dat_nx22_"
    test_fname = "ReportedResult/MUTAG/GRAN/GRAN_MUTAG_lattice_graph__gen_adj.npy"

    models.append([test_fname,pred_fname, None])
    #---------------------------------------------
    # big-lobsr




    models.append([test_fname,pred_fname, None])
    # load the data

    for model in models:
        if model[-1]!= None:
            list_adj, _ = list_graph_loader(model[-1])
            _, test_list_adj, _, _ = data_split(list_adj)
            test_list_adj = [nx.from_numpy_matrix(graph.toarray()) for graph in test_list_adj]
        else:
            test_list_adj = load_graph_list(model[0])

        generated_graphs = load_graph_list(model[1])
        generated_graphs = generated_graphs[:len(test_list_adj)]
        np.save(model[1]+".npy", generated_graphs, allow_pickle=True)
        np.save(model[0]+".npy", test_list_adj, allow_pickle=True)







        Visualize = True
        import plotter

        if (Visualize):
            import plotter

            for i, G in enumerate(generated_graphs[:20]):
                plotter.plotG(G, "generated", file_name=model[1] + "_" + str(i) + ".png", plot_it=True)

        if(Visualize):

            for i,G in enumerate(test_list_adj[:20]):
                # G = nx.from_scipy_sparse_matrix(G)
                plotter.plotG(G, "test",  file_name=model[1]+"__test__"+str(i)+".png")






        mmd_eval(generated_graphs, test_list_adj, True)
        print("=============================================================================")
