
import networkx as nx
import scipy
import numpy as np
from  plotter import  plotG
import numpy
from operator import itemgetter
import random




def Synthetic_data(type= "grid", rand = False):
    if rand==True:
        if type == "grid":
            G = grid(random.randint(10,15), random.randint(10,15))
        elif type == "community":
            G = n_community([50, 50], p_inter=0.05)
        elif type == "ego":
            G = ego()
        elif type == "lobster":
            G = lobster()
        elif type == "multi_rel_com":
            G = multi_rel_com()
    else:
        numpy.random.seed(4812)
        np.random.RandomState(1234)
        random.seed(245)

        if type == "grid":
            G = grid()
        elif type== "community":
            G = n_community([50,50,50,50], p_inter=0.05)
        elif type == "ego":
            G = ego()
        elif type=="lobster":
            G=lobster()
        elif type =="multi_rel_com":
            G = multi_rel_com()

        plotG(G, type)
    return nx.adjacency_matrix(G), scipy.sparse.lil_matrix(scipy.sparse.identity(G.number_of_nodes()))

def grid(m= 10, n=10 ):
    # https: // networkx.github.io / documentation / stable / auto_examples / drawing / plot_four_grids.html
    G = nx.grid_2d_graph(m, n)  # 4x4 grid
    return G

def n_community(c_sizes, p_inter=0.1, p_intera=0.4):
    graphs = [nx.gnp_random_graph(c_sizes[i], p_intera, seed=i) for i in range(len(c_sizes))]
    G = nx.disjoint_union_all(graphs)
    communities = list(nx.connected_components(G))
    for i in range(len(communities)):
        subG1 = communities[i]
        nodes1 = list(subG1)
        for j in range(i+1, len(communities)):
            subG2 = communities[j]
            nodes2 = list(subG2)
            has_inter_edge = False
            for n1 in nodes1:
                for n2 in nodes2:
                    if np.random.rand() < p_inter:
                        G.add_edge(n1, n2)
                        has_inter_edge = True
            if not has_inter_edge:
                G.add_edge(nodes1[0], nodes2[0])
    # print('connected comp: ', len(list(nx.connected_components(G))))
    return G

def multi_rel_com(comunities =[[50,50,50,50],  [100,100]], graph_size= 200):
    """

    :param comunities: a list of lists, in which each list determine a seet of communities and the size of each one,
    the inter and intera edge probablity will be random.
    :node_num  the graph size
    :return:
    """
    graphs = []
    for community in comunities:
        graphs.append(ncommunity(community, graph_size, random.uniform(.0001,.01), random.uniform(.2,.7)))

    H = nx.compose(graphs[0], graphs[1])
    for i in range(2, len(graphs)):
        H = nx.compose(H, graphs[i])
    return H


def ncommunity(c_sizes, graph_size, p_inter=0.1, p_intera=0.4 ):
        graphs = [nx.gnp_random_graph(c_sizes[i], p_intera, seed=i) for i in range(len(c_sizes))]
        G = nx.disjoint_union_all(graphs)
        communities = list(nx.connected_components(G))
        for i in range(len(communities)):
            subG1 = communities[i]
            nodes1 = list(subG1)
            for j in range(i + 1, len(communities)):
                subG2 = communities[j]
                nodes2 = list(subG2)
                has_inter_edge = False
                for n1 in nodes1:
                    for n2 in nodes2:
                        if np.random.rand() < p_inter:
                            G.add_edge(n1, n2)
                            has_inter_edge = True
                if not has_inter_edge:
                    G.add_edge(nodes1[0], nodes2[0])

        x = list(range(graph_size))
        random.shuffle(x)

        if(len(G)> graph_size):
            G.add_nodes_from([i for i in range(len(G), graph_size)])
        mapping = {k: v for k, v in zip(list(range(graph_size)), x)}
        G = nx.relabel_nodes(G, mapping)
        return G

def lobster():
    p1 = 0.7
    p2 = 0.7
    mean_node = 80
    G = nx.random_lobster(mean_node, p1, p2)
    return G

def ego():
    # Create a BA model graph
    n = 2000
    m = 3
    G = nx.generators.barabasi_albert_graph(n, m)
    # find node with largest degree
    node_and_degree = G.degree()
    (largest_hub, degree) = sorted(node_and_degree, key=itemgetter(1))[-1]
    # Create ego graph of main hub
    hub_ego = nx.ego_graph(G, largest_hub)
    return hub_ego


if __name__ == '__main__':
    Synthetic_data("multi_rel_com")
    print("closed")
