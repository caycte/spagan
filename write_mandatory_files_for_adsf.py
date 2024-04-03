from utils import *

from rwr_process import *
import os
import numpy as np
import torch

def write_shortest_path_adj_matrix(dataset):
    x, y, tx, ty, allx, ally, graph = load_raw_data(dataset)
    G = nx.DiGraph(graph)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj=adj.astype(np.float32)
    adj = preprocess_adj(adj)
    adj_delta = torch.FloatTensor(np.array(adj.todense()))
    for node_i,path in tqdm(nx.shortest_path_length(G)):
        for node_j, path_len in path.items():                
            adj_delta[node_i][node_j] = path_len
    update_or_create_pickle("data/adsf/", f"dijkstra_{dataset}.pkl", adj_delta)