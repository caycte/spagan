import numpy as np
import scipy.sparse as sp
import torch
import pickle
import time
import os
import graph_tool.topology as g_topo
import graph_tool.generation as g_gen
import networkx as nx
import sys
from tqdm import tqdm


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum+9e-15, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized

def load_data_orggcn(dataset_str):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pickle.load(f, encoding='latin1'))
            else:
                objects.append(pickle.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[:,5] = 1   # just add fake label for np.where(labels)
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
    
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    features = normalize(features)


    features = torch.FloatTensor(np.array(features.todense()))

    
    print('node number:', features.shape[0])
    print('training sample:',sum(train_mask))
    print('validation sample:',sum(val_mask))
    print('testing sample:',sum(test_mask))
    idx_train = torch.LongTensor(np.where(train_mask)[0])
    idx_val = torch.LongTensor(np.where(val_mask)[0])
    idx_test = torch.LongTensor(np.where(test_mask)[0])
    

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    adj=adj.astype(np.float32)
    adj_ad1 = adj
    adj_sum_ad = np.sum(adj_ad1, axis=0)
    adj_sum_ad = np.asarray(adj_sum_ad)
    adj_sum_ad = adj_sum_ad.tolist()
    adj_sum_ad = adj_sum_ad[0]
    adj_ad_cov = adj
    Mc = adj_ad_cov.tocoo()
    adj = preprocess_adj(adj)
    adj_delta = torch.FloatTensor(np.array(adj.todense()))
    # caculate n-hop neighbors
    G = nx.DiGraph(graph)
    #inf= pickle.load(open('adj_citeseer.pkl', 'rb'))
   
    print('astar length')
    for node_i,path in tqdm(nx.shortest_path_length(G)):
        for node_j, path_len in path.items():                
            adj_delta[node_i][node_j] = path_len
    a = open("dijskra_citeseer.pkl", 'wb')
    pickle.dump(adj_delta, a)
   #######


    fw = open('ri_index_c_0.5_citeseer_highorder_1_x_abs.pkl', 'rb')
    ri_index = pickle.load(fw)
    fw.close()

    fw = open('ri_all_c_0.5_citeseer_highorder_1_x_abs.pkl', 'rb')
    ri_all = pickle.load(fw)
    fw.close()
    # Evaluate structural interaction between the structural fingerprints of node i and j
    adj_delta = structural_interaction(ri_index, ri_all, adj_delta)
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features,idx_train, idx_val, idx_test, train_mask, val_mask, test_mask,labels,adj_delta



def single_gen_path(pathRes, shape, pathLen=3, Nratio=0.6, Ndeg=2):
    cols = []
    rows = []
    pathValues = []

    for row in pathRes.keys():
        degree = 0
        tmppath = []; tmplen = []; tmpcols = []; tmprows = []
        dists = pathRes[row][0]
        paths = pathRes[row][1]
        for col in paths.keys():
            path = paths[col]
            if len(path)==pathLen:
                tmpcols.append(col)
                tmprows.append(row)
                tmplen.append(dists[col])
                tmppath.append(path)
            if len(path)==2:
                degree += 1
        if pathLen==2:
            ratio = 1.0
            maxRange = int(ratio*len(tmplen))
        else:
            ratio = Nratio
            maxRange = int( min([degree*Ndeg ,int(ratio*len(tmplen))]) )
            
        topk = sorted( range(len(tmplen)), key=lambda i:tmplen[i])[ 0: maxRange]
        
        pathValues = pathValues +  [tmppath[i] for i in topk]
        
        cols = cols + [tmpcols[i] for i in topk]
        rows = rows + [tmprows[i] for i in topk]

    cols = cols + [i for i in range(shape[0])]
    rows = rows + [i for i in range(shape[0])]
    pathValues = pathValues + [[j]*pathLen for j in range(shape[0])]

    cols = np.array(cols).reshape(1,-1)
    rows = np.array(rows).reshape(1,-1)
    pathValues = np.array(pathValues)
    i = np.concatenate((rows, cols), axis=0)
    i = torch.LongTensor(i)
    
    pathV = torch.FloatTensor(pathValues)
    
    pathM = torch.sparse.FloatTensor(i, pathV, torch.Size([shape[0], shape[1], pathLen]))
    
    return pathM, 0


def load_pathm(dataset_str='cora', matpath=''):
    with open(os.path.join(matpath, 'pathDict.pickle'), 'rb') as pathfile:
        pathM = pickle.load(pathfile)
    for layer in pathM.keys():
            for head in pathM[layer].keys():
                for pathlen in pathM[layer][head].keys():
                    pathM[layer][head][pathlen]['indices'] \
                        = pathM[layer][head][pathlen]['indices'].long()
                    pathM[layer][head][pathlen]['values'] \
                        = pathM[layer][head][pathlen]['values'].long()

    return pathM


def graph_tool_apsp(spmatrix, cutoff=3):
    nodeN = spmatrix.shape[0]
    edgeN = spmatrix.data.shape[0]
    weightM = spmatrix.data
    print(nodeN, edgeN)
    
    t = time.time()
    g = g_gen.Graph()
    g.add_vertex(nodeN)
    row = spmatrix.row.reshape(-1,1)
    col = spmatrix.col.reshape(-1,1)
    edge_list = np.hstack((row, col)).tolist()
    g.add_edge_list(edge_list)

    weights = g.new_edge_property("double")
    for i in range(edgeN):
        weights[g.vertex(row[i]),g.vertex(col[i])] = weightM[i]
    print("construct time: {:.4f}".format(time.time()-t))

    pathRes = {}
    for centerNode in range(nodeN):
        #print('\rnode:{:07d}'.format(centerNode))
        pathOneRes = {}
        distDict = {}
        pathDict = {}

        dist, pred = g_topo.shortest_distance(g, source=g.vertex(centerNode),  weights=weights,pred_map=True)
        dist = dist.a  # to list
        pred = pred.a  # to list
        pathDict[centerNode] = [centerNode]
        pathkeys = set(pathDict.keys())
        for i in range(1,cutoff):   # generate path with length at most cutoff
            newpathkeys = []
            for col in range(nodeN):
                if col!=centerNode and pred[col] in pathkeys:
                    pathDict[col] = pathDict[pred[col]]+[col]
                    distDict[col] = dist[col]
                    newpathkeys.append(col)
            pathkeys = set(newpathkeys)
        
        pathOneRes[0] = distDict
        pathOneRes[1] = pathDict
        pathRes[centerNode] = pathOneRes

    return pathRes

def gen_pathm(nheads=[8], matpath=None, Nratio=0.6, Ndeg=2):     # nheads=[8]: does not generate for the last layer
    # generate path matrix for each attention head
    # pathM is a dict of dict of dict, each element is a pytorch sparse matrix
    #   the first key is which 'layer' (start from 0)
    #   the second key is which 'head'  (start from 0)
    #   the third key is which 'path length' (start from 2)
    pathM = {}
    layerPathM = {}
    for layer, nhead in enumerate(nheads):
        #if layer>0:
        #    continue
        headPathM = {}
        headMatrix = None
        for head in range(nhead):
            if not matpath:
                spmatrix = sp.load_npz('models/exp1/attmat_{:d}_{:d}.npz'.format(layer+1, head))
            else:
                spmatrix = sp.load_npz(matpath+'/attmat_{:d}_{:d}.npz'.format(layer+1, head))    
            if headMatrix is None:
                headMatrix = spmatrix
            else:
                headMatrix.data += spmatrix.data
        headMatrix.data = headMatrix.data / nhead
        
        headMatrix.data = -headMatrix.data
        headMatrix.setdiag(0)       # not include self attention weight

        attMin = min(headMatrix.data)
        attMax = max(headMatrix.data)
        headMatrix.data = (headMatrix.data - attMin)/(attMax-attMin)
        
        print(min(headMatrix.data), max(headMatrix.data))
        
        #G = nx.from_scipy_sparse_matrix(headMatrix)
        t = time.time()
        #pathRes = dict(nx.all_pairs_dijkstra(G, cutoff=3))
        pathRes = graph_tool_apsp(headMatrix, cutoff=3)
        print('shortest path time:', time.time()-t)
        
        # single_path: sparse matrix N*N*pathLen
        t = time.time()
        lenPathM = {}
        for pathLen in range(2,4):  # generate path 2,3,...
            indexValue = {}
            single_path, _ = single_gen_path(pathRes, headMatrix.shape, pathLen=pathLen, Nratio=Nratio, Ndeg=Ndeg)
            indexValue['indices'] = single_path._indices()
            indexValue['values'] = single_path._values()
            print("pathlen:{:d}, #{:d}".format(pathLen, indexValue['indices'].shape[1]))
            lenPathM[pathLen] = indexValue
        headPathM[0] = lenPathM
        print('generate path time:', time.time()-t)
        
        layerPathM[layer] = headPathM
    pathM = layerPathM
    # pytorch doesn't support sparse matrix save, so we need to save it as indices and values
    with open(os.path.join(matpath, 'pathDict.pickle'), 'wb') as pathfile:
        pickle.dump(pathM, pathfile)
        print('pathM saved')
    return pathM


def structural_interaction(ri_index, ri_all, g):
    """structural interaction between the structural fingerprints for citeseer"""
    for i in range(len(ri_index)):
        for j in range(len(ri_index)):
            intersection = set(ri_index[i]).intersection(set(ri_index[j]))
            union = set(ri_index[i]).union(set(ri_index[j]))
            intersection = list(intersection)
            union = list(union)
            intersection_ri_alli = []
            intersection_ri_allj = []
            union_ri_alli = []
            union_ri_allj = []
            if i < len(g) and j < len(g[i]):
                g[i][j] = 0
                if len(intersection) == 0:
                    g[i][j] = 0.0001
                    break
                else:
                    for k in tqdm(range(len(intersection))):
                        intersection_ri_alli.append(ri_all[i][ri_index[i].tolist().index(intersection[k])])
                        intersection_ri_allj.append(ri_all[j][ri_index[j].tolist().index(intersection[k])])
                    union_rest = set(union).difference(set(intersection))
                    union_rest = list(union_rest)
                    if len(union_rest) == 0:
                        g[i][j] = 0.0001
                        break
                    else:
                        for k in range(len(union_rest)):
                            if union_rest[k] in ri_index[i]:
                                union_ri_alli.append(ri_all[i][ri_index[i].tolist().index(union_rest[k])])
                            else:
                                union_ri_allj.append(ri_all[j][ri_index[j].tolist().index(union_rest[k])])
                    k_max = max(intersection_ri_allj, intersection_ri_alli)
                    k_min = min(intersection_ri_allj, intersection_ri_alli)
                    union_ri_allj = k_max + union_ri_allj
                    union_num = np.sum(np.array(union_ri_allj), axis=0)
                    inter_num = np.sum(np.array(k_min), axis=0)
                    g[i][j] = inter_num / union_num

    return g