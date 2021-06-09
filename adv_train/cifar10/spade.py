import hnswlib
import numpy as np
from scipy.io import loadmat
import scipy.sparse
from scipy.sparse import coo_matrix, diags, identity, csr_matrix
from scipy.sparse.linalg import eigsh
import networkx as nx
from numpy import linalg as LA
import sklearn
import time
import sys
from scipy.io import mmwrite

def eigs(l_in, l_out, k):
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from julia import Main
    Main.include("./eigen.jl")
    print('Generate eigenpairs')
    eigenvalues, eigenvectors = Main.main(l_in, l_out, k)

    return eigenvalues.real, eigenvectors.real

def node_score(l_in, l_out):
    print("computing eigens")
    eigenvalues, eigenvectors = eigs(l_in, l_out, 2)
    eigenvalues = np.sqrt(eigenvalues).reshape(1,-1)
    embeddings = eigenvalues * eigenvectors

    G_in = nx.read_gpickle("knn_graph.gpickle")

    print("computing weighted degree")
    for e in G_in.edges():
        src = e[0]
        dst = e[1]
        diff = embeddings[src] - embeddings[dst]
        G_in[src][dst]['weight'] = np.inner(diff, diff)

    degree = {pair[0]: pair[1] for pair in G_in.degree()}
    weighted_degree = {pair[0]: pair[1] for pair in G_in.degree(weight='weight')}

    print("computing node spade score")
    num_nodes = len(degree)
    node_spade_score = np.zeros(num_nodes)
    for i in range(num_nodes):
        if degree[i] == 0:
            print("isolated node, pls check the knn graph!!!!!!")
            sys.exit()
        node_spade_score[i] = weighted_degree[i]/degree[i]

    node_spade_score = (1.0/np.amax(node_spade_score)) * node_spade_score

    return node_spade_score

def construct_adj(neighs):
    dim = neighs.shape[0]
    k = neighs.shape[1] - 1

    idx0 = np.asarray(list(range(dim)))
    idx1 = neighs[:,0]
    mismatch_idx = ~np.isclose(idx0, idx1, rtol=1e-6)
    neighs[mismatch_idx, 1:] = neighs[mismatch_idx, :k]
    row = (np.repeat(idx0.reshape(-1,1), k, axis=1)).reshape(-1,)
    col = neighs[:,1:].reshape(-1,)
    all_row = np.concatenate((row, col), axis=0)
    all_col = np.concatenate((col, row), axis=0)
    data = np.ones(all_row.shape[0])
    adj = csr_matrix((data, (all_row, all_col)), shape=(dim, dim))
    adj.data[:] = 1

    return adj

def adj2laplacian(A):
    D = diags(np.squeeze(np.asarray(A.sum(axis=1))), 0)
    L = D - A + identity(A.shape[0]).multiply(1e-6)

    return L

def hnsw(features, k=10, ef=100, M=48, save_index_file=None, gen_graph=None):
    print("constructing hnsw")
    num_samples, dim = features.shape
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_samples, ef_construction=ef, M=M)
    labels_index = np.arange(num_samples)
    p.add_items(features, labels_index)

    p.set_ef(ef)

    if save_index_file:
        p.save_index(save_index_file)

    print("constructing knn")
    neighs, _ = p.knn_query(features, k+1)
    adj = construct_adj(neighs)

    if gen_graph:
        print("construct graph!!!!!!")
        G = nx.from_scipy_sparse_matrix(adj)
        nx.write_gpickle(G, "knn_graph.gpickle")

    laplacian = adj2laplacian(adj)
    #scipy.sparse.save_npz('laplacian.npz', laplacian)

    return laplacian

def spade(feat_in, feat_out, k=10):
    #l_in = hnsw(feat_in, k=k, gen_graph=True)
    #scipy.sparse.save_npz("l_in.npz", l_in)
    #sys.exit()
    l_in = scipy.sparse.load_npz("l_in.npz")
    l_out = hnsw(feat_out, k=k)
    node_spade_score = node_score(l_in, l_out)

    return node_spade_score
