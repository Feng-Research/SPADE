import hnswlib
import numpy as np
from scipy.sparse import csr_matrix
from julia.api import Julia
import networkx as nx
from scipy.sparse.csgraph import laplacian

def spade(data_input, data_output, k=10, num_eigs=2,sparse=False,weighted=False): 

    neighs_in, distance_in = hnsw(data_input, k)
    neighs_out, distance_out = hnsw(data_output, k)
    if weighted:
        adj_in, _, G_in = construct_weighted_adj(neighs_in, distance_in)
        adj_out, _, G_out = construct_weighted_adj(neighs_out, distance_out)
    else:
        adj_in, _, G_in = construct_adj(neighs_in, distance_in)
        adj_out, _, G_out = construct_adj(neighs_out, distance_out)

    assert nx.is_connected(G_in), "input graph is not connected"
    assert nx.is_connected(G_out), "output graph is not connected"

    if sparse:
        adj_in = SPF(adj_in, 4)
        adj_out = SPF(adj_out, 4)
    L_in = laplacian(adj_in, normed=False)
    L_out = laplacian(adj_out, normed=False)
  
    TopEig, TopEdgeList, TopNodeList, node_score, Dxy, Uxy = GetRiemannianDist(G_in, G_out, L_in, L_out, num_eigs)# full function
    return TopEig, TopEdgeList, TopNodeList, node_score, L_in, L_out, Dxy, Uxy


def hnsw(features, k=10, ef=100, M=48):
    num_samples, dim = features.shape

    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_samples, ef_construction=ef, M=M)
    labels_index = np.arange(num_samples)
    p.add_items(features, labels_index)
    p.set_ef(ef)

    neighs, distance = p.knn_query(features, k+1)
  
    return neighs, distance


def construct_adj(neighs, weight):
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
    lap = laplacian(adj, normed=False)
    G = nx.from_scipy_sparse_array(adj)

    return adj, lap, G

def construct_weighted_adj(neighs, distances):
    dim = neighs.shape[0]
    k = neighs.shape[1] - 1
    weights = np.exp(-distances)

    idx0 = np.asarray(list(range(dim)))
    idx1 = neighs[:,0]
    mismatch_idx = ~np.isclose(idx0, idx1, rtol=1e-6)
    neighs[mismatch_idx, 1:] = neighs[mismatch_idx, :k]
    row = (np.repeat(idx0.reshape(-1,1), k, axis=1)).reshape(-1,)
    col = neighs[:,1:].reshape(-1,)
    # calculate weights for each edge
    edge_weights = weights[:,1:].reshape(-1,)
    all_row = np.concatenate((row, col), axis=0)
    all_col = np.concatenate((col, row), axis=0)
    all_data = np.concatenate((edge_weights, edge_weights), axis=0)  # use weights instead of ones
    adj = csr_matrix((all_data, (all_row, all_col)), shape=(dim, dim))
    G = nx.from_scipy_sparse_array(adj)
    # construct a graph from the adjacency matrix
    lap = laplacian(adj, normed=False)

    return adj, lap, G

def SPF(adj, L, ICr=0.11):
    jl = Julia(compiled_modules=False)
    from julia import Main
    Main.include("./my_utils/SPF.jl")
    agj_c = Main.SPF(adj, L, ICr)

    return agj_c

def GetRiemannianDist(Gx, Gy, Lx, Ly, num_eigs): 
    # Gy not updated 
    Lx = Lx.asfptype()
    Ly = Ly.asfptype()
    Dxy, Uxy = julia_eigs(Lx, Ly, num_eigs)
    num_node_tot = Uxy.shape[0]
    TopEig=max(Dxy)
    NodeDegree=Lx.diagonal()
    num_edge_tot=len(Gx.edges) # number of total edges  
    Zpq=np.zeros((num_edge_tot,));# edge embedding distance
    p = np.array(Gx.edges)[:,0];# one end node of each edge
    q = np.array(Gx.edges)[:,1];# another end node of each edge
    for i in np.arange(0,num_eigs):
        Zpq = Zpq + np.power(Uxy[p,i]-Uxy[q,i], 2)*Dxy[i]
    Zpq = Zpq/max(Zpq)

    node_score=np.zeros((num_node_tot,))        
    for i in np.arange(0,num_edge_tot):
        node_score[p[i]]=node_score[p[i]]+Zpq[i]
        node_score[q[i]]=node_score[q[i]]+Zpq[i]
    node_score=node_score/NodeDegree
    node_score=node_score/np.amax(node_score)

    TopNodeList = np.flip(node_score.argsort(axis=0))
    TopEdgeList=np.column_stack((p,q))[np.flip(Zpq.argsort(axis=0)),:]

    return TopEig, TopEdgeList, TopNodeList, node_score, Dxy, Uxy

def julia_eigs(l_in, l_out, num_eigs):
    jl = Julia(compiled_modules=False)
    from julia import Main
    Main.include("./my_utils/eigen.jl")
    print('Generate eigenpairs')
    eigenvalues, eigenvectors = Main.main(l_in, l_out, num_eigs)
    return eigenvalues.real, eigenvectors.real






