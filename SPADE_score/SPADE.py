import hnswlib
import numpy as np
from scipy.sparse import csr_matrix
from julia.api import Julia
from scipy.sparse.csgraph import laplacian
from scipy.sparse import find, triu


def hnsw(features, k=10, ef=100, M=48):
    num_samples, dim = features.shape

    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_samples, ef_construction=ef, M=M)
    labels_index = np.arange(num_samples)
    p.add_items(features, labels_index)
    p.set_ef(ef)

    neighs, weight = p.knn_query(features, k+1)
  
    return neighs, weight

def spade(adj_in, data_output, k=10, num_eigs=2): 

    #G_in = nx.from_scipy_sparse_matrix(adj_in)
    neighs, distance = hnsw(data_output, k)
    adj_out, _, G_out = construct_adj(neighs, distance)

    #assert nx.is_connected(G_in), "input graph is not connected"
    #assert nx.is_connected(G_out), "output graph is not connected"

    #adj_in = SPF(adj_in, 4)
    L_in = laplacian(adj_in, normed=True)#.tocsr()#adj2laplacian(adj_in)

    #adj_out = SPF(adj_out, 4)
    L_out = laplacian(adj_out, normed=True)#.tocsr()#adj2laplacian(adj_out)
  
    TopEig, TopEdgeList, TopNodeList, node_score = GetRiemannianDist_nonetworkx(L_in, L_out, num_eigs)# full function
    return TopEig, TopEdgeList, TopNodeList, node_score



def GetRiemannianDist_nonetworkx(Lx, Ly, num_eigs):
    # Gy not updated 
    Lx = Lx.asfptype()
    Ly = Ly.asfptype()
    Dxy, Uxy = julia_eigs(Lx, Ly, num_eigs)
    num_node_tot = Uxy.shape[0]
    TopEig=max(Dxy)
    NodeDegree=Lx.diagonal()

    laplacian_upper = triu(Lx, k=1)# k=1 excludes the diagonal
    rows, cols, _ = find(laplacian_upper)# Find the indices of non-zero elements
    num_edge_tot = len(rows)# Number of total edges
    Zpq = np.zeros((num_edge_tot,))# Initialize edge embedding distance array
    p = rows# one end node of each edge
    q = cols# another end node of each edge

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

    return TopEig, TopEdgeList, TopNodeList, node_score


def julia_eigs(l_in, l_out, num_eigs):
    jl = Julia(compiled_modules=False)
    from julia import Main
    Main.include("./eigen.jl")
    print('Generate eigenpairs')
    eigenvalues, eigenvectors = Main.main(l_in, l_out, num_eigs)

    return eigenvalues.real, eigenvectors.real



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
    #G = nx.from_scipy_sparse_matrix(adj)
    G = None
    #lap = nx.laplacian_matrix(G, weight=None)
    lap = None

    return adj, lap, G






