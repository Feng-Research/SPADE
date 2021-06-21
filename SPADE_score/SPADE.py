#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        
# function [TopEig, TopEdgeList, TopNodeList]=RiemannianDist(Gx,Gy,num_eigs,gnd) 
def GetRiemannianDist(Gx, Gy, num_eigs, gnd, one_class, julia, node):
    
    import scipy.sparse.linalg as sla
    import numpy as np
    import scipy as sp
    import networkx as nx
    import math
    
    Lx= nx.laplacian_matrix(Gx, nodelist=None, weight=None)
    Ly= nx.laplacian_matrix(Gy, nodelist=None, weight=None)
    
    Lx = Lx.asfptype()
    Ly = Ly.asfptype()
    
    if julia:
        from julia.api import Julia
        jl = Julia(compiled_modules=False)
        from julia import Main
        Main.include("./eigen.jl")
        print('Generate eigenpairs')
        [Dxy, Uxy] = Main.main(Lx, Ly, num_eigs)
    else:
        [Dxy, Uxy] = sla.eigs(Lx, num_eigs, Ly)

    num_node_tot=Uxy.shape[0]
    num_top_node=10;
    dist=np.zeros((1,1));
    num_cluster=5;

    TopEig=max(Dxy)
    NodeDegree=Lx.diagonal()

    num_edge_tot=len(Gx.edges) # number of total edges  

    Zpq=np.zeros((num_edge_tot,));# edge embedding distance

    p = np.array(Gx.edges)[:,0];# one end node of each edge
    q = np.array(Gx.edges)[:,1];# another end node of each edge

    density=math.ceil(num_edge_tot/num_node_tot)

    for i in np.arange(0,num_eigs):
        Zpq = Zpq + np.power(Uxy[p,i]-Uxy[q,i], 2)*Dxy[i]

    Zpq = Zpq/max(Zpq)
    node_score=np.zeros((num_node_tot,))
    
    if not one_class:
        label_score=np.zeros(np.unique(gnd).shape,)
    else:
        label_score=np.zeros(10,)
        
    for i in np.arange(0,num_edge_tot):
        node_score[p[i]]=node_score[p[i]]+Zpq[i]
        node_score[q[i]]=node_score[q[i]]+Zpq[i]
    node_score=node_score/NodeDegree
    node_score=node_score/np.amax(node_score)

    for i in np.arange(0,num_node_tot):
        idx = int(gnd[i]);
        label_score[idx]=label_score[idx]+node_score[i];   

    label_score=label_score/max(label_score);

    II = np.flip(node_score.argsort(axis=0))
    YY = node_score    
    YY.sort(axis=0)    
    YY = np.flip(YY)
    TopNodeList=II
    node_top=II[0:num_top_node]         
    gndSort=gnd[II];
    I = np.flip(Zpq.argsort(axis=0))
    Y = node_score
    Y.sort(axis=0)
    Y = np.flip(Y)
    TopEdgeList=np.column_stack((p,q))[I,:]

    if node:
        return TopEig, TopEdgeList, TopNodeList,node_score
    else:
        return TopEig, TopEdgeList, TopNodeList

def hnsw(data, k):
    import hnswlib
    import numpy as np
    import pickle

    dim = data.shape[1]
    num_elements = data.shape[0]
    data_labels = np.arange(num_elements)

    # Declaring index
    p = hnswlib.Index(space = 'l2', dim = dim) # possible options are l2, cosine or ip

    # Initializing index - the maximum number of elements should be known beforehand
    p.init_index(max_elements = num_elements, ef_construction = 200, M = 16)

    # Element insertion (can be called several times):
    p.add_items(data, data_labels)

    # Controlling the recall by setting ef:
    p.set_ef(50) # ef should always be > k

    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    labels, distances = p.knn_query(data, k)
    
    return labels, distances

def Spade(data_input, data_output, data_labels, k, num_eigs, massive_data=True, full_function=True, The_faiss=False, graph=False, one_class=False, julia = False, node = False): 
    import numpy as np
    import pandas as pd
    import networkx as nx
    import scipy.linalg as la
    from scipy.linalg import eigh
    import scipy.sparse.linalg as sla
    import numpy as np
    from scipy.sparse import lil_matrix
    import scipy.sparse as sp

    t = Timer()
    t.start()

    if data_input.shape[0] == data_output.shape[0] is False:
        return('input and output must have the same sample numbers')
    
    # use faiss
    if The_faiss:
        import faiss
        data_input = data_input.astype('float32')
        num_elements = data_input.shape[0]
        dim = data_input.shape[1]
        index = faiss.IndexFlatL2(dim)   # build the index
        index.add(data_input)            # add vectors to the index
        print("totual elements:{}".format(index.ntotal))
        _, I_in = index.search(data_input, k)        # actual search


        data_output = data_output.astype('float32')
        num_elements = data_output.shape[0]
        dim = data_output.shape[1]
        index = faiss.IndexFlatL2(dim)   # build the index
        index.add(data_output)           # add vectors to the index
        print("totual elements:{}".format(index.ntotal))
        _, I_out = index.search(data_output, k)     # actual search

    #  use hnsw 
    else:
        I_in,_ = hnsw(data_input, k)
        I_out,_ = hnsw(data_output, k)      
        
    if massive_data:
        A_in = lil_matrix((data_input.shape[0], data_input.shape[0]))
        A_out = lil_matrix((data_input.shape[0], data_input.shape[0]))
        for i in range(0, data_input.shape[0]):
            for j_in in I_in[i][1:]:
                A_in[i,j_in] = 1

            for j_out in I_out[i][1:]:
                A_out[i,j_out] = 1
        Gx = nx.from_scipy_sparse_matrix(A_in)
        Gy = nx.from_scipy_sparse_matrix(A_out)

    else:
        A_in = np.zeros((data_input.shape[0], data_input.shape[0]))
        A_out = np.zeros((data_output.shape[0], data_output.shape[0]))
        for i in range(0, data_input.shape[0]):
            for j_in in I_in[i][1:]:
                A_in[i][j_in] = 1
            for j_out in I_out[i][1:]:
                A_out[i][j_out] = 1

        A_in = pd.DataFrame(A_in)
        A_out = pd.DataFrame(A_out)
        Gx = nx.from_pandas_adjacency(A_in)
        Gy = nx.from_pandas_adjacency(A_out)
        
    print('Is input graph connected?:',nx.is_connected(Gx))
    print('Is output graph connected?:',nx.is_connected(Gy))

    if full_function:
        if node:
            TopEig, TopEdgeList, TopNodeList, node_score = GetRiemannianDist(Gx, Gy, num_eigs, data_labels,one_class, julia, node)# full function
            t.stop()
            if graph:
                return TopEig, TopEdgeList, TopNodeList, node_score, Gx, Gy
            else:
                return TopEig, TopEdgeList, TopNodeList, node_score
        else:
            TopEig, TopEdgeList, TopNodeList = GetRiemannianDist(Gx, Gy, num_eigs, data_labels,one_class, julia, node)# full function
            t.stop()
            if graph:
                return TopEig, TopEdgeList, TopNodeList, Gx, Gy
            else:
                return TopEig, TopEdgeList, TopNodeList

    else:
        input_ = nx.laplacian_matrix(Gx, nodelist=None, weight=None)#.todense() 
        output_ = nx.laplacian_matrix(Gy, nodelist=None, weight=None)#.todense()
        input_ = input_.asfptype()
        output_ = output_.asfptype()
        if julia:
            from julia.api import Julia
            jl = Julia(compiled_modules=False)
            from julia import Main
            Main.include("./eigen.jl")
            print('Generate eigenpairs')
            [V,D] = Main.main(input_, output_, num_eigs)
        
        else:
            [V,D] = sla.eigs(input_, num_eigs, output_)#for eigs only 
        t.stop() 
        
        if graph:
            return V, Gx, Gy
        else:
            return V






