SPADE_score
===============================

SPADE: A Spectral Method for Black-Box Adversarial Robustness Evaluation


Requirements
-----
hnswlib == 0.3.4
or 
faiss == 1.7.0

Usage
-----

**SPADE-Score Evaluation Usage**

Download SPADE.py file. In your Python code, using:
1. `from SPADE import Spade`
2. `TopEig, TopEdgeList, TopNodeList = Spade(data_input, data_output, data_labels, k, num_eigs)`(num_eigs>=2).
3. `data_input` and ` data_output` requie flattening. Please do image flattening if your data is multidimensional array
4. Options default: full_function=True, The_faiss=True, graph=False, one_class=False, julia = False
5. Different graph-based manifold constructions can be chose by `The_faiss`, `The_faiss=Flse` import hnswlib, `The_faiss=True` import faiss
6. Quick eigenvalues calculation using `EigenValues = Spade(data_input, data_output, data_labels, k, num_eigs, full_function=False)`
7. We noticed in some rare case, `scipy.sparse.linalg.eigs` cannot compute eigenvalues correctly(like maximum eigenvalue is negative) if you experience the same case, `julia = True` may helps but it requires Julia installation and download file `eigen.jl`

**SPADE-Guided Robustness Evaluation**
1. You can extract data from your own model then repeat `SPADE-SCORE Evaluation Usage`
2. To reproduce our experiment, `cd evaluation/`
