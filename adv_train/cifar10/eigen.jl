using LinearAlgebra
using LinearMaps
using MAT
using SparseArrays
using Arpack

using PyCall, SparseArrays

function scipyCSC_to_julia(A)
    m, n = A.shape
    colPtr = Int[i+1 for i in PyArray(A."indptr")]
    rowVal = Int[i+1 for i in PyArray(A."indices")]
    nzVal = Vector{Float64}(PyArray(A."data"))
    B = SparseMatrixCSC{Float64,Int}(m, n, colPtr, rowVal, nzVal)
    return PyCall.pyjlwrap_new(B)
end


function main(PyX, PyY, k::Int64)
    m, n = PyX.shape
    colPtr = Int[i+1 for i in PyArray(PyX."indptr")]
    rowVal = Int[i+1 for i in PyArray(PyX."indices")]
    nzVal = Vector{Float64}(PyArray(PyX."data"))
    X = SparseMatrixCSC{Float64,Int}(m, n, colPtr, rowVal, nzVal)
    
    m, n = PyY.shape
    colPtr = Int[i+1 for i in PyArray(PyY."indptr")]
    rowVal = Int[i+1 for i in PyArray(PyY."indices")]
    nzVal = Vector{Float64}(PyArray(PyY."data"))
    Y = SparseMatrixCSC{Float64,Int}(m, n, colPtr, rowVal, nzVal)

    (Λ, V) = eigs(X, Y, nev=k, tol=1e-6, which=:LM)

    return Λ, V
end
