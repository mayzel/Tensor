#from __future__ import division
cimport cython
cimport numpy as np
import numpy as np

np.import_array()

import scipy.sparse as sparse


@cython.boundscheck(False)
def HadamardProdOfSparseTensor(np.ndarray[np.float_t,ndim=2] U not None,np.ndarray[np.float_t,ndim=2] V not None,np.ndarray[np.float_t,ndim=2] W not None,np.ndarray[np.int_t,ndim=1] Obs not None):
    #print "HP start"
    cdef int size
    cdef int n,m,l
    #cdef Coords *Obs
    cdef int i,j,k
    #cdef np.ndarray[np.float_t,ndim=2] U,V,W
    cdef np.ndarray[np.float_t,ndim=1] data

    cdef int R,r
    cdef float sumval

    size = Obs.size / 3

    n=U.shape[0]
    m=V.shape[0]
    l=W.shape[0]
    R=U.shape[1]

    #Obs must be given as list of coordinate tuples
    data = np.zeros(size,dtype=np.float)

    for n in range(size):
        i = Obs[n*3]
        j = Obs[n*3+1]
        k = Obs[n*3+2]
        #data[n] = sum(U[i,:]*V[j,:]*W[k,:])
        sumval = 0.0
        for r in range(R):
            sumval += U[i,r]*V[j,r]*W[k,r]
        data[n] = sumval
    
    #print "HP end"
    return data


@cython.boundscheck(False)
def unfold(np.ndarray[np.float_t,ndim=1] X not None, int mode, tuple shape not None,np.ndarray[np.int_t,ndim=1] ObsList not None):
    #X is dense vector with only observed elements
    #print "unfold start"

    cdef int n,m,l,size,N,M,L,i,j,k
    cdef np.ndarray[np.int_t,ndim=1] col,row
    #cdef np.ndarray[np.float_t,ndim=1] val 
    #cdef float val
    #cdef sparse.lil_matrix
    
    N,M,L = shape[0],shape[1],shape[2]
    size = ObsList.size / 3


    if mode==0:
        row = np.zeros(size,dtype=int)
        col = np.zeros(size,dtype=int)
        for n in range(size):
            i,j,k = ObsList[n*3],ObsList[n*3+1],ObsList[n*3+2]
            row[n] = i
            col[n] = j+k*M
        #print "unfold end"
        return sparse.csr_matrix((X,(row,col)),shape=(N,M*L))
    elif mode==1:
        row = np.zeros(size,dtype=int)
        col = np.zeros(size,dtype=int)
        for n in range(size):
            i,j,k = ObsList[n*3],ObsList[n*3+1],ObsList[n*3+2]
            row[n] = j
            col[n] = i+k*N
        #print "unfold end"
        return sparse.csr_matrix((X,(row,col)),shape=(M,N*L))
    else:
        row = np.zeros(size,dtype=int)
        col = np.zeros(size,dtype=int)
        for n in range(size):
            i,j,k = ObsList[n*3],ObsList[n*3+1],ObsList[n*3+2]
            row[n] = k
            col[n] = i+j*N
        #print "unfold end"
        return sparse.csr_matrix((X,(row,col)),shape=(L,N*M))


#Khatri-Rao product
cdef np.ndarray[np.float_t,ndim=2] KRproduct(np.ndarray[np.float_t,ndim=2] A ,np.ndarray[np.float_t,ndim=2] B):
    """
    Khatri-Rao積
    """
    cdef int x,y
    x,y = A.shape[0],A.shape[1]

    return np.array([np.kron(A[:,i],B[:,i]) for i in xrange(y)]).T


cdef tuple unpackAs(np.ndarray[np.float_t,ndim=1] Asv,int su,int sv,int sw):
    cdef np.ndarray[np.float_t,ndim=1] dU,dV,dW
    dU = Asv[0:su]
    dV = Asv[su:su+sv]
    dW = Asv[su+sv:]
    return (dU,dV,dW)

def Gradient(np.ndarray[np.float_t,ndim=1] X not None,np.ndarray[np.int_t,ndim=1] Obs not None,tuple As,np.ndarray[np.float_t,ndim=2] L not None,tuple shape,int R,int mode):
    #Ls must be given as CSR_matrix
    #Obs must be given as list of coordinate tuples
    #X is given as dense vector with only observed elements
    #print "Gradient starat"


    cdef int n,m,l,ind,i,j,k
    cdef float val
    cdef np.ndarray[np.float_t,ndim=2] U,V,W,dA
    #cdef np.ndarray[np.float_t,ndim=1] vU,vV,vW
    
    (n,m,l) = shape

    U=As[0]
    V=As[1]
    W=As[2]
    
    XO = HadamardProdOfSparseTensor(U,V,W,Obs)

    Left = XO - X
    Left_f = unfold(Left,mode,shape,Obs)

    if mode==0:
        dA = 2*Left_f.dot(KRproduct(W,V)) + L.dot(U)
    elif mode==1:
        dA = 2*Left_f.dot(KRproduct(W,U)) + L.dot(V)
    else:
        dA = 2*Left_f.dot(KRproduct(V,U)) + L.dot(W)
    #print "Gradient end"
    return dA.squeeze()

    #return squeezeAs((dU,dV,dW))
@cython.boundscheck(False)
def CompressSparseTensorToVector(np.ndarray[np.float_t,ndim=3] X not None, np.ndarray[np.int_t,ndim=1] ObservedList not None):
    cdef int size
    cdef np.ndarray[np.float_t,ndim=1] Xdense
    size = ObservedList.size / 3

    Xdense = np.zeros(size)
    size = len(ObservedList) / 3
    for n in range(size):
        i = ObservedList[3*n]
        j = ObservedList[3*n+1]
        k = ObservedList[3*n+2]
        Xdense[n] = X[i,j,k]
    return Xdense


