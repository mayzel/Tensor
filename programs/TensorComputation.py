# coding: utf-8
from numpy import *
from numpy.linalg import *


def get_unfolding_matrix_size(A, n):
    """
    展開後の行列のサイズを計算する
    """
    row_count = A.shape[n]
    col_count = 1
    for i in xrange(A.ndim):
	if i != n: col_count *= A.shape[i]
    return (row_count, col_count)



def swapIndex(n):
    """
    インデックス並べ替え。軸の番号を表示上の直感と合わせるため。
    """
    if n==0:
	n=1
    elif n==1:
	n=0
    return n

def getAxis(dim,n):
    return n
    kn = swapIndex(n)
    return dim - n - 1

def createUnitTensor(dim,D):
    """
    dim次元大きさRの超対角な単位テンソルを作る
    """
    size = tuple(repeat(D,dim))
    elements = pow(D,dim)

    I = repeat(0,elements).reshape(*size)
    for loc in [tuple(repeat(i,dim)) for i in xrange(D)]:
        I[loc]=1
    return I


def expand(G,As):
    """
    コアテンソルGと因子Asの積をとりテンソルを生成。
    """
    N = len(As)
    X = G
    for n in xrange(N):
        X = Nproduct(X,As[n],n)
    return X


#軸の順序がそのままなので注意すること！
def randomTensorOfNorm(sizelist,ranks,std):
    """
    obsolete
    """
    X = 0
    A = zeros(sizelist)
    N = len(sizelist)

    if isinstance(ranks,int):
	As = [random.rand(sizelist[i],ranks) for i in xrange(N)]
	As = [A - mean(A) for A in As]
	I = createUnitTensor(N,ranks)
	A = expand(I,As)
    else:
	As = [random.rand(sizelist[i],ranks[i]) for i in xrange(N)]
	As = [A - mean(A) for A in As]
	G = random.rand(*ranks)
	A = expand(G,As) 

    A = A / norm(A)
    noise =  random.normal(0,1,sizelist)
    noise = noise / norm(noise) * std

    return (A + noise) / norm(A + noise)


def unfold(A, n):
    """
    テンソルのモードn展開を計算する
    """
    def get_transposing_permutation(n,dim):
	r = range(dim)
	r.pop(n)
	r = r[::-1]
	r.insert(0,n)
	return r
    n = getAxis(A.ndim,n)

    (row_count, col_count) = get_unfolding_matrix_size(A, n)
    perm = get_transposing_permutation(n, A.ndim)
#    print perm
#    print (row_count, col_count)
    return A.transpose(perm).reshape(row_count,col_count)


def Nproduct(X,U,n):
    """
    テンソルと行列のnモード積を計算する
    """
    #calculate axes to sum over from n
    dimx = X.ndim 
    dimu = U.ndim 

    axex = getAxis(dimx,n)

    if dimu==2:
	perm = range(dimx)
	perm.insert(axex,perm.pop())
	#perm.append(perm.pop(axex))
	return tensordot(X,U,(axex,1)).transpose(perm)
    elif dimu==1:
	return tensordot(X,U,(axex,0))

#Khatri-Rao product
def KRproduct(A,B):
    """
    Khatri-Rao積
    """
    (x,y) = A.shape
    return array([kron(A[:,i],B[:,i]) for i in xrange(y)]).transpose()


#Hadamard product （不要？
def Hproduct(A,B):
    """
    Calcurate Hadamard Product.Not in Use.
    """
    return A*B

def getG(X,As):
    """
    テンソルと因子からTucker分解の場合に最適なコアテンソルを生成。
    """
    G = X
    for n in xrange(X.ndim):
        G = Nproduct(G,As[n].transpose(),n)
    return G

#Khatri-Rao product
def KRproduct(A,B):
    """
    Khatri-Rao積
    """
    (x,y) = A.shape
    return array([kron(A[:,i],B[:,i]) for i in xrange(y)]).transpose()


#Hadamard product （不要？
def Hproduct(A,B):
    """
    Calcurate Hadamard Product.Not in Use.
    """
    return A*B

def getLeadingSingularVects(X,R):
    """
    大きい方からR個の左特異ベクトルをとる
    """
    return getLeadingEigenVects(dot(X, X.T), R)

import numpy.linalg
import scipy.linalg
def getLeadingEigenVects(X,R):
    """
    大きい方からR個の固有ベクトルをとる
    """
    X = (X + X.T)/2
    [val,vec] = scipy.linalg.eig(X)
    vec = real(vec)
    #[val,vec] = numpy.linalg.eigh(X)
    index = argsort(val, kind='mergesort')
    index = index[::-1]

    return vec[:,index[:R]]
