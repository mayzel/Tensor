# coding: utf-8

# Relation Regularized Matrix Factorization

from scipy import *

from numpy import *
from numpy.linalg import *

import algorithm as alg
import Completion as comp

import scipy.sparse as sparse

#3階のテンソルについて学習を行う。A：接続行列　D：落としこむ次元
def Learn3(X,A,D):

    pass



def getLaplacian(A):
    L = sparse.lil_matrix(-A)
    dims = map(lambda v:sum(v), A)
    
    L.setdiag(dims)
    return L.tocsr 

    
