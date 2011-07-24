
#coding: utf-8

from numpy import *
from numpy.linalg import *

import itertools as it

import sys
sys.path.insert(0,"oldcodes")

import const

from logger import *


import scipy.sparse as sparse
import scipy.optimize as opt


import algorithm as alg
import SVD

import CPWOPTcritical as critical



#def flattenAs(Div):
#    (dU,dV,dW) = Div
#    dU = dU.flatten()
#    dV = dV.flatten()
#    dW = dW.flatten()
#    return append(dU,append(dV,dW))



"""
CP-WOPTによってテンソルを補完する
"""
def CompletionGradient(X,shape,ObservedList,R,Ls,alpha=0,XoriginalTensor=None):
    #X is Dense Vector with only observed elements

    #Observed must be given as list of coordinate tuples
    #TrueX is 3 dimensional array

    Ls[:] = [L*alpha for L in Ls]

    N = 3
    Xns = [critical.unfold(X,n,shape,ObservedList) for n in xrange(N)]
    print "unfolded"
    As = [SVD.getLeadingSingularVects(Xns[n],R) for n in xrange(N)]
    print "HOSVD finished"


    n,m,l=shape
    def lossfunc(U,V,W):
        print "loss start"
        XO = critical.HadamardProdOfSparseTensor(U,V,W,ObservedList)
        loss = norm(XO - X)
        print "loss end"
        return loss

    #Xinit = flattenAs(As)
    print "start bfgs"


    U,V,W = As
    Lu,Lv,Lw=Ls
    print U.shape, V.shape, W.shape
    maxiter=3
    while True:
        print "optimization of U"
        grad = lambda vU:critical.Gradient(X,ObservedList,(vU.reshape(n,R),V,W),Lu,shape,R,0).flatten()
        loss = lambda vU:lossfunc(vU.reshape(n,R),V,W)
        vu = U.flatten()
        vU = opt.fmin_bfgs(loss,U.flatten(),grad,maxiter=maxiter)
        U = vU.reshape(n,R)

        print "optimization of V"
        grad = lambda vV:critical.Gradient(X,ObservedList,(U,vV.reshape(m,R),W),Lv,shape,R,1).flatten()
        loss = lambda vV:lossfunc(U,vV.reshape(m,R),W)
        vV = opt.fmin_bfgs(loss,V.flatten(),grad,maxiter=maxiter)
        V = vV.reshape(m,R)

        print "optimization of W"
        grad = lambda vW:critical.Gradient(X,ObservedList,(U,V,vW.reshape(l,R)),Lw,shape,R,2).flatten()
        loss = lambda vW:lossfunc(U,V,vW.reshape(l,R))
        vW = opt.fmin_bfgs(loss,W.flatten(),grad,maxiter=maxiter)
        W = vW.reshape(l,R)


def createObservedCoordList(ObsTensor):
    n,m,l = ObsTensor.shape
    ObsList = []
    for i in xrange(n):
        for j in xrange(m):
            for k in xrange(l):
                if ObsTensor[i,j,k] > 0:
                    ObsList.append(i)
                    ObsList.append(j)
                    ObsList.append(k)
    return ObsList

if __name__=="__main__":

    if True:
        import benchmark 
        #data = benchmark.Artificial()
        data = benchmark.Bonnie()
        Ls = data["L"]
        X = data["X"]
        X = X / norm(X)
        n,m,l = X.shape
        W = zeros(n*m*l).reshape(n,m,l)


        per = 1
        obsize = n*m*l * per / 100 / 2
        print obsize
        #Wlst=set([])
        for ind in xrange(obsize):
            i = random.randint(n)
            j = random.randint(m)
            k = random.randint(l)
            #Wlst.add((i,j,k))
            W[i,j,k]=1

        
        rate=0.01
        Ls[:] = [rate*L for L in Ls]
        #Ls=[0,0,0]

        Wlst = list(createObservedCoordList(W))
        Wlst = array(Wlst)
        #Wlst = list(Wlst)

        print "hogefuga"

        Xdense = critical.CompressSparseTensorToVector(X,Wlst)

        print "created test data"
        #raw_input()
        CompletionGradient(Xdense,(n,m,l),Wlst,8,Ls)



        
