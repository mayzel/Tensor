#coding: utf-8

from numpy import *
from numpy.linalg import *

import itertools as it

import const

from logger import *
from algorithm import *


"""
EM-ALSによってテンソルを補完する
"""
def CompletionStep(TrueX,Observed,X,updater):
    """
    @param TrueX 真のテンソル。答え。
    @param Observed 観測要素を表すテンソル。1:観測 0:未観測
    @param X これ意味ないかも…
    @param updater XとObservedを受け取って新しい補完Xを返す函数

    @blief EM-ALSによりテンソルを補完する。
    """
    #updater :: (X,As) -> (G,As)
    pred = vectorize(lambda x: 0 if x==0 else 1)
    Observed = pred(Observed)

    elems = prod(X.shape)
    n = sum(Observed)

    mean = sum(X * Observed) * 1.0 / n
    Unobserved = 1 - Observed
    rand = random.rand(*Observed.shape)
    overwrite = vectorize(lambda x,y,w:x if w!=0 else y)

    #X = overwrite(TrueX,Unobserved*(rand - 0.5)*mean*2.0,Observed)
    X = overwrite(TrueX,Unobserved*mean,Observed)
    X = overwrite(TrueX,Unobserved*0,Observed)
    #X = overwrite(TrueX,Init,Observed)
    #print TrueX
    #print "initdiff:", norm(TrueX-X)
    #log.WriteLine(norm(TrueX-X),False)
    errorold = float("inf")

    As = None
    G = None
    maxiter = 600
    threshold = const.ConvergenceThreshold_NewCompletion
    vint = vectorize(int)
    print "rapid algorithm"
    for steps in it.count():

        import matplotlib.pyplot as plt
        im = plt.imshow(X.reshape(89*4,100*3))
        plt.savefig("Est_Flow%03d.png" % steps,vmax=0.01566,vmin=-0.00010001)

        #benchmark.SaveImage(X.reshape(89*4,100*3),"Est_Flow%03d" % steps)
        (G,As)= updater(X,G,As) #update parameters
        Xnew = expand(G,As)

        #print norm(X*Observed)
        #Xnew = Xnew * norm(X*Observed) / norm(Xnew*Observed) #ノルムの調整

        #Enron用
        if False:
            Xnew = Xnew * (sign(Xnew) + 1) / 2 #必ず正に
            errorObserved = norm((TrueX-vint(Xnew))*Observed)
            error = norm(TrueX-vint(X)) # * sqrt(elems * 1.0 / n)
        else:
            #Xnew = TrueX*Observed
			#Xnew = vint(Xnew)
            errorObserved = norm((TrueX-Xnew)*Observed)
            error = norm(TrueX-X) # * sqrt(elems * 1.0 / n)

        diff = norm(Xnew/norm(Xnew)-X/norm(X))
        X = overwrite(TrueX,Xnew,Observed)
        #import benchmark


        print "iter:",steps," err:",error ," oberr:",errorObserved, " diff:", errorObserved-errorold, "norm;", norm(Xnew)

        faultThreshold = 1e4
        if error > faultThreshold:
            X = faultThreshold
            return X

        if abs(errorObserved- errorold) < threshold or steps + 1 >= maxiter:
            print "estimation finished in ",(steps+1),"steps."
            break

        errorold = errorObserved 
        
    return X





Ps=[]
Ds=[]
def resetRRMFCP():
    Ps=[]
    Ds=[]

def RRMFCP_obsolete(TrueX,Observed,X,R,beta,Ls=None,alpha=0,Ps=None,Ds=None):
    """
    Obsolete
    """
    #log = Logger("CP")
    pred = vectorize(lambda x: 0 if x==0 else 1)
    Observed = pred(Observed)

    elems = prod(X.shape)
    n = sum(Observed)

    mean = sum(X * Observed) * 1.0 / n
    Unobserved = 1 - Observed

    rand = random.rand(*Observed.shape)
    
    overwrite = vectorize(lambda x,y,w:x if w!=0 else y)

    #X = overwrite(TrueX,Unobserved*(rand - 0.5)*mean*2.0,Observed)
    X = overwrite(TrueX,Unobserved*mean,Observed)
    #X = overwrite(TrueX,Unobserved*0,Observed)
    #X = overwrite(TrueX,Init,Observed)
    #print TrueX
    #print "initdiff:", norm(TrueX-X)
    #log.WriteLine(norm(TrueX-X),False)
    errorold = float("inf")
    e = const.ConvergenceThreshold_Completion * 1
    maxiter = 500
    N = X.ndim

    N = X.ndim

    if Ls == None:
        Ls = [None for i in xrange(N)]

    if Ps==None or Ds==None:
        Ds=[]
        Ps=[]
        for n in xrange(len(Ls)):
            if Ls[n]==None:
                Ps.append(None)
                Ds.append(None)
            else:
                (v,P) = eigh(Ls[n])
                Ps.append(P)
                Ds.append(v)

    Xns=[None for i in xrange(N)]
    for n in xrange(N):
        Xns[n] = unfold(X,n)

    As = [None for i in xrange(N)]
    As=[]
    for n in xrange(N):
        #initialize A^(n)
        #As.append(random.randn(X.shape[n],R))
        #As.append(random.rand(X.shape[n],R))
        As.append(getLeadingSingularVects(Xns[n],R))

    threshold = const.ConvergenceThreshold_NewCompletion

    Itensor = createUnitTensor(N,R)
    xnorm = 0
    xnormold = -10000
    UnitTensor = createUnitTensor(N,R)
    for steps in it.count():
        for n in xrange(N):
            Xn = unfold(X,n)
            #As[n] = getLeadingSingularVects(Xn,R)
            Xns[n] = Xn

        for iterate in xrange(1):
            for n in xrange(N):

                S = reduce(lambda x, y: KRproduct(y,x), [As[i] for i in range(N) if i != n]) #順序が逆
                #print "DIFF:",norm(StS - dot(S.T,S))
                #StS = dot(S.T,S)
                #Xn = unfold(X,n)
                #Xn = unfold(X,(n+1)%N)
                Xn = Xns[n]
                L = Ls[n]
                if L == None:
                    StS = reduce(lambda x,y: dot(x.T,x)*dot(y.T,y), [As[i] for i in range(N) if i != n])
                    I = eye(*StS.shape)
                    As[n] = solve(StS + beta*I, (dot(Xn,S)).T).T
                else:
                    #print beta
                    An = As[n]
                    if True:
                        for d in xrange(R):
                            g = beta + dot(S[:,d].T,S[:,d])
                            de = 1.0 / (g + alpha*Ds[n])
                            Finv = dot(Ps[n]*de,Ps[n].T)
                            #Finv = dot(dot(Ps[n],diag(de)),Ps[n].T)
                            Ed = dot((Xn - dot(An,S.T) + outer(An[:,d],S[:,d])), S[:,d])
                            An[:,d] = dot(Finv,Ed)
                    else:
                        StS = reduce(lambda x,y: dot(x.T,x)*dot(y.T,y), [As[i] for i in range(N) if i != n])
                        I = eye(*L.shape)
                        M = beta * I + alpha * L
                        As[n] = lyap(M,StS,-dot(Xn,S))

        Xnew = expand(UnitTensor,As)

        Xnew = Xnew * norm(X*Observed) / norm(Xnew*Observed)
        errorObserved = norm((TrueX-Xnew)*Observed)

        diff = norm(Xnew/norm(Xnew)-X/norm(X))
        X = overwrite(TrueX,Xnew,Observed)


        error = norm(TrueX-X) # * sqrt(elems * 1.0 / n)
        print "iter:",steps," err:",error ," oberr:",errorObserved, " diff:", errorObserved-errorold, "norm;", norm(Xnew)

        #log.WriteLine(diff,False)

        #if abs(diff) < e or steps + 1 >= maxiter:
        if abs(errorObserved- errorold) < threshold or steps + 1 >= maxiter:
            print "estimation finished in ",(steps+1),"steps."
            break

        errorold = errorObserved 


    return X

