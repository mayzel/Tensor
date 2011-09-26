#coding: utf-8

from numpy import *
from numpy.linalg import *

import itertools as it

import const

from logger import *

from TensorComputation import *

class Completion:
    def __init__(self,X,L,decomposition=None):
        self.X = X
        self.L = L
        self.decomposition = decomposition

    def estimator(self,param, trainingData):
        pass
    def lossFunction(self):
        pass

    def DecomposeLaplacians(self,Ls):
        """
        グラフラプラシアンを固有値分解する。
        """
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
        return (Ps,Ds)

#completion
    """
    EM-ALSによってテンソルを補完する
    """
    def CompletionStep(self,TrueX,Observed,X,updater):
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

            #import matplotlib.pyplot as plt
            #im = plt.imshow(X.reshape(89*4,100*3))
            #plt.savefig("Est_Flow%03d.png" % steps,vmax=0.01566,vmin=-0.00010001)

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






