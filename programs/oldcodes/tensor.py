# coding: utf-8
"""@package debug
デバッグ用コード一式

:Author:成田敦博

:abstract:
デバッグとパラメータの調整に用いた
コードのゴミがここに集積されています。

"""
from scipy import *

from numpy import *
from numpy.linalg import *
import numpy

import algorithm as alg
import Completion as comp

from logger import *

import matlab as mat
#import cpold 

def createMask(sizelist,zerorate):
    """
    ランダムなマスクを作成。
    1が観測、0が未観測
    """
    elems = prod(sizelist)
    W = repeat(1,elems)

    
    for i in xrange(int(elems*zerorate)):
        while True:
            index = random.randint(elems)
            if(W[index] == 1):
                W[index] = 0
                break
    return W.reshape(*sizelist) 



Ytest = array(
[[[ 0.28171704 , 0.28503647 , 0.10801784],
  [ 0.41287935  , 0.41815519 , 0.15572325],
  [ 0.40670058 , 0.41193002 , 0.15318799]],

 [[ 0.55231554 , 0.52395359 , 0.43118031],
  [ 0.63637749 , 0.5937978 ,  0.55910515],
  [ 0.6131417 ,  0.57111898 , 0.54496872]],

 [[ 0.60396803 , 0.59666539 , 0.32230538],
  [ 0.81359122 , 0.80301761 , 0.43880366],
  [ 0.79574554 , 0.78534038 , 0.42957812]]]
)

Wtest = array(
[[[ 1 , 0 , 1],
  [ 0 , 1 , 1],
  [ 0 , 1 , 0]],

 [[ 0 , 1 , 0],
  [ 1 , 0 , 1],
  [ 1 , 1 , 1]],

 [[ 1 , 1 , 1],
  [ 0 , 0 , 1],
  [ 1 , 1 , 0]]]
)

def Evaluation_Tucker():
    """
    Tucker分解のクロスバリデーションによる評価
    """
    logger = Logger("TuckerEvaluation")
    s = 20
    r = 3
    re = 2 
    size = [s,s,s]
    rank = [r,r,r]
    rank_estimate = [re,re,re]

    logger.WriteLine("TensorSize="+str(size))
    logger.WriteLine("TensorRank="+str(rank))
    logger.WriteLine("EstimationRank"+str(rank_estimate))

    Y = alg.randomTensorOfNorm(size,rank,0.05) * 100 
    while True:
        testrates = [0.02 * (i+1) for i in xrange(49)]
        for zerorate in testrates:

            W = createMask(size,zerorate)

            def approximate(Xin):
                (G,As) = alg.HOOI(Xin,rank_estimate)
                Xs = alg.expand(G,As)
                return Xs

            X = comp.Completion(Y,W,approximate)
            diff = norm(Y-X)
            logger.WriteLine(str(zerorate)+ " " + str(diff))
            print "rate:", zerorate, "error:", norm(Y-X)

def Evaluation_CP():
    """
    CP分解のクロスバリデーションによる評価
    """
    logger = Logger("CPEvaluation")
    s = 20
    r = 3
    re = 2 
    size = [s,s,s]
    rank = r 
    rank_estimate = re 

    alpha = 0.001

    logger.WriteLine("TensorSize="+str(size))
    logger.WriteLine("TensorRank="+str(rank))
    logger.WriteLine("EstimationRank"+str(rank_estimate))

    Y = alg.randomTensorOfNorm(size,rank,0.05) * 100 
    I = alg.createUnitTensor(len(size),rank)
    while True:
        testrates = [0.1 * (i+1) for i in xrange(9)]
        for zerorate in testrates:

            W = createMask(size,zerorate)

            def approximate(Xin):
                As = alg.RRMFCP(Xin,rank_estimate,alpha)
                Xs = alg.expand(I,As)
                return Xs

            X = comp.Completion(Y,W,approximate)
            diff = norm(Y-X)
            logger.WriteLine(str(zerorate)+ " " + str(diff))
            print "rate:", zerorate, "error:", norm(Y-X)

def testCompletion():
    """
    テンソル補完のテスト用コード
    """
    re = 4
    rank_estimate = [re,re,re]
    zerorate = 0.99
    import benchmark
    #data = benchmark.Artificial()
    data = benchmark.Flow_Injection()
    Ls = data["L"]
    X = data["X"]#

    #X = X - mean(X)
    alpha = 1e-2
    method = "CP"
    method = "Tucker"
    #method = "KSCP"
    #method = "KSTucker"
    #method = "DistanceTucker"
    #method = "DistanceCP"
    #method = "TuckerProd"
    #method = "CPProd"
    #method="CP"

    #method = "KPCP"
    #method = "KPTucker"
    #Ls[0] = None
    #Ls[1] = None
    #Ls = [None,None,None]
    

    normX = numpy.linalg.norm(X)
    print "norm",normX
    #print max(X)
    #print sum(sign(X))
    X = X / normX
    
    #Y = alg.randomTensorOfNorm(size,rank,0.01) * 10 
    Y = X
    print Y.shape
    #Ls = [None,None,None]
    #Ls[0]=None
    #Ls[1]=None

    W = createMask(list(Y.shape),zerorate)
    print sum(W)," / ",prod(Y.shape)

    beta = 0
    I = alg.createUnitTensor(X.ndim,re)
   
    print "test fo Completion starts, size:", Y.shape

    print method
    if method == "CP":
        X = comp.CompletionCP_EveryStep(Y,W,re,Ls,alpha,beta)
    elif method == "Tucker":
        X = comp.CompletionTucker_EveryStep(Y,W,rank_estimate,Ls,alpha)
    elif method == "KSCP":
        X = comp.CompletionKS_CP_EveryStep(Y,W,re,Ls,alpha)
    elif method == "KSTucker":
        X = comp.CompletionKS_Tucker_EveryStep(Y,W,rank_estimate,Ls,alpha)
    elif method == "KPCP":
        X = comp.CompletionKP_CP_EveryStep(Y,W,re,Ls,alpha)
    elif method == "KPTucker":
        X = comp.CompletionKP_Tucker_EveryStep(Y,W,rank_estimate,Ls,alpha)
    elif method == "DistanceTucker":
        X = comp.CompletionDistance_Tucker_EveryStep(Y,W,rank_estimate,Ls,alpha)
    elif method == "DistanceCP":
        X = comp.CompletionDistance_CP_Everystep(Y,W,re,Ls,alpha)
    elif method == "CPProd":
        X = comp.CompletionCPProd_EveryStep(Y,W,re,Ls,alpha,beta)
    elif method == "TuckerProd":
        X = comp.CompletionTuckerProd_EveryStep(Y,W,rank_estimate,Ls,alpha)

    print "final estimation error:", norm(Y-X)
    #print "original \n",abs(Y-X)
    #print "estimated \n",X
    vint = vectorize(int)
    #Y=sign(Y)
    #X = X * normX
    #X=vint(X+0.5)

    #benchmark.SaveImage(Y,"Sugar")
    #benchmark.SaveImage(X,"Est10_Sugar")
    #Y=Y*(1-W)
    #X=X*(1-W)
    #Y=sign(Y)
    #X=sign(X)
    #benchmark.SaveImage(X.reshape(89*4,100*3),"Est_Flow")
    #benchmark.SaveImage(Y.reshape(151*2,151*19),"Enron")
    #benchmark.SaveImage(X.reshape(151*2,151*19),"Est_Enron")
    #benchmark.SaveImage(abs(Y-X).reshape(151*2,151*19),"EDiff_Enron")
    #print X



def createTestLaplacian(size):
    """
    実験用のラプラシアンを生成する。
    size x sizeの大きさの三重対角な行列を生成する。
    長さsizeの鎖状のグラフに相当するラプラシアンを返す。
    """
    A = repeat(0,size*size).reshape(size,size)
    for i in xrange(size-1):
        A[i,i+1] = A[i+1,i] = 1

    return alg.createLaplacian(A)

import benchmark

def testTucker():
    """
    Tucker分解テスト
    """
    s = 8
    r = 5 
    re = 10
    size = [s,s,s]
    rank = [r,r,r]
    rank_estimate = [re,re,re]
   
    L = createTestLaplacian(s)
    alpha = 0.01

    while True:
        #X = alg.randomTensorOfNorm(size,rank,0)
        #dat = benchmark.ThreeDNoseData()
        dat = benchmark.Bonnie()
        #dat = benchmark.Enron()
        #dat = benchmark.Flow_Injection()
        X = dat["X"]
        L = dat["L"]
        X = X / norm(X)
        L=[None,None,None]

        print "start estimation for tensor size of", X.shape 

        (G,As) = alg.HOOI_obsolete(X,rank_estimate,alpha,L)


        print "finished"

        Result = alg.expand(G,As)

        #print "original \n", X
        #print "estimated \n",Result
        print "error \n", norm(X - Result)
        raw_input()


def testCP():
    """
    CP分解テスト
    """
    s = 100 
    size = [s,s,s]
    rank = 10
    rank_estimate = 15
   
    alpha = 0.1
    beta = 0.00

    #X = alg.randomTensorOfNorm(size,rank,0.05)
    #dat = benchmark.ThreeDNoseData()
    dat = benchmark.RandomSmallTensor()
    #dat = benchmark.Flow_Injection()
    #dat = benchmark.Bonnie()
    #dat = benchmark.Enron()
    #dat = benchmark.Sugar()
    #dat = benchmark.Artificial()
    while True:
        X = dat["X"]
        L = dat["L"]

        #X = X - mean(X)
        #X = X + min(X.reshape(prod(X.shape)))
        print "norm:",norm(X)

        originalNorm = norm(X)
        X = X / norm(X)
        #X = X+0.1
        
        L = [None,None,None]
        print X.shape

        #As = alg.RRMFCP(X,rank_estimate,alpha,[L,None,None],beta)
        start = datetime.datetime.today()
        As = alg.RRMFCP(X,rank_estimate,beta,L,alpha)
        end = datetime.datetime.today()
        #As = cpold.RRMFCP(X,rank_estimate,alpha)

        print "finished"
        print end-start
        
        I = alg.createUnitTensor(len(size),rank_estimate)
        Result = alg.expand(I,As)

        #vint = vectorize(int)
        #Result = vint(Result*originalNorm+0.5) / originalNorm
        #Result = sign(vint(Result))

        #benchmark.SaveImage(X.reshape(151*2,151*19),"Sugar")
        #benchmark.SaveImage(Result.reshape(151*2,151*19),"Est_Sugar")
        #benchmark.SaveImage(abs(X-Result).reshape(151*2,151*19),"Diff_Sugar")
        #Result = Result / norm(Result)
        
        #print "original \n", X
        #print "estimated \n",abs(X-Result)
        print "error \n", norm(X - Result)
        #print As[0]
        raw_input()

def testL():
    """
    過去の遺産
    """
    s = 100 
    size = [s,s,s]
    rank = 4
    rank_estimate = 15
   
    alpha = 0.1
    beta = 0.00

    while True:
        #X = alg.randomTensorOfNorm(size,rank,0.05)
        dat = benchmark.Bonnie()
        X = dat["X"]
        L = dat["L"]

        #X = X - mean(X)
        #X = X + min(X.reshape(prod(X.shape)))
        print "norm:",norm(X)

        originalNorm = norm(X)
        X = X / norm(X)
        #X = X+0.1
        
        print [det(l) for l in L]
        return
        print X.shape

        #As = alg.RRMFCP(X,rank_estimate,alpha,[L,None,None],beta)
        As = alg.RRMFCP(X,rank_estimate,beta,L,alpha)
        #As = cpold.RRMFCP(X,rank_estimate,alpha)

        print "finished"
        
        I = alg.createUnitTensor(len(size),rank_estimate)
        Result = alg.expand(I,As)

        #vint = vectorize(int)
        #Result = vint(Result*originalNorm+0.5) / originalNorm
        #Result = sign(vint(Result))

        #benchmark.SaveImage(X.reshape(151*2,151*19),"Sugar")
        #benchmark.SaveImage(Result.reshape(151*2,151*19),"Est_Sugar")
        #benchmark.SaveImage(abs(X-Result).reshape(151*2,151*19),"Diff_Sugar")
        #Result = Result / norm(Result)
        
        #print "original \n", X
        print "estimated \n",abs(X-Result)
        print "error \n", norm(X - Result)
        #print As[0]
        raw_input()

#testL()
#mat.Matlab.Open()
#testTucker()
#testCP()
testCompletion()
#testCP()

#mat.Matlab.Open()
##Evaluation_Tucker()
#while True:
#    testCP()
#    #testTucker()
#    #testCompletion()
#    raw_input("")
#
#mat.Matlab.Close()




