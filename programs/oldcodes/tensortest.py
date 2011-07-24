# coding: utf-8

from scipy import *

#
#	参考URL
#	Tensor unfolding with numpy | Notes
#	http://pramook.wordpress.com/2009/07/14/tensor-unfolding-with-numpy/
#


from numpy import *
from numpy.linalg import *

import algorithm as alg
import Completion as comp

from logger import *

import matlab as mat
#import cpold 

def createMask(sizelist,zerorate):
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
    s = 10
    r = 3
    re = 7
    size = [s,s,s]
    rank = [r,r,r]
    rank_estimate = [re,re,re]
    zerorate = 0.9

    import benchmark
    #data = benchmark.Wine_v6()
    #data = benchmark.Bonnie()
    data = benchmark.Flow_Injection()
    #data = benchmark.ThreeDNoseData()
    #data = benchmark.RandomSmallTensor()
    #data = benchmark.Artificial()
    Ls = data["L"]
    X = data["X"]#
    import numpy.linalg
    #X = X - mean(X)
    normX = numpy.linalg.norm(X)
    print normX
    X = X / normX
    
    #Y = alg.randomTensorOfNorm(size,rank,0.01) * 10 
    
    Y = X

    print Y.shape
    #print map(lambda x:str(x.shape),Ls)
    alpha = 0.1

    #L = createTestLaplacian(s)
    Ls = [None,None,None]

    W = createMask(list(Y.shape),zerorate)

    beta = 0
    I = alg.createUnitTensor(len(size),re)
    def approximateCP(Xin):
        print re
        print alpha
        print Ls
        print beta
        As = alg.RRMFCP(Xin,re,beta,Ls,alpha)
        Xs = alg.expand(I,As)
        return Xs
    
    def approximateTucker(Xin):
        (G,As) = alg.HOOI(Xin,rank_estimate,alpha,Ls)
        Xs = alg.expand(G,As)
        return Xs
   
    print "test fo Completion starts, size:", Y.shape
   
    if False:
        if False:
            X = comp.CompletionCP(Y,W,re,Ls,alpha,beta)
        else:
            X = comp.CompletionTucker(Y,W,rank_estimate,Ls,alpha)
    else:
        import newalg
        X = newalg.HOOI(Y,W,Y,rank_estimate,alpha,Ls)
   # print "final est. error rate", abs(1- X / Y)
    print "final estimation error:", norm(Y-X)
    #print "original \n",abs(Y-X)
    #print "estimated \n",X



def createTestLaplacian(size):
    A = repeat(0,size*size).reshape(size,size)
    for i in xrange(size-1):
        A[i,i+1] = A[i+1,i] = 1

    return alg.createLaplacian(A)

import benchmark

def testTucker():
    s = 8
    r = 5 
    re = 3
    size = [s,s,s]
    rank = [r,r,r]
    rank_estimate = [re,re,re]
   
    L = createTestLaplacian(s)
    alpha = 0.01

    while True:
        #X = alg.randomTensorOfNorm(size,rank,0)
        dat = benchmark.ThreeDNoseData()
        X = dat["X"]
        L = dat["L"]
        X = X / norm(X)

        print "start estimation for tensor size of", X.shape 

        (G,As) = alg.HOOI(X,rank_estimate,L,alpha)


        print "finished"

        Result = alg.expand(G,As)

        #print "original \n", X
        #print "estimated \n",Result
        print "error \n", norm(X - Result)
        raw_input()


def testCP():
    s = 100 
    size = [s,s,s]
    rank = 4
    rank_estimate = 5
   
    alpha = 0.1
    beta = 0.0001

    while True:
        #X = alg.randomTensorOfNorm(size,rank,0.05)
        #dat = benchmark.ThreeDNoseData()
        #dat = benchmark.RandomSmallTensor()
        dat = benchmark.Flow_Injection()
        #dat = benchmark.Bonnie()
        #dat = benchmark.Artificial()
        X = dat["X"]
        L = dat["L"]

        #X = X - mean(X)
        #X = X + min(X.reshape(prod(X.shape)))
        print "norm:",norm(X)
        X = X / norm(X)
        #X = X+0.1
        
        L = [None,None,None]
        print X.shape

        #As = alg.RRMFCP(X,rank_estimate,alpha,[L,None,None],beta)
        As = alg.RRMFCP(X,rank_estimate,beta,L,alpha)
        #As = cpold.RRMFCP(X,rank_estimate,alpha)

        print "finished"
        
        I = alg.createUnitTensor(len(size),rank_estimate)
        Result = alg.expand(I,As)

        benchmark.SaveImage(X,"Flow")
        benchmark.SaveImage(Result,"Est_Flow")
        #Result = Result / norm(Result)
        
        #print "original \n", X
        #print "estimated \n",Result
        print "error \n", norm(X - Result)
        #print As[0]
        raw_input()

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




