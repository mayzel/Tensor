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

#X = array([[[1,2,3],
#            [4,5,6],
#            [7,8,9],
#            [10,11,12]],
#           [[13,14,15],
#            [16,17,18],
#            [19,20,21],
#            [22,23,24]]
#           ])
#X = array([[[1,4,7,10],
#            [2,5,8,11],
#            [3,6,9,12]],
#           [[13,16,19,22],
#            [14,17,20,23],
#            [15,18,21,24]]])
#



#軸の順序がそのままなので注意すること！
def randomTensorOfNorm(sizelist,ranks):
    X = 0
    A = zeros(sizelist)
    N = len(sizelist)
    As = [random.rand(sizelist[i],ranks[i]) for i in xrange(N)]

    G = random.rand(*ranks)
    return alg.expand(G,As)

    return A
def createMask(sizelist,rate):
    elems = prod(sizelist)
    W = repeat(1,elems)
  
    for i in xrange(round(elems*rate)):
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
def testCompletion():
    size = [3,3,3]
    rank = [2,2,2]
    rank_estimate = [2,2,2]
    W = createMask(size,0.5)
    W = Wtest
    #print W
    Y = randomTensorOfNorm(size,rank)
    #Y = arange(prod(size)).reshape(*size) + 3
    #print Y
    #raw_input("")
    Y = Ytest

    X = comp.Completion(Y,W,rank_estimate)
    print "original \n",Y
    print "estimated \n",X


def testTucker():
    size = [3,3,3]
    rank = [3,3,3]
    rank_estimate = [1,1,1]
    
    X = randomTensorOfNorm(size,rank)

    (G,As) = alg.HOOI(X,rank_estimate)

    print "finished"

    Result = alg.expand(G,As)

    print "original \n", X
    print "estimated \n",Result
    print "error \n", norm(X - Result)

testCompletion()
#testTucker()


#X = arange(240).reshape(8,5,6)
#U = array([[1,3,5],[2,4,6]])
#v = array([1,2,3,4])
#print X
#print U
#print v
#print alg.Nproduct(X,v,1)
#print alg.Nproduct(X,U,0)


#(L,As) = alg.CP_ALS(X,1)





