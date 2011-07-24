#coding: utf-8

from scipy import *

from numpy import *
from numpy.linalg import *

import itertools as it

from matlab import *

from algorithm import *


#CP分解　収束の判定がかなりいい加減
def CP_ALS(X,R):
    N = X.ndim
    As = []
    for n in xrange(N):
        #initialize A^(n)
        #とりあえずランダムでいいかな？
        As.append(rand(X.shape[getAxis(N,n)],R))

    print As
    e = 0.000001
    for time in xrange(100):
        for n in xrange(N):
#calculate V
            factors = [dot(As[i].transpose(),As[i]) for i in range(N) if i != n]
            V = reduce(lambda x, y: x*y, factors)

#calculate An
            mid = reduce(lambda x, y: KRproduct(y,x), [As[i] for i in range(N) if i != n]) #順序が逆
            Xn = unfold(X,n)
            Vt=V.transpose()
            Vd=solve(dot(V,Vt),Vt)
            
            print mid
            print Xn
            print Vd
            An = dot(dot(Xn,mid),Vd)

            L = map(lambda l: norm(l) , An.transpose())
            An = An / L #normalize

            As[n] = An #store

        if(time % 10 == 0):
            print "Lambda: " + str(L)
            print "A^("+str(n)+")"
            print An

            print map(lambda l: norm(l) , An.transpose())
            #ddd = raw_input("")
            print ""

    return (L,As)


#Tucker分解　HOSVD 未検証。とりあえず書くだけ書いた
def HOSVD(X,Rs):
    As = initializeAs(X,Rs)

    G = X
    for n in range(N):
        An = As[n]
        G = Nproduct(G,An.transpose(),n)

    return G



