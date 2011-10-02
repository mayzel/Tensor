#coding: utf-8

from numpy import *
from numpy.linalg import *

import itertools as it

import const

from logger import *


import scipy.sparse as sparse
import scipy.optimize as opt


import algorithm as alg
import SVD

import CPWOPTcritical as critical

global info

#def flattenAs(Div):
#    (dU,dV,dW) = Div
#    dU = dU.flatten()
#    dV = dV.flatten()
#    dW = dW.flatten()
#    return append(dU,append(dV,dW))


#matrix valued l-BFGS method
def LBFGS_matrix(f,xinit,fprime,maxiter):
    from Queue import Queue

    m = 8
    #ベクトルはぜんぶ行列
    #a,b / rho,s,y
    a,b,rho,s,y = [],[],[],[],[]

    x = xinit
    q = fprime(x)

    for iter in xrange(maxiter):
        #print f(x)
        #print "-------------------------"
        #raw_input()
        M = len(a)

        for k in range(M)[::-1]:
            a[k] = rho[k]*sum((s[k]*q).flatten())
            q -= a[k]*y[k]

        for k in range(M):
            b[k] = rho[k]*sum((y[k]*q).flatten())
            q += (a[k]-b[k])*s[k]

        q = q.reshape(x.shape)
        d = -q

        #golden separation method
        near=0.0;far=5.0
        goldenration = 0.618033988
        for sepaiter in range(6):
            tn = near + (far - near)*(1-goldenration)
            tf = near + (far - near)*goldenration

            vtn,vtf = f(x + d*tn),f(x + d*tf) #どれか一つが一番小さい
            if vtf < vtn:
                near = tn
            else:
                far = tf
        d = d * (far + near) * 0.5
        #print (near,far)


        xnew = x + d
        qnew = fprime(xnew).reshape(x.shape)

        snew = xnew - x
        ynew = qnew - q
        denom = sum((ynew*snew).flatten())
        if denom <= 0:
            x=xnew
            break
        rhonew = 1.0 / denom


        s.append(snew)
        y.append(ynew)
        rho.append(rhonew)
        a.append(0)
        b.append(0)
        if len(a) > m:
            a.pop(0)
            b.pop(0)
            s.pop(0)
            y.pop(0)
            rho.pop(0)
        x = xnew
        q = qnew
        #print "-------------------------"

    return x


def separateLaplacian(L,size):
    """
    グラフラプラシアンを類似度行列と対角成分に分解
    """
    if L == None:
        return ones(size),eye(size)

    D = diag(L)+1
    K = diag(D) - L
    
    return D,K
"""
CP-WOPTによってテンソルを補完する
"""
def CompletionGradient(X,shape,ObservedList,R,Ls,alpha=0,XoriginalTensor=None,isProd=False):
    #X is Dense Vector with only observed elements

    #Observed must be given as list of coordinate tuples
    #TrueX is 3 dimensional array


    N = 3
    Xns = [critical.unfold(X,n,shape,ObservedList) for n in xrange(N)]
    print "unfolded"
    As = [zeros((shape[n],R)) for n in xrange(N)]
    #As = [SVD.getLeadingSingularVects(Xns[n],R) for n in xrange(N)]
    Rev = 100
    for i in xrange(Rev):
        Asadd = [SVD.getLeadingSingularVects(Xns[n],R) for n in xrange(N)]
        for k in xrange(N):
            As[k] += Asadd[k]
    for k in xrange(N):
        As[k] /= Rev 
        #As[k] += random.randn(shape[k],R)*0.01



    n,m,l=shape
    def lossfunc(U,V,W):
        #print "loss start"
        XO = critical.HadamardProdOfSparseTensor(U,V,W,ObservedList)
        loss = norm(XO - X)
        #print "loss end"
        return loss

    #Xinit = flattenAs(As)
    #print "start bfgs"


    J = alg.createUnitTensor(3,R)
    #Mask = getMaskTensor(ObservedList,shape)

    U,V,W = As

    Ls[:] = [L*alpha if not L==None else 0 for L in Ls]

    Lu,Lv,Lw=Ls
    beta = 1e-8
    isProd = False

    if not isProd:
        LuUse = alpha * Lu + beta * eye(n) 
        LvUse = alpha * Lv + beta * eye(m) 
        LwUse = alpha * Lw + beta * eye(l) 
    #print U.shape, V.shape, W.shape

    Xest = XoriginalTensor #memory consuming


    threshold = 0.5*const.ConvergenceThreshold_NewCompletion
    maxiter=700
    bfgs_maxiter=3
    errorold = inf
    errorTest = inf
    expandedX = 0

    Dw,Kw = separateLaplacian(Lw,l)
    Dv,Kv = separateLaplacian(Lv,m)
    Du,Ku = separateLaplacian(Lu,n)

    import itertools
    for steps in itertools.count():
        if isProd:
            DSu = alpha * trace(dot(W.T*Dw,W)) * trace(dot(V.T*Dv,V))
            KSu = alpha * trace(dot(W.T,dot(Kw,W))) * trace(dot(V.T,dot(Kv,V)))
            LuUse = DSu * Du - KSu * Ku + eye(n)

        #print "optimization of U"
        #print [U.shape,V.shape,W.shape]
        grad = lambda U:critical.Gradient(X,ObservedList,(U,V,W),LuUse,shape,R,0)
        loss = lambda U:lossfunc(U,V,W)
        U = LBFGS_matrix(loss,U,grad,maxiter=bfgs_maxiter)
        #print [U.shape,V.shape,W.shape]

        if isProd:
            DSv = alpha * trace(dot(U.T*Du,U)) * trace(dot(W.T*Dw,W))
            KSv = alpha * trace(dot(U.T,dot(Ku,U))) * trace(dot(W.T,dot(Kw,W)))
            LvUse = DSv * Dv - KSv * Kv + eye(m)
        #print "optimization of V"
        grad = lambda V:critical.Gradient(X,ObservedList,(U,V,W),LvUse,shape,R,1)
        loss = lambda V:lossfunc(U,V,W)
        V = LBFGS_matrix(loss,V,grad,maxiter=bfgs_maxiter)

        if isProd:
            DSw = alpha * trace(dot(U.T*Du,U)) * trace(dot(V.T*Dv,V))
            KSw = alpha * trace(dot(U.T,dot(Ku,U))) * trace(dot(V.T,dot(Kv,V)))
            LwUse = DSw * Dw - KSw * Kw + eye(l)
        #print "optimization of W"
        grad = lambda W:critical.Gradient(X,ObservedList,(U,V,W),LwUse,shape,R,2)
        loss = lambda W:lossfunc(U,V,W)
        W = LBFGS_matrix(loss,W,grad,maxiter=bfgs_maxiter)

        #grad = lambda (U,V,W)
        
        if False:
            XO = critical.HadamardProdOfSparseTensor(U,V,W,ObservedList)
            normrate = norm(XO) / norm(X)
            qubicnormrate = normrate ** (1.0 / 3)
            U /= qubicnormrate
            V /= qubicnormrate
            W /= qubicnormrate

        if steps % 20 == 1:
            Xest = alg.expand(J,[U,V,W])
            errorTest = norm(Xest - XoriginalTensor)
            #print U

        errorObserved = lossfunc(U,V,W)


        if steps % 20 == 1:
            print "iter:",steps," err:",errorTest ," oberr:",errorObserved, " diff:", errorObserved-errorold, "norm;", norm(Xest)

        faultThreshold = 1e4
        if errorObserved > faultThreshold or errorObserved != errorObserved:
            #やりなおし
            print "-----Try Again-----"
            print steps," steps, observation error=",errorObserved
            return CompletionGradient(X,shape,ObservedList,R,Ls,alpha,XoriginalTensor)

        if abs(errorObserved - errorold) < threshold or steps >= maxiter:
            expandedX = alg.expand(J,[U,V,W])
            print "estimation finished in ",(steps+1),"steps."
            break

        errorold = errorObserved

    return expandedX


def getMaskTensor(ObsList,shape):
    n,m,l=shape
    W = zeros(shape)
    size = len(ObsList)/3
    for ind in xrange(size):
        p = ind * 3
        i = ObsList[p]
        j = ObsList[p+1]
        k = ObsList[p+2]
        W[i,j,k] = 1

    return W



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

def CompressSparseTensorToVector(*param):
    return critical.CompressSparseTensorToVector(*param)

if __name__=="main__":

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
        obsize = n*m*l * per / 100 *5
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



        
