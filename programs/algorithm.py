#coding: utf-8
"""
テンソル分解の為のコード一式。
主に変数の更新式からなる。補完はしない。
テンソルの基本的な演算もここ。
"""

#from scipy import *

from numpy import *
from numpy.linalg import *

import itertools as it

import const

from logger import *


from TensorComputation import *



def separateLaplacian(L,size):
    """
    グラフラプラシアンを類似度行列と対角成分に分解
    """
    if L == None:
        return ones(size),eye(size)

    D = diag(L)+1
    K = diag(D) - L
    
    return D,K



def HOOI(X,Rs,alpha = 0.0,Ls = None):
    """
    Tucker分解
    """
    N = X.ndim

    if Ls == None:
        Ls=[]
        for i in xrange(N):
            Ls.append(None)

    assert(isinstance(Ls,list))

    #print "initializing..."
    As = initializeAs(X,Rs)

    gnorm = 0
    gnormold = 0

    threshold = const.ConvergenceThreshold_Decomposition
    G = None
    for time in it.count():
        #print "iteration: " + str(time)
        (G,As) = HOOIstep(G,As,X,Rs,alpha,Ls)

        #収束性判定（重そう）
        gnorm = norm(G)
        if(abs(gnorm - gnormold) < threshold):
            #print "Tucker: converged after ", time, " iterations."
            break
        gnormold = gnorm

    return (G,As)

#Modified Gram-Schmidt orthonormalization
def mgs(a):
    """
    MGS     Modified Gram-Schmidt QR factorization.
        q, r = mgs(a) uses the modified Gram-Schmidt method to compute the
        factorization a = q*e for m-by-n a of full rank,
        where q is m-by-n with orthonormal columns and R is n-by-n.
    """
    (m,n) = a.shape
    q = zeros((m,n))
    r = zeros((n,n))
    for k in range(n):
        r[k, k]     = linalg.norm(a[:, k])
        q[:, k]     = a[:,k] / r[k,k]
        r[k, k+1:n] = dot(q[:,k], a[:,k+1:n])
        a[:, k+1:n] = a[:,k+1:n] - outer(q[:,k], r[k,k+1:n])
    return q, r

#solve  max tr(A'X) w.r.t. X
def MaximizeTraceAX(A):
    """
    Maximize tr(A'X) w.r.t. X, subject to X'X=I
    """
    colindex = argsort(map(lambda v:norm(v),A))[::-1]
    (X,r) = mgs(A.T[:,tuple(colindex)])
    return X


#距離を変えるバージョン
def TuckerDistanceStep(G,As,X,Rs,Ls,alpha=0.0):
    """
    損失関数にラプラシアンを導入するTucker分解1ステップ
    """
    N=X.ndim
    if As==None:
        As = initializeAs(X,Rs)
    for time in xrange(1):
        for n in xrange(N):
            Y = X
            for i in xrange(N):
                if Ls[i] != None:
                    if n==i:
                        Y = Y + Nproduct(Y,alpha*Ls[i],i)
                    else:
                        Y = Nproduct(Y,As[i].T + alpha*dot(As[i].T,Ls[i]),i)
                else:
                    if not n==i:
                        Y = Nproduct(Y,As[i].T,i)
            Yn = unfold(Y,n)
            As[n] = getLeadingSingularVects(Yn,Rs[n])

        G = X
        for n in xrange(N):
            if Ls[n] == None:
                G = Nproduct(G,As[n].T,n)
            else:
                G = Nproduct(G,As[n].T + alpha*dot(As[n].T,Ls[n]),n)
    return (G,As)


def CPDistanceStep(As,X,R,Ls,alpha=0.0,Xns=None):
    """
    損失関数にラプラシアンを入れるCP分解1ステップ
    """
    N=X.ndim
    if Xns == None:
        Xns = [unfold(X,n) for n in xrange(N)]
    if As==None:
        As = [getLeadingSingularVects(Xns[n],R) for n in xrange(N)]
    for times in xrange(1):
        for n in xrange(N):
            Left = eye(X.shape[n])
            if not Ls[n] == None:
                Left = Left + alpha * Ls[n]

            Right = ones((R,R))
            for i in (k for k in xrange(N) if k!=n):
                T = dot(As[i].T,As[i])
                if Ls[i] != None:
                    T = T + alpha * dot(dot(As[i].T,Ls[i]),As[i]) 
                Right = Right * T

            def getM(i):
                T = As[i]
                if Ls[i]!=None:
                    T = T + alpha*dot(Ls[i],As[i])
                return T
            Ms = (getM(i) for i in xrange(N) if i != n)
            C = reduce(lambda A,B:KRproduct(B,A),Ms)
            C = dot(Xns[n],C)
            if Ls[n] != None:
                C = C + alpha * dot(Ls[n],C)

            #LinvC = solve(Left,C)
            #LinvCRinv = solve(Right.T,LinvC.T).T
            #As[n] = LinvCRinv
            
            print "hogehoge"
            #As[n] = solveGeneralizedSylvester([Left,None],[None,Right],C)

            print C
            print dot(Left,As[n])+dot(As[n],Right)

    return As


def TuckerKsumStep(G,As,X,Rs,Ls,Ps,Ds,alpha = 0.0):
    """
    テンソル全体に対する正則化/Tucker分解1ステップ
    """
    N=X.ndim
    NAZO = False#謎アルゴリズムを使うかどうか
    repeatingTimes = 1
    def createG(X,As):
        G = X
        if NAZO:
            for n in xrange(N):
                if Ls[n] == None:
                    G = Nproduct(G,As[n].T,n)
                else:
                    I = eye(Rs[n])
                    dd = dot(As[n].T,Ls[n])
                    dd = dot(dd,As[n])
                    G = Nproduct(G,solve(I + alpha * dd, As[n].T),n)
        else:
            for n in xrange(N):
                G = Nproduct(G,As[n].T,n)
        return G
    if As == None:
        As = initializeAs(X,Rs)
    if G == None:
        print "RESET G"
        G = createG(X,As)
    #IL=[]
    ILinv=[]
    if not NAZO:
        for i in xrange(len(Ls)):
            if Ls[i] != None:
                #d = 1.0 / sqrt(1 + alpha * Ds[i])
                d = sqrt(1 + alpha * Ds[i])
                dinv = 1.0 / d

                ILinv.append(dot(Ps[i]*dinv,Ps[i].T))
            else:
                ILinv.append(eye(X.shape[0]))
                #ILinv.append(eye(*Ls[i].shape))

        for time in xrange(repeatingTimes):
            for n in xrange(N):
                Y = X
                for i in xrange(N):
                    if i != n:
                        Y = Nproduct(Y,transpose(As[i]),i) 

                if Ls[n] != None:
                    Yn = dot(ILinv[n],unfold(Y,n))
                    And = getLeadingSingularVects(Yn,Rs[n])
                    An = dot(ILinv[n],And)
                    As[n] = An
                else:
                    Yn = unfold(Y,n)
                    As[n] = getLeadingSingularVects(Yn,Rs[n])

            G = createG(X,As)
    else:
        for time in xrange(repeatingTimes):
            for n in xrange(N):
                Y = G
                for i in xrange(N):
                    if i != n:
                        Y = Nproduct(Y,As[i],i) 
                S = dot(unfold(Y,n),unfold(X,n).T)
                #もっといい方法があるはず…
                As[n] = MaximizeTraceAX(S)
            G = createG(X,As)

    return (G,As)



#columnwise minimizing : tr(U'AU) + tr(UBU') - 2tr(UX')
#where A and B are symmetric
#update AX + XB = C 
#P and D is eigenvalue decomp of A
def sylvesterColumnwiseUpdating(P,D,B,C,X):
    """
    列ごとにSylvester方程式を解く。どうやらAとBが正定値でないと収束は保証されないっぽい。
    """
    (n,m) = X.shape
    for d in xrange(m):
        de = 1.0 / (D + B[d,d])
        Linv = dot(P*de,P.T)

        b = B[:,d]
        R = C[:,d] - dot(X,b) + X[:,d] * B[d,d]

        u = dot(Linv,R)
        X[:,d] = u
    return X
#A = random.rand(3,3);A=A+A.T#+eye(3)
#B = random.rand(2,2);B=B+B.T#;B+=eye(2)
#C = random.rand(3,2)
#D,P=eigh(A)
#X = random.rand(3,2)
#for i in xrange(2):
#    X=sylvesterColumnwiseUpdating(P,D,B,C,X)
#print dot(A,X)+dot(X,B)
#print C


def TuckerKprodStep(G,As,X,Rs,Ds,Ws,PWs,DWs,alpha = 0.0):
    """
    テンソル全体に対する正則化/Kornecker積バージョン/Tucker分解1ステップ
    """
    N=X.ndim
    #謎アルゴリズムを用いる
    def createG(X,As):
        MX = X
        for n in xrange(N):
            MX = Nproduct(MX,As[n].T,n)

        # alpha << 1
        G = MX
        T = MX
        for n in xrange(N):
            T = Nproduct(T,dot(As[n].T*Ds[n],As[n]),n)
        G -= alpha * T
        T = MX
        for n in xrange(N):
            T = Nproduct(T,dot(As[n].T,dot(Ws[n],As[n])),n)
        G += alpha * T

        return G

    if As == None:
        As = initializeAs(X,Rs)
    if G == None:
        print "RESET G"
        G = createG(X,As)

    for time in xrange(1):
        for n in xrange(N):
            Y = G
            for i in xrange(N):
                if i != n:
                    Y = Nproduct(Y,As[i],i) 
            S = dot(unfold(Y,n),unfold(X,n).T)
            #もっといい方法があるはず…
            As[n] = MaximizeTraceAX(S)
        G = createG(X,As)

    return (G,As)

def CPKprodStep(As,X,R,Ds,Ws,PWs,DWs,alpha=0,Xns = None):
    """
    テンソル全体に対する正則化/Kronecker積バージョン/CP分解1ステップ
    """
    N=X.ndim
    if Xns == None:
        Xns = [unfold(X,n) for n in xrange(N)]
    if As==None:
        print "INITIALIZED"
        As = [getLeadingSingularVects(Xns[n],R) for n in xrange(N)]
        #As = [random.rand(Xns[n].shape[0],R)-0.5 for n in xrange(N)]

    for time in xrange(1):
        for n in xrange(N):
            S = reduce(lambda x, y: KRproduct(y,x), [As[i] for i in range(N) if i != n]) 
            StS = reduce(lambda x,y: dot(x.T,x)*dot(y.T,y), [As[i] for i in range(N) if i != n])
            #if Ls[n] != None:
            #else:
            XS = solve(StS,dot(Xns[n],S).T).T

            B = zeros((R,R))
            Bf = ones((R,R))
            for i in (i for i in xrange(N) if i!=n):
                al = dot(As[i].T*Ds[i],As[i])
                Bf = Bf *al
            B = B + Bf
            C = zeros((R,R))
            Cf = ones((R,R))
            for i in (i for i in xrange(N) if i!=n):
                al = dot(dot(As[i].T,Ws[i]),As[i])
                Cf = Cf *al
            C = C + Cf

            B = solve(StS,B.T).T
            C = solve(StS,C.T).T

            An = XS - alpha*dot(((XS.T)*Ds[n]).T , B) + alpha*dot(Ws[n],dot(XS,C))
            As[n] = An

    return As

def CPKsumStep(As,X,R,Ls,Ps,Ds,alpha=0,Xns = None):
    """
    テンソル全体に対する正則化/CP分解1ステップ
    """
    N=X.ndim
    if Xns == None:
        Xns = [unfold(X,n) for n in xrange(N)]
    if As==None:
        print "INITIALIZED"
        As = [getLeadingSingularVects(Xns[n],R) for n in xrange(N)]
        #As = [random.rand(Xns[n].shape[0],R)-0.5 for n in xrange(N)]

    import scipy.linalg
    for time in xrange(1):
        for n in xrange(N):
            An = As[n]
            S = reduce(lambda x, y: KRproduct(y,x), [As[i] for i in range(N) if i != n]) 
            StS = reduce(lambda x,y: dot(x.T,x)*dot(y.T,y), [As[i] for i in range(N) if i != n])
            #StS=dot(S.T,S)

            #M = cholesky(StS)
            C = dot(Xns[n],S)

            B = zeros((R,R))
            ilst = [i for i in xrange(N) if i!=n]
            for k in ilst:
                if Ls[k] == None:
                    continue
                Bf = ones((R,R))
                #Bf = eye(R,R)
                for i in ilst:
                    if i != k:
                        Bf = Bf * dot(As[i].T,As[i])
                        #Bf = KRproduct(As[i],Bf)
                    else:
                        half = dot(Ls[i],As[i])
                        al = dot(half.T,half)
                        Bf = Bf *al
                        #Bf = KRproduct(half,Bf)
                #B = B + dot(Bf.T,Bf)*alpha
                B = B + Bf * alpha

            (B,res,rank,s) = lstsq(StS,B.T);B=B.T
            (C,res,rank,s) = lstsq(StS,C.T);C=C.T
            #(B,res,rank,s) = lstsq(S,B.T);B=B.T
            #(C,res,rank,s) = lstsq(S,C.T);C=C.T
            if Ls[n] != None:
                An = sylvesterColumnwiseUpdating(Ps[n],1+alpha*Ds[n],B,C,An)
                #An = lyap(eye(*Ls[n].shape)+alpha*Ls[n],B,-C)
            else:
                An = solve(B + eye(*B.shape),C.T).T
            As[n] = An

    return As


def RRMFCP(X,R,beta,Ls=None,alpha=0,Ps=None,Ds=None):
    """
    CP分解
    """
    #log = Logger("CP")
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

    Xns=[]
    for n in xrange(N):
        Xns.append(unfold(X,n))

    As = []
    for n in xrange(N):
        As.append(getLeadingSingularVects(Xns[n],R))

    Itensor = createUnitTensor(N,R)
    Xns = [unfold(X,n) for n in xrange(N)]

    e = const.ConvergenceThreshold_Decomposition 
    xnorm = 0
    xnormold = -10000
    for time in it.count():
        As = RRMFCPstep(As,X,R,beta,Ls,Ps,Ds,alpha,Xns)

        if time % const.CheckingFrequence == 0:
            Xest = expand(Itensor,As)
            xnorm = norm(X - Xest)
            #log.WriteLine(str(xnormold - xnorm))
            print xnorm, " ", -(xnorm - xnormold)
            #raw_input()
            if abs(xnorm-xnormold) < e * const.CheckingFrequence:
                #print "CP: converged after ", time, " iterations."
                return As
            xnormold = xnorm

    return As

#from mlabwrap import mlab
#solve AX + XB + C = 0

#import matlab as mat 
def lyap(A,B,C):
    """
    Matlabが呼べる環境でのみ使える。lyapのラッパ。
    """
    pass
    #return mlab.lyap(A,B,C)
    #import mlabwrap as mlab
    #return mlab.lyap(A,B,C)
    #return mat.Matlab.lyap(A,B,C)

#mat.Matlab.Open()
#A = random.rand(3,3)
#B=random.rand(3,3)
#C=random.rand(3,3)
#X=lyap(A,B,C)
#print X
#print dot(A,X) + dot(X,B) + C
#mat.Matlab.Close()

def SumCP(L,A,B,C):
    """
    CP分解の結果から元のテンソルを作る。（低速）
    [Obsolete]
    """
    r = len(L)
    NA = len(A)
    NB = len(B)
    NC = len(C)
    
    print [r,NA,NB,NC]
    X = zeros((NA,NB,NC))
    for (i,j,k) in [(i,j,k) for i in range(NA) for j in range(NB) for k in range(NC)] :
        for n in range(r):
            X[i,j,k] += A[i,n]*B[j,n]*C[k,n]*L[n]
    return X


def getG(X,As):
    """
    テンソルと因子からTucker分解の場合に最適なコアテンソルを生成。
    """
    G = X
    for n in xrange(X.ndim):
        G = Nproduct(G,As[n].transpose(),n)
    return G



def createLaplacian(A):
    """
    類似度行列からラプラシアンを生成
    """
    dims = map(lambda v:sum(v),A)
    D = diag(dims)
    L = D - A
    return L


#HOOI  収束の判定をまだ作っていないのでとりあえず１００回。うまくできないものか…
def HOOI_obsolete(X,Rs,alpha = 0.0,Ls = None):
    """
    Obsolete
    """
    N = X.ndim

    if Ls == None:
        Ls=[]
        for i in xrange(N):
            Ls.append(None)

    assert(isinstance(Ls,list))

    #print "initializing..."
    As = initializeAs(X,Rs)

    errorprev = 0 
    gnorm = 0
    gnormold = 0

    threshold = const.ConvergenceThreshold_Decomposition
    for time in it.count():
        #print "iteration: " + str(time)
        for n in xrange(N):
            Y = X
            for i in xrange(N):
                if i != n:
                    Y = Nproduct(Y,transpose(As[i]),i) 

            Yn = unfold(Y,n)
            
            L = Ls[n]
            if L != None:
                An = getLeadingEigenVects(dot(Yn,Yn.T) - alpha * L, Rs[n])
            else:
                An = getLeadingSingularVects(Yn,Rs[n])
            As[n] = An

        if(time % const.CheckingFrequence == 0):
            #収束性判定（重そう）
            gnorm = norm(getG(X,As))
            print gnorm
            if(abs(gnorm - gnormold) < threshold):
                #print "Tucker: converged after ", time, " iterations."
                break
            gnormold = gnorm

    G = getG(X,As)

    return (G,As)





