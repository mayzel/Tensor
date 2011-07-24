#coding: utf-8


from numpy import *
from numpy.linalg import *

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
    for k in xrange(n):
        #print type(a[:,k])
        #print a[:,k].shape
        r[k, k]     = linalg.norm(a[:, k])
        q[:, k]     = a[:,k] / r[k,k]
        r[k, k+1:n] = dot(q[:,k], a[:,k+1:n])
        a[:, k+1:n] = a[:,k+1:n] - outer(q[:,k], r[k,k+1:n])
    return q, r

#A = random.randn(15,3)
#B = random.randn(15,15)
#A,q=mgs(A)
#print dot(dot(B,A),A.T)-B

def RedSVD(A,R):
    (n,m) = A.shape
    O = random.randn(n,R)
    #Y = dot(A.T,O)
    Y = A.T.dot(O)
    if hasattr(Y,"todense"):
        Y = Y.todense()
    Y,r = mgs(Y)

    #B = dot(A,Y)
    B = A.dot(Y)

    P = random.randn(R,R)
    #Z = dot(B,P)
    Z = B.dot(P)
    Z,r = mgs(Z)
    #C = dot(Z.T,B)
    C = Z.T.dot(B)
    (U,S,V)    = svd(C)
    V=V.T


    return dot(Z,U),S,dot(Y,V)

def getLeadingSingularVects(X,R):

    u,s,r = RedSVD(X,R)
    return u

def getLeadingSingularVects_obs(X,R):
    """
    大きい方からR個の左特異ベクトルをとる
    """
    return getLeadingEigenVects_obs(dot(X, X.T), R)

import numpy.linalg
import scipy.linalg
def getLeadingEigenVects_obs(X,R):
    """
    大きい方からR個の固有ベクトルをとる
    """
    X = (X + X.T)/2
    [val,vec] = scipy.linalg.eig(X)
    vec = real(vec)
    #[val,vec] = numpy.linalg.eigh(X)
    index = argsort(val, kind='mergesort')
    index = index[::-1]

    return vec[:,index[:R]]


if __name__=="__main__":
    s=500
    A = zeros((s,s))
    for i in range(300):
        A += outer(random.randn(s)+2,random.randn(s)+1)
    #A=random.randn(s,s)
    print "start"
    #print RedSVD(A,3)
    #print svd(A)

    E=getLeadingSingularVects(A,3).T
    for i in range(100):
        E+=getLeadingSingularVects(A,3).T
    E = E / 101

    F=getLeadingSingularVects_obs(A,3).T
    E = E * sign(E[0,0])
    F = F * sign(F[0,0])
    print E
    print F
    print E/F  
    print "end"
