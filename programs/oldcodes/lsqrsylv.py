#coding: utf-8

import itertools as it
import const
from logger import *

import numpy as np
from math import sqrt
from scipy.sparse.linalg.interface import aslinearoperator
__all__ = ['lsqr']

def _sym_ortho(a,b):
    aa = abs(a)
    ab = abs(b)
    if b == 0.:
        s = 0.
        r = aa
        if aa == 0.:
            c = 1.
        else:
            c = a/aa
    elif a == 0.:
        c = 0.
        s = b / ab
        r = ab
    elif ab >= aa:
        sb = 1
        if b < 0: sb=-1
        tau = a/b
        s = sb * (1 + tau**2)**-0.5
        c = s * tau
        r = b / s
    elif aa > ab:
        sa = 1
        if a < 0: sa = -1
        tau = b / a
        c = sa * (1 + tau**2)**-0.5
        s = c * tau
        r = a / c

    return c, s, r


#一般化シルベスタ方程式を解く
#L1XR1 + L2XR2 + ... = C
#ただし、L1\otimesR1 + L2\otimesR2 ... は対称
#またLi Riはすべて正方
def lsqrsylv(Left,Right, C, damp=0.0, atol=1e-8, btol=1e-8, conlim=1e8,
         iter_lim=None, show=False, calc_var=False):
    def dotsylv(V):
        def dots(L,R):
            if L == None and R == None:
                return V
            elif L != None and R == None:
                return np.dot(L,V)
            elif L==None and R!=None:
                return np.dot(V,R)
            else:
                return np.dot(np.dot(L,V),R)
        terms = (dots(Left[i],Right[i]) for i in xrange(len(Left)))
        return reduce(lambda a,b:a+b,terms)

    
    n = np.prod(C.shape)
    m = n
    if iter_lim is None: iter_lim = n*2
    var = np.zeros(C.shape)

    itn = 0
    istop = 0
    nstop = 0
    ctol = 0
    if conlim > 0: ctol = 1/conlim
    anorm = 0
    acond = 0
    dampsq = damp**2
    ddnorm = 0
    res2 = 0
    xnorm = 0
    xxnorm = 0
    z = 0
    cs2 = -1
    sn2 = 0

    """
    Set up the first vectors u and v for the bidiagonalization.
    These satisfy  beta*u = b,  alfa*v = A'u.
    """
    __xm = np.zeros(m) # a matrix for temporary holding
    __xn = np.zeros(n) # a matrix for temporary holding
    V = np.zeros(C.shape)
    U = C
    X = np.zeros(C.shape)
    alfa = 0
    beta = np.linalg.norm(U)
    W = np.zeros(C.shape)

    if beta > 0:
        U = (1/beta) * U
        V = dotsylv(U) #symmetric
        alfa = np.linalg.norm(V)

    if alfa > 0:
        V = (1/alfa) * V
        W = V.copy()

    rhobar = alfa
    phibar = beta
    bnorm = beta
    rnorm = beta
    r1norm = rnorm
    r2norm = rnorm

    # Reverse the order here from the original matlab code because
    # there was an error on return when arnorm==0
    arnorm = alfa * beta
    if arnorm == 0:
        print msg[0];
        return X, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var

    head1  = '   Itn      x[0]       r1norm     r2norm ';
    head2  = ' Compatible    LS      Norm A   Cond A';

    # Main iteration loop.
    while itn < iter_lim:
        itn = itn + 1
        """
        %     Perform the next step of the bidiagonalization to obtain the
        %     next  beta, u, alfa, v.  These satisfy the relations
        %                beta*u  =  a*v   -  alfa*u,
        %                alfa*v  =  A'*u  -  beta*v.
        """
        U = dotsylv(V) - alfa * U
        beta = np.linalg.norm(U)

        if beta > 0:
            U = (1/beta) * U
            anorm = sqrt(anorm**2 + alfa**2 + beta**2 + damp**2)
            V = dotsylv(U) - beta * V
            alfa  = np.linalg.norm(V)
            if alfa > 0:
                V = (1 / alfa) * V

        # Use a plane rotation to eliminate the damping parameter.
        # This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
        rhobar1 = sqrt(rhobar**2 + damp**2)
        cs1 = rhobar / rhobar1
        sn1 = damp / rhobar1
        psi = sn1 * phibar
        phibar = cs1 * phibar

        # Use a plane rotation to eliminate the subdiagonal element (beta)
        # of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
        cs, sn, rho = _sym_ortho(rhobar1, beta)

        theta = sn * alfa
        rhobar = -cs * alfa
        phi = cs * phibar
        phibar = sn * phibar
        tau = sn * phi

        # Update x and w.
        t1 = phi / rho
        t2 = -theta / rho
        DK = (1 / rho) * W

        X = X + t1 * W
        W = V + t2 * W
        ddnorm = ddnorm + np.linalg.norm(DK)**2

        if calc_var:
            var = var + DK**2

        # Use a plane rotation on the right to eliminate the
        # super-diagonal element (theta) of the upper-bidiagonal matrix.
        # Then use the result to estimate norm(x).
        delta = sn2 * rho
        gambar = -cs2 * rho
        rhs = phi - delta * z
        zbar = rhs / gambar
        xnorm = sqrt(xxnorm + zbar**2)
        gamma = sqrt(gambar**2 +theta**2)
        cs2 = gambar / gamma
        sn2 = theta  / gamma
        z = rhs / gamma
        xxnorm = xxnorm  +  z**2

        # Test for convergence.
        # First, estimate the condition of the matrix  Abar,
        # and the norms of  rbar  and  Abar'rbar.
        acond = anorm * sqrt(ddnorm)
        res1 = phibar**2
        res2 = res2 + psi**2
        rnorm = sqrt(res1 + res2)
        arnorm = alfa * abs(tau)

        # Distinguish between
        #    r1norm = ||b - Ax|| and
        #    r2norm = rnorm in current code
        #           = sqrt(r1norm^2 + damp^2*||x||^2).
        #    Estimate r1norm from
        #    r1norm = sqrt(r2norm^2 - damp^2*||x||^2).
        # Although there is cancellation, it might be accurate enough.
        r1sq = rnorm**2 - dampsq * xxnorm
        r1norm = sqrt(abs(r1sq))
        if r1sq < 0:
            r1norm = -r1norm
        r2norm = rnorm

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.
        test1 = rnorm / bnorm
        test2 = arnorm / (anorm * rnorm)
        test3 = 1 / acond
        t1 = test1 / (1 + anorm * xnorm / bnorm)
        rtol = btol + atol *  anorm * xnorm / bnorm

        # The following tests guard against extremely small values of
        # atol, btol  or  ctol.  (The user may have set any or all of
        # the parameters  atol, btol, conlim  to 0.)
        # The effect is equivalent to the normal tests using
        # atol = eps,  btol = eps,  conlim = 1/eps.
        if itn >= iter_lim: istop = 7
        if 1 + test3 <= 1: istop = 6
        if 1 + test2 <= 1: istop = 5
        if 1 + t1 <= 1: istop = 4

        # Allow for tolerances set by the user.
        if test3 <= ctol: istop = 3
        if test2 <= atol: istop = 2
        if test1 <= rtol: istop = 1

        if istop != 0: break

    # End of iteration loop.
    # Print the stopping condition.
    if show:
        print ' '
        print 'LSQR finished'
        print msg[istop]
        print ' '
        str1 = 'istop =%8g   r1norm =%8.1e' % (istop, r1norm)
        str2 = 'anorm =%8.1e   arnorm =%8.1e' % (anorm, arnorm)
        str3 = 'itn   =%8g   r2norm =%8.1e' % (itn, r2norm)
        str4 = 'acond =%8.1e   xnorm  =%8.1e' % (acond, xnorm)
        print str1+ '   ' + str2
        print str3+ '   ' + str4
        print ' '

    return X

#from numpy import *
#s=6
#k=2
#A = random.rand(s,s) - 0.5
#A = A+A.T
#B = random.rand(k,k) - 0.5
#B = B+B.T
#C = random.rand(s,k)
#U = lsqrsylv([None,A],[B,None],C)
#
#print dot(A,U)+dot(U,B)
#print C
