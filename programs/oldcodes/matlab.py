# coding: utf-8

#MatlabŒÄ‚Ño‚µŠÖ”‚Ìƒ‰ƒbƒp

#pymat 
#import pymat

from numpy import *

class Matlab:
    
    @staticmethod
    def __del__():
        Matlab.Close()

    @staticmethod
    def Eval(exp):
        return pymat.eval(Matlab.Instance,exp)

    @staticmethod
    def Put(name,val):
        pymat.put(Matlab.Instance,name,val)

    @staticmethod
    def Get(name):
        return pymat.get(Matlab.Instance,name)

    @staticmethod
    def Open():
        print "called Open()"
        if (not hasattr(Matlab,"Instance")) or Matlab.Instance == None:
            Matlab.Instance = pymat.open()

    @staticmethod
    def Close():
        if Matlab.Instance != None:
            pymat.close(Matlab.Instance)

    #solve AX + XB + C = 0
    @staticmethod
    def lyap(A,B,C):
        Matlab.Put("A",A)
        Matlab.Put("B",B)
        Matlab.Put("C",C)
        Matlab.Eval("R = lyap(A,B,C)")
        
        R = pymat.get(Matlab.Instance,"R")
        return R
        #return Matlab.Get('R')

