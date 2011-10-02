#coding: utf-8

from numpy import *
from numpy.linalg import *

import itertools as it


import os

import matplotlib.pyplot as plt

try:
    import cPickle as pickle
except:
    import pickle

def plot(information):
    if not type(information) == list:
        information = [information]
    
    title = ""
    plt.figure()
    if not title == "":
        plt.title()

    colors = []

    for i,info in it.izip(it.count(),information):
        result = info["result"]
        setting = info["setting"]
        method = setting["method"]
        withRel = setting["using relation data"]

        labelstr = method+":"+("with" if withRel else "without")

        y,y_err,x=[],[],[]
        print result[0.99]
        for maskrate in sorted(result.keys()):
            errorsdict = result[maskrate]

            errors=[]
            for param,errorsitem in errorsdict.iteritems():
                print errorsitem
                errors.extend(errorsitem)
            print errors
            m = mean(errors)
            s = std(errors)
            y.append(m)
            y_err.append(s)
            x.append(maskrate)
        plt.errorbar(x,y,y_err,mec="black",label=labelstr)
    
    plt.legend(loc=2)
    plt.show()


def loadInfo(filename):
    #try:
        f = file(filename,"r")
        info = pickle.load(f)
        f.close()
        return info
    #except:
        #return None

def getMethod(info):
    try:
        return info["setting"]["method"]
    except:
        return ""
def withRelation(info):
    try:
        return info["setting"]["using relation data"]
    except:
        return None




if __name__ == '__main__':
    import os
    os.chdir("jsonlog/Flow_Injection")
    files = os.listdir(".")

    files = [f for f in files if ".dat" in f]


    infotoplot=[]
    for f in files:
       information = loadInfo(f) 
       #print information
       if information==None:
           continue

       if withRelation(information) or True:
           #print f, getMethod(information)
           infotoplot.append(information)

    #print infotoplot
    plot(infotoplot)
