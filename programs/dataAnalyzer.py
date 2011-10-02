#coding: utf-8

#from scipy import *

from numpy import *
from numpy.linalg import *

import itertools as it


import os

import matplotlib.pyplot as plt


IgnoreThreshold = 100.0
#あるデータのある手法ごとの処理
def processMethod(files):
    #print "###",files
    import csv
    errorCP={}
    errorTucker={}

    for fname in files:
        method=""
        content = open(fname)
        rows = (row for row in csv.reader(content))
        for row in rows:
            if row == []:
                continue
            if "CP" in row[0]:
                errors=errorCP
                method="CP"
            elif "Tucker" in row[0]:
                errors=errorTucker
                method="Tucker"

            if row !=[] and row[0] == "unobserved":
                unobsRate = float(row[1])
                error = float(row[-1])
                if str(error)=="nan":
                    continue
                #if "KSum" in fname and method == "CP":
                #    print "UnOBS : ",unobsRate, "   ERROR : ", error
                if not unobsRate in errors:
                    errors[unobsRate]=[]
                if error > IgnoreThreshold:
                    continue
                errors[unobsRate].append(error)

            if method=="CP":
                errorCP=errors
            elif method=="Tucker":
                errorTucker=errors
            else:
                assert(False)

    rCP={}
    for unobsRate in errorCP:
        m = mean(errorCP[unobsRate])
        s = std(errorCP[unobsRate])
        rCP[unobsRate] = {"mean":m,"std":s}
    rTucker={}
    for unobsRate in errorTucker:
        m = mean(errorTucker[unobsRate])
        s = std(errorTucker[unobsRate])
        rTucker[unobsRate] = {"mean":m,"std":s}
    return (rCP,rTucker)



#実験データごとの処理
def processOneData(files):
    #print "$$$",files
    methods = ["Normal","Relation","KSum"] 
    
    datCP ={}
    datTucker ={}
    for method in methods:
        #print method
        (rcp,rtucker) = processMethod([f for f in files if method in f])
        #print rcp
        def xyyerr(result):
            x=[]
            y=[]
            yerr=[]
            for k,v in result.iteritems():
                m = v["mean"]
                s = v["std"]
                x.append(k)
                y.append(m)
                yerr.append(s)
            if x==[]:
                return x,y,yerr
            l=argsort(x)
            return array(x)[l],array(y)[l],array(yerr)[l]
        x,y,yerr = xyyerr(rcp)
        datCP[method] = {"unobs":x,"error":y,"var":yerr}
        x,y,yerr = xyyerr(rtucker)
        datTucker[method] = {"unobs":x,"error":y,"var":yerr}

    return datCP,datTucker
#        pylab.errorbar(x,y,yerr,marker="s",mfc="red",mec="green",ms=20,mew=4)
        #raw_input()

def Process(dataname=None):
    path = os.getcwd()
    files = os.listdir(path)
    logfiles =  [f for f in files if ".log" in f]

    if dataname==None:
        datasets = ["Bonnie","Flow","ThreeD"]
    else:
        datasets = [dataname]

    num = 1
    data = {}
    for dname in datasets:
        result = processOneData([f for f in logfiles if dname in f])
        data[dname] = result

        cp, tucker = result
        plotData(num,cp,dname + ": CP")
        num+=1
        
        plotData(num,tucker,dname + ": Tucker")
        num+=1

    return data


colors={"Normal":"blue","Relation":"red","KSum":"green"}

def plotData(num,data, title):
    plt.figure(num)
    plt.xlabel("Unobserved Entry")
    plt.ylabel("Error")
    leglabels=[]
    for method,result in data.iteritems():
        x=result["unobs"]
        y=result["error"]
        yerr=result["var"]
        if x==[]:
            continue
        labels = {"KSum":"All","Relation":"Subspace","Normal":"Normal"}
        plt.errorbar(x,y,yerr,mfc=colors[method],mec="black",label=labels[method])
        #plt.ylim(ymin=0.0)
        #plt.ylim(ymax=0.8)
        plt.xlim(xmin=0.7)


    plt.legend(loc=2)
    plt.title(title)

    plt.show()


Process("ThreeD")
