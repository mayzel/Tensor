# coding: utf-8
"""@package Evaluation
クロスバリデーションによってテンソルの補完を評価するためのコード塊

"""



from numpy import *
import numpy.linalg
import CPWOPT

from logger import *

import Toolbox
import random

import itertools as it

import benchmark
import gc
import functools

#from info import *

def EvaluateCompletion(data,mask,method,useRelation,execTimes,logger,information,unobservedRates = None,alpha=None,ranks=None):
    """
    @param data 穴埋めするデータ
    @parma mask 未観測要素を作る方法を指定する。{"Random","Fiber","Slice"}のいずれかを取る。
    @param method 穴埋めの方法を指定する。{"CP","Tucker","KSCP","KSTucker","DistanceTucker","DistanceCP","TuckerProd","CPProd","KPCP""KPTucker"}のいずれかを取る。
    @param useRelation 関係情報を利用するかどうか。
    @param execTimes 実験するサンプルの数
    @param logger 実験結果を保存するためのロガーを与える。
    
    論文で用いた数値実験を行う。
    テンソル補完の方法を指定して、クロスバリデーションにより評価し、結果を逐次出力する。

    """
    #try:
    return EvaluateCompletionMain(data,mask,method,useRelation,execTimes,logger,information,unobservedRates,alpha,ranks)
    #except Exception,e:
        #print e


def EvaluateCompletionMain(data,mask,method,useRelation,execTimes,logger,information,unobservedRates = None,alpha=None,ranks=None):
    """
    数値実験本体
    """
    global log 
    log = logger

    varianceTimes = execTimes

    L = data["L"]
    X = data["X"]
    normX = numpy.linalg.norm(X)
    #X = X / normX

    if not useRelation:
        L = [None for i in range(X.ndim)]
        alpha = [1]

    #if unobservedRates == None:
    #    unobservedRates = array([0.5,0.75,0.9])
    #    #unobservedRates = array([0.75,0.9,0.95])
    #    #unobservedRates = unobservedRates[::-1]
    #if alpha == None:
    #    #alpha =[pow(10,x) for x in [-4,-3,-2,-1,0,1]] #for L
    #    alpha =[pow(10,x) for x in [-5,-4,-3,-2,-1]] #for L
    #    alpha =[pow(10,x) for x in [-7,-6,-5,-4,-3,-2,-1]] #for L

    #if ranks == None:
    #    #ranks = [2,3,5]
    #    ranks = [5,7,9]
    #    ranks = [40]
    #    ranks = [5,10,15]
    #    ranks=[35]
    #    #ranks = [7]

    shape = X.shape

    #alphaはLにしか関係ない
    if all(map(lambda i:i==None,L)):
        print "hogehogehogehogehogehoge"
        alpha = [1]


    maskAxis = 1
    elems = prod(X.shape) 
    print elems, "kdkdkdkd"
    if mask == "Random":
        targetelems = elems
        print "MASKING: RANDOM"
        def createObservedTensor(data):
            data = array(data)
            X = zeros(elems)
            def setter(index):
                X[index] = 1
            vectset = vectorize(setter)
            vectset(data)
            return X.reshape(shape)
    elif mask == "Fiber":
        targetelems = elems / X.shape[maskAxis]
        print "MASKING: FIBER"
        def createObservedTensor(data):
            data = array(data)
            S = zeros(targetelems)
            def setter(index):
                S[index] = 1
            vectset = vectorize(setter)
            vectset(data)
            X = zeros(elems).reshape(shape)
            if maskAxis == 0:
                S = S.reshape(shape[1],shape[2])
                for i in xrange(shape[0]):
                    X[i,:,:] = S
            elif maskAxis == 1:
                S = S.reshape(shape[0],shape[2])
                for i in xrange(shape[1]):
                    X[:,i,:] = S
            elif maskAxis == 2:
                S = S.reshape(shape[0],shape[1])
                for i in xrange(shape[2]):
                    X[:,:,i] = S

            return X.reshape(shape)
    elif mask == "Slice":
        targetelems = X.shape[maskAxis]
        print "MASKING: SLICE"
        def createObservedTensor(data):
            data = array(data)
            S = zeros(targetelems)
            def setter(index):
                S[index] = 1
            vectset = vectorize(setter)
            vectset(data)
            X = zeros(elems).reshape(shape)
            if maskAxis == 0:
                for i in xrange(X.shape[1]):
                    for j in xrange(X.shape[2]):
                        X[:,i,j] = S
            elif maskAxis == 1:
                for i in xrange(X.shape[0]):
                    for j in xrange(X.shape[2]):
                        X[i,:,j] = S
            elif maskAxis == 2:
                for i in xrange(X.shape[0]):
                    for j in xrange(X.shape[1]):
                        X[i,j,:] = S

            return X.reshape(shape)

    evalDataGenerator = lambda separatingNumber,unobservedRate,targetIndeces:dataGenerator(
            int(targetelems* unobservedRate),separatingNumber,unobservedRate, targetIndeces)

    hpOptimDataGenerator = lambda separatingNumber,unobservedRate,targetIndeces:dataGenerator(
            min(int(targetelems* unobservedRate * 0.5),int(len(targetIndeces) * 0.5)),
            separatingNumber,unobservedRate, targetIndeces)

    def dataGenerator(hiddens,separatingNumber,unobservedRate,targetIndeces):
        #print elems
        #print len(targetIndeces)
        rs = Toolbox.GenerateRandomSeparation(targetIndeces, hiddens)
        return Toolbox.Take(rs,separatingNumber)

    import CompletionMethods
    import Decomposition
    if method in ["Tucker","TuckerSum"]:
        decomposition = Decomposition.TuckerSum()
        completionMethod = CompletionMethods.Tucker(X,L,decomposition)
    elif method in ["CP","CPSum"]:
        print "hogehogehgoehgoe"
        decomposition = Decomposition.CPSum()
        completionMethod = CompletionMethods.CP(X,L,decomposition)
    elif method == "TuckerProd":
        decomposition = Decomposition.TuckerProd()
        completionMethod = CompletionMethods.Tucker(X,L,decomposition)
    elif method == "CPProd":
        decomposition = Decomposition.CPProd()
        completionMethod = CompletionMethods.CP(X,L,decomposition)
    elif method == "CPWOPT":
        completionMethod = CompletionMethods.CPWOPT(X,L)
        if not useRelation:
            alpha = [0]
    elif method == "CPWOPTProd":
        completionMethod = CompletionMethods.CPWOPTProd(X,L)
        if not useRelation:
            alpha = [0]

    #convert list of indeces to binary tensor
    completionMethod.createObservedTensor = createObservedTensor

    estimator = completionMethod.estimator
    

    def lossFunction(estimation,evalData):
        #evalData = Toolbox.Take(evalData,500)
        W = createObservedTensor(evalData)
        Y=estimation
        return numpy.linalg.norm((Y - X)*W) * sqrt(1.0*elems / len(evalData))

    trainingData = range(targetelems)

    log.WriteLine("Start Evaluatig method:" + method + " ")
    log.WriteLine("Using Relation Data" if useRelation else "Without Relation Data")
    log.WriteLine("Ranks for Estimation:"+str(ranks))
    log.WriteLine("Unobserved Rates:"+str(unobservedRates))
    log.WriteLine("HyperParameters alpha to try:"+str(alpha))
    log.WriteLine("hyperParameters rank to try:" + str(ranks))
    print type(information)
    information["setting"]={}
    information["setting"]["method"] = method
    information["setting"]["using relation data"] = useRelation
    information["setting"]["rank for estimation"] = ranks
    information["setting"]["fraction of unobserved elements"] = unobservedRates
    information["setting"]["tested alpha"] = alpha 
    information["setting"]["tested rank"] = ranks
    information["result"]={}


    for unobservedRate in unobservedRates:
        information["result"][unobservedRate]={}
        #log.WriteLine("unobserved rate, "+str(unobservedRate)+ " ")

        time = varianceTimes

        import CrossValidation
        
        print unobservedRate, "kkkkkkkk"
        parameters = [(a,rank) for a in alpha for rank in ranks] 
        errors = CrossValidation.Evaluate(
                trainingData,
                estimator,
                lossFunction,
                functools.partial(evalDataGenerator,time,unobservedRate),
                parameters,
                functools.partial(hpOptimDataGenerator,1,unobservedRate))

        #[log.Write(", " + str(error),False) for error in errors]
        for error in errors:
            e = error["error"]
            param = error["param"]
            log.WriteLine("unobserved, "+ str(unobservedRate)+", bestparam,"+str(param)+", error, "+str(e))

            if not param in information["result"][unobservedRate]:
                information["result"][unobservedRate][param]=[]
            information["result"][unobservedRate][param].append(e)
            print "score logged:", e
            


        log.WriteLine()

    return information
