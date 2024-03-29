# coding: utf-8
"""
テンソル分解のコードたち
"""


from numpy import *
import numpy.linalg
import algorithm as alg
import Completion as comp

from logger import *

import Toolbox
import random

import itertools as it

import DataStream
import benchmark
import gc
import functools


def EvaluateCompletionMain(data,mask,method,useRelation,execTimes,logger,unobservedRates = None,alpha=None,ranks=None):
    if mask == "Random":
        targetelems = elems
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

    def dataGenerator(hiddens,separatingNumber,unobservedRate,targetIndeces):
        #print elems
        #print len(targetIndeces)
        rs = Toolbox.GenerateRandomSeparation(targetIndeces, hiddens)
        return Toolbox.Take(rs,separatingNumber)

    evalDataGenerator = lambda separatingNumber,unobservedRate,targetIndeces:dataGenerator(
            int(targetelems* unobservedRate),separatingNumber,unobservedRate, targetIndeces)

    hpOptimDataGenerator = lambda separatingNumber,unobservedRate,targetIndeces:dataGenerator(
            min(int(targetelems* unobservedRate * 0.5),int(len(targetIndeces) * 0.5)),
            separatingNumber,unobservedRate, targetIndeces)

    if method == "Tucker":
        def estimator(param,trainingData):
            #rank_estimate = param
            print param
            print len(trainingData) , " / " , elems
            (alpha,r)=param 
            rank_estimate = [r,r,r]
            print "alpha:",alpha
            print "rank_estimate",rank_estimate

            trainingData = createObservedTensor(trainingData)
            Xobs = X * trainingData
            Xobs = X
            return comp.CompletionTucker_EveryStep(Xobs,trainingData,rank_estimate,L,alpha)
    elif method == "CP":
        def estimator(param,trainingData):
            print param
            print len(trainingData) , " / " , elems
            beta = 1e-8 #普通の正則化はとりあえず固定
            #alpha = param
            (alpha,rank_estimate) = param
            trainingData = createObservedTensor(trainingData)
            Xobs = X * trainingData  
            Xobs = X
            return comp.CompletionCP_EveryStep(Xobs,trainingData,rank_estimate,L,alpha,beta)
    elif method == "TuckerProd":
        def estimator(param,trainingData):
            #rank_estimate = param
            print param
            print len(trainingData) , " / " , elems
            (alpha,r)=param 
            rank_estimate = [r,r,r]
            print "alpha:",alpha
            print "rank_estimate",rank_estimate

            trainingData = createObservedTensor(trainingData)
            Xobs = X * trainingData
            Xobs = X
            return comp.CompletionTuckerProd_EveryStep(Xobs,trainingData,rank_estimate,L,alpha)
    elif method == "CPProd":
        def estimator(param,trainingData):
            print param
            print len(trainingData) , " / " , elems
            beta = 1e-8 #普通の正則化はとりあえず固定
            #alpha = param
            (alpha,rank_estimate) = param
            trainingData = createObservedTensor(trainingData)
            Xobs = X * trainingData  
            Xobs = X
            return comp.CompletionCPProd_EveryStep(Xobs,trainingData,rank_estimate,L,alpha,beta)
    elif method == "KSTucker":
        def estimator(param,trainingData):
            #rank_estimate = param
            print param
            print len(trainingData) , " / " , elems
            (alpha,r)=param 
            rank_estimate = [r,r,r]
            print "alpha:",alpha
            print "rank_estimate",rank_estimate

            trainingData = createObservedTensor(trainingData)
            Xobs = X * trainingData
            Xobs = X
            return comp.CompletionKS_Tucker_EveryStep(Xobs,trainingData,rank_estimate,L,alpha)
    elif method == "KSCP":
        def estimator(param, trainingData):
            print param
            print len(trainingData) , " / " , elems
            (alpha,rank_estimate) = param
            trainingData = createObservedTensor(trainingData)
            Xobs = X * trainingData  
            Xobs = X
            return comp.CompletionKS_CP_EveryStep(Xobs,trainingData,rank_estimate,L,alpha)
    elif method == "KPTucker":
        def estimator(param,trainingData):
            #rank_estimate = param
            print param
            print len(trainingData) , " / " , elems
            (alpha,r)=param 
            rank_estimate = [r,r,r]
            print "alpha:",alpha
            print "rank_estimate",rank_estimate

            trainingData = createObservedTensor(trainingData)
            Xobs = X * trainingData
            Xobs = X
            return comp.CompletionKP_Tucker_EveryStep(Xobs,trainingData,rank_estimate,L,alpha)
    elif method == "KPCP":
        def estimator(param, trainingData):
            print param
            print len(trainingData) , " / " , elems
            (alpha,rank_estimate) = param
            print "alpha:",alpha
            trainingData = createObservedTensor(trainingData)
            Xobs = X * trainingData  
            Xobs = X
            return comp.CompletionKP_CP_EveryStep(Xobs,trainingData,rank_estimate,L,alpha)

    

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
    for unobservedRate in unobservedRates:
        #log.WriteLine("unobserved rate, "+str(unobservedRate)+ " ")

        time = varianceTimes
        
        parameters = [(a,rank) for a in alpha for rank in ranks] 
        errors = CrossValidation(
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
            log.WriteLine("unobserved, "+ str(unobservedRate)+", param,"+str(param)+", error, "+str(e))

        log.WriteLine()

import forkmap

#評価。
#dataSeparator :: originalIndex -> (evaluationData, trainingData)
def CrossValidation(
        trainingData,
        estimator,
        lossFunction,
        evaluatingDataSeparator,
        hyperParameters,
        parameterOptimizingDataSeparator):
    """
    @param trainingData 訓練データのジェネレータ
    @param estimator ハイパーパラメータと訓練データを受け取って学習し、モデルを返す関数
    @param lossFunction 誤差関数。評価データとモデルを受け取り、誤差を返す。
    @param evaluatingDataSeparator 訓練データの一部を評価データとして分割するための関数
    @param hyperParameters 最適化を行うためのハイパーパラメータの集合をジェネレータで与える。
    @param parameterOptimizingDataSeparator 訓練データからハイパーパラメータ最適化用にデータを分割するための関数

    ハイパーパラメータ最適化をしながらクロスバリデーションを行う。
    """

    #Simplified CrossValidation

    global log

    import forkmap 

    @forkmap.parallelizable(6)
    def evaluateData(evalAndTrainIx,hyperParameters):
        (evalIx, trainIx) = evalAndTrainIx
        
        bestScore = float("inf")
        if not isinstance(hyperParameters,list):
            hyperParameters = [hyperParameters]

        doesOptimizeHyperParameter = len(hyperParameters) > 1

        alldata = arange(len(trainingData))

        assert(len(set(evalIx)&set(trainIx)) == 0)
        if(doesOptimizeHyperParameter):
            #ハイパーパラメータ最適化のループ
            for hyperParameter in hyperParameters:

                rawList = parameterOptimizingDataSeparator(trainIx)
                for (evalIxHP, trainIxHP) in rawList:

                    assert(len(set(evalIxHP)&set(evalIx)) == 0)
                    assert(len(set(trainIxHP) & set(evalIx)) == 0)

                    #print "Run LossFunction. hyperParam:", hyperParameter, 
                    estimation = estimator(hyperParameter,trainIxHP)
                    score = lossFunction(estimation,evalIxHP)
                    break #Simplified

                print "error:",score," at param=",hyperParameter
                if bestScore > score:
                    bestScore = score
                    bestParameter = hyperParameter
        else:
            bestParameter = hyperParameters[0]

        #最良のモデルを使う
        if doesOptimizeHyperParameter:
            print "Best parameter:",bestParameter
        #log.Write(", param, " + str(bestParameter))
        estimation = estimator(bestParameter,trainIx)

        score = lossFunction(estimation,alldata)
        print "Evaluation:",score
        return {"error":score,"param":bestParameter}
        #scores.append(score)
    
    from ThreadFunc import tmap
    #評価のループ。バリアンeスを作るループ
    #datastream = Toolbox.ToArray(evaluatingDataSeparator(arange(len(trainingData))))
    #datastream = Toolbox.ToArray(datastream)
    #datastream = Toolbox.ToArray(evaluatingDataSeparator(arange(len(trainingData))))
    datastream = evaluatingDataSeparator(arange(len(trainingData)))
    #datastream = Toolbox.ToArray(datastream)
    try:
        assert(False)
        print "Multi-Threading"
        def forkeval(data):
            return evaluateData(data,hyperParameters)
        result  = forkmap.map(forkeval,datastream)
        for r in result:
            yield r
    except Exception, e:
        print "Single Threading"
        for evalAndTrainIx in datastream:
            yield evaluateData(evalAndTrainIx,hyperParameters)
        #return map(lambda evalAndTrainIx:evaluateData(evalAndTrainIx,hyperParameters),datastream)


    #return scores
    
#EvaluateCompletion("CP",False)

#mat.Matlab.Open()

#mat.Matlab.Close()



#まだ使わない
def indecesOfEvaluationBlocks(targetIndeces,blocks):
    """
    [Obsolete]
    """
    #シーケンスをn個に分割する配列の大きさのリストを返す
    def getSeparatingArraySize(length,blocks):
        assert(length > blocks)
        offset = 0.0
        blocksize=[]

        lower = length / blocks
        uppers = length - blocks*lower
        for i in xrange(blocks - uppers):
            yield lower
        for i in xrange(uppers):
            yield lower+1
    #シーケンスをn個に分割する
    def separateArrayRandomly(length,blocks):
        seq = []
        index = 0
        for size in getSeparatingArraySize(length,blocks):
            seq.extend([index] * size)
            index += 1

        return Toolbox.ShuffleArray(seq)

    length = len(targetIndeces)

    blockNumbers = separateArrayRandomly(length,blocks)
    for block in xrange(blocks):
        evaluation = [targetIndeces[i] for i in xrange(length) if blockNumbers[i] == block]
        training = [targetIndeces[i] for i in xrange(length) if blockNumbers[i] != block]
        y
