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

import DataStream
import benchmark
import gc
import functools


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


