# coding: utf-8

from logger import *

import sys
sys.path.insert(0,"oldcodes")

import benchmark

import EvaluationCPWopt
import Evaluation as EvaluationOld



data = benchmark.Artificial()
#data = benchmark.Flow_Injection()

dataname = "Artificial0817"
dataname = "Flowl0817"

def evaluate(Mask,unobserved):
    prefix = Mask + "_" + dataname

    logger = Logger(prefix+"_CPWOPT")
    EvaluationCPWopt.EvaluateCompletion(data,Mask,"CPWOPT",True,exectimes,logger,unobserved,alpha,ranks)

    if False:
        logger = Logger(prefix+"_Subspace_Relation")
        EvaluationOld.EvaluateCompletion(data,Mask,"CP",True,exectimes,logger,unobserved,alpha,ranks)
        #EvaluationOld.EvaluateCompletion(data,Mask,"Tucker",True,exectimes,logger,unobserved,alpha,ranks)

        logger = Logger(prefix+"_Subspace_Normal")
        EvaluationOld.EvaluateCompletion(data,Mask,"CP",False,exectimes,logger,unobserved,alpha,ranks)
        #EvaluationOld.EvaluateCompletion(data,Mask,"Tucker",False,exectimes,logger,unobserved,alpha,ranks)




alpha =[pow(10,x) for x in [-5,-4,-3,-2,-1]] #for L
#alpha =[pow(10,x) for x in [-3]] #for L
exectimes = 10
ranks=[2]

#evaluate("Random",[0.75,0.9,0.95,0.99])
evaluate("Random",[0.95,0.99])
#evaluate("Random",[0.99])
#for i in xrange(10):
#    evaluate("Random",[0.75,0.9,0.95,0.99])
#    evaluate("Fiber"[0.5,0.75,0.9,0.95])
#    evaluate("Slice",[0.25,0.50,0.75,0.9])


