# coding: utf-8

import sys
sys.path.insert(0,"oldcodes")

from logger import *
import benchmark

from EvaluationCPWopt import *

#data = benchmark.Artificial()
data = benchmark.Bonnie()

dataname = "Artificial0302"

def evaluate(Mask,unobserved):
    prefix = Mask + "_" + dataname

    logger = Logger(prefix+"_CPWOPT")
    EvaluateCompletion(data,Mask,"CPWOPT",True,exectimes,logger,unobserved,alpha,ranks)

    #logger = Logger(prefix+"_Subspace_Relation")
    #EvaluateCompletion(data,Mask,"CP",True,exectimes,logger,unobserved,alpha,ranks)
    #EvaluateCompletion(data,Mask,"Tucker",True,exectimes,logger,unobserved,alpha,ranks)

    #logger = Logger(prefix+"_Subspace_Normal")
    #EvaluateCompletion(data,Mask,"CP",False,exectimes,logger,unobserved,alpha,ranks)
    #EvaluateCompletion(data,Mask,"Tucker",False,exectimes,logger,unobserved,alpha,ranks)


alpha =[pow(10,x) for x in [-5,-4,-3,-2,-1]] #for L
alpha =[pow(10,x) for x in [-3]] #for L
exectimes = 2
ranks=[2]

evaluate("Random",[0.99])
#for i in xrange(10):
#    evaluate("Random",[0.75,0.9,0.95,0.99])
#    evaluate("Fiber"[0.5,0.75,0.9,0.95])
#    evaluate("Slice",[0.25,0.50,0.75,0.9])


