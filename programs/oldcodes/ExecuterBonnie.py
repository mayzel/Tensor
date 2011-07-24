# coding: utf-8

from logger import *
from Evaluation import *

#data = benchmark.Wine_v6()
#data = benchmark.ThreeDNoseData()
#data = benchmark.RandomSmallTensor()
#data = benchmark.Sugar()
data = benchmark.Bonnie()
dataname = "Bonnie0128"

def evaluate(Mask,unobserved):
    prefix = Mask + "_" + dataname

    logger = Logger(prefix+"_Subspace_Prod")
    ranks = [8]
    EvaluateCompletion(data,Mask,"CPProd",True,exectimes,logger,unobserved,alpha,ranks)
    ranks = [15]
    EvaluateCompletion(data,Mask,"TuckerProd",True,exectimes,logger,unobserved,alpha,ranks)

    logger = Logger(prefix+"_Subspace_Relation")
    ranks = [8]
    EvaluateCompletion(data,Mask,"CP",True,exectimes,logger,unobserved,alpha,ranks)
    ranks = [15]
    EvaluateCompletion(data,Mask,"Tucker",True,exectimes,logger,unobserved,alpha,ranks)

    logger = Logger(prefix+"_Subspace_Normal")
    ranks = [8]
    EvaluateCompletion(data,Mask,"CP",False,exectimes,logger,unobserved,alpha,ranks)
    ranks = [15]
    EvaluateCompletion(data,Mask,"Tucker",False,exectimes,logger,unobserved,alpha,ranks)


alpha =[pow(10,x) for x in [-3,-2,-1]] #for L
exectimes = 1
while True:
    evaluate("Random",[0.99])
    #evaluate("Random",[0.75,0.9,0.95,0.99])
    #evaluate("Fiber",[0.5,0.75,0.9,0.95])
    #evaluate("Slice",[0.25,0.50,0.75,0.9])
