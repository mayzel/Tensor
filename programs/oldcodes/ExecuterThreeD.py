# coding: utf-8

from logger import *
from Evaluation import *

#data = benchmark.Bonnie()
#data = benchmark.Wine_v6()
data = benchmark.ThreeDNoseData()
#data = benchmark.RandomSmallTensor()
dataname = "ThreeD0128"

unobservedRates = array([0.75,0.9,0.95,0.99])
alpha =[pow(10,x) for x in [-3,-2,-1,0]] #for L
ranks = [3]

exectimes = 1
while True:
    logger = Logger(dataname+"_Distance")
    EvaluateCompletion(data,"DistanceTucker",True,exectimes,logger,unobservedRates,alpha,ranks)
    EvaluateCompletion(data,"DistanceCP",True,exectimes,logger,unobservedRates,alpha,ranks)



        #EvaluateCompletion(data,"DistanceCP",True,exectimes,logger,unobservedRates,alpha,ranks)
        #logger = Logger(dataname+"_Subspace_Normal")
        #EvaluateCompletion(data,"CP",False,exectimes,logger,unobservedRates,alpha,ranks)
        #EvaluateCompletion(data,"Tucker",False,exectimes,logger,unobservedRates,alpha,ranks)
        #logger = Logger(dataname+"_Subspace_Relation")
        #EvaluateCompletion(data,"CP",True,exectimes,logger,unobservedRates,alpha,ranks)
        #EvaluateCompletion(data,"Tucker",True,exectimes,logger,unobservedRates,alpha,ranks)
        #logger = Logger(dataname+"_KSum")
        #EvaluateCompletion(data,"KSCP",True,exectimes,logger,unobservedRates,alpha,ranks)
        #EvaluateCompletion(data,"KSTucker",True,exectimes,logger,unobservedRates,alpha,ranks)
