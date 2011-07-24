# coding: utf-8

from logger import *
from Evaluation import *

#data = benchmark.Bonnie()
#data = benchmark.Wine_v6()
#data = benchmark.ThreeDNoseData()
data = benchmark.RandomSmallTensor()
dataname = "Random"

exectimes = 6
logger = Logger(dataname+"_Tucker_Normal")
EvaluateCompletion(data,"Tucker",False,exectimes,logger)
logger = Logger(dataname+"_Tucker_Relation")
EvaluateCompletion(data,"Tucker",True,exectimes,logger)

exectimes = 6
logger = Logger(dataname+"_CP_Normal")
EvaluateCompletion(data,"CP",False,exectimes,logger)
logger = Logger(dataname+"_CP_Relation")
EvaluateCompletion(data,"CP",True,exectimes,logger)
