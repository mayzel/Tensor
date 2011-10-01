# coding: utf-8

from logger import *

#sys.path.insert(0,"oldcodes")
import benchmark

import EvaluationCPWopt 
from info import *
#import Evaluation as EvaluationOld


def evaluate(Mask,unobserved):
    prefix = Mask + "_" + dataname

    logger = Logger(prefix+"_CPWOPT")
    EvaluationCPWopt.EvaluateCompletion(data,Mask,method,True,exectimes,logger,unobserved,alpha,ranks)
    #EvaluationCPWopt.EvaluateCompletion(data,Mask,"CPWOPT",True,exectimes,logger,unobserved,alpha,ranks)

    if False:
        logger = Logger(prefix+"_Subspace_Relation")
        EvaluationOld.EvaluateCompletion(data,Mask,"CP",True,exectimes,logger,unobserved,alpha,ranks)
        #EvaluationOld.EvaluateCompletion(data,Mask,"Tucker",True,exectimes,logger,unobserved,alpha,ranks)

        logger = Logger(prefix+"_Subspace_Normal")
        EvaluationOld.EvaluateCompletion(data,Mask,"CP",False,exectimes,logger,unobserved,alpha,ranks)
        #EvaluationOld.EvaluateCompletion(data,Mask,"Tucker",False,exectimes,logger,unobserved,alpha,ranks)


if __name__ == "__main__":
    #global information
    #method = "CPWOPT"
    method = "CP"
    #method = "CPProd"
    #method = "Tucker"
    #method = "TuckerProd"
    exectimes = 5
    ranks=[2]
    alpha =[pow(10,x) for x in [-5,-4,-3,-2,-1]] #for L
    alpha =[pow(10,x) for x in [-3]] #for L

    #----------------loading data----------------
    data = benchmark.Artificial()
    #data = benchmark.Flow_Injection()

    dataname = "Artificial0926"

    masking = "Random"
    #masking = "Fiber"
    #masking = "Slice"
    evaluate("Random",[0.9])


    #----------------file output----------------
    import json
    import datetime
    name = "loggingTest"
    d = datetime.datetime.today()
    filename = "jsonlog/" + name + '%04d.%02d%02d.%02d%02d.%02d.log' % (d.year, d.month, d.day, d.hour, d.minute, d.second)

    fh = file(filename,"w") 
    information = strInfo(information)
    sttr = json.dumps(information)
    print sttr
    fh.write(sttr)
    fh.close()







