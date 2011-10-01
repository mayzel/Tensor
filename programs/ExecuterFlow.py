# coding: utf-8

from logger import *

#sys.path.insert(0,"oldcodes")
import benchmark

import EvaluationCPWopt 
from info import *
#import Evaluation as EvaluationOld


def flushInfo(name):
    global information
    #----------------file output----------------
    import json
    import datetime
    name = "loggingTest"
    d = datetime.datetime.today()
    filename = "jsonlog/" + name + '%04d.%02d%02d.%02d%02d.%02d.log' % (d.year, d.month, d.day, d.hour, d.minute, d.second)
    filenamedat = "jsonlog/" + name + '%04d.%02d%02d.%02d%02d.%02d.dat' % (d.year, d.month, d.day, d.hour, d.minute, d.second)

    import pickle
    fd = file(filenamedat,"w") 
    pickle.dump(information,fd)
    fd.close()

    fh = file(filename,"w") 
    information = strInfo(information)
    sttr = json.dumps(information)
    print sttr
    fh.write(sttr)
    fh.close()

    resetInfo()



def evaluate(Mask,unobserved):
    prefix = Mask + "_" + dataname

    #method = "CPWOPT"
    #logger = Logger(prefix+"_"+method)
    #EvaluationCPWopt.EvaluateCompletion(data,Mask,method,True,exectimes,logger,unobserved,alpha,ranks)
    #flushInfo(method+" _with_")
    #EvaluationCPWopt.EvaluateCompletion(data,Mask,method,False,exectimes,logger,unobserved,alpha,ranks)
    #flushInfo(method+" _without_")

    method = "CP"
    logger = Logger(prefix+"_"+method)
    EvaluationCPWopt.EvaluateCompletion(data,Mask,method,True,exectimes,logger,unobserved,alpha,ranks)
    flushInfo(method+" _with_")
    EvaluationCPWopt.EvaluateCompletion(data,Mask,method,False,exectimes,logger,unobserved,alpha,ranks)
    flushInfo(method+" _without_")

    method = "CPProd"
    logger = Logger(prefix+"_"+method)
    EvaluationCPWopt.EvaluateCompletion(data,Mask,method,True,exectimes,logger,unobserved,alpha,ranks)
    flushInfo(method+" _with_")
    EvaluationCPWopt.EvaluateCompletion(data,Mask,method,False,exectimes,logger,unobserved,alpha,ranks)
    flushInfo(method+" _without_")


if __name__ == "__main__":
    #method = "CPWOPT"
    #method = "CP"
    #method = "CPProd"
    #method = "Tucker"
    #method = "TuckerProd"
    exectimes = 5
    ranks=[4]
    alpha =[pow(10,x) for x in [-4,-3,-2,-1]] #for L

    #----------------loading data----------------
    data = benchmark.Flow_Injection()
    dataname = "Flow_0930"


    masking = "Random"
    #masking = "Fiber"
    #masking = "Slice"
    evaluate("Random",[0.75,0.9,0.95,0.99])
    #evaluate("Random",[0.25,0.5,0.75,0.9])







