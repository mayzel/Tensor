# coding: utf-8

from logger import *

#sys.path.insert(0,"oldcodes")
import benchmark

import EvaluationCPWopt 
#from info import *
#import Evaluation as EvaluationOld


def flushInfo(information,name):
    #----------------file output----------------
    import json
    import datetime
    d = datetime.datetime.today()
    filename = "jsonlog/" + name + '%04d.%02d%02d.%02d%02d.%02d.log' % (d.year, d.month, d.day, d.hour, d.minute, d.second)
    filenamedat = "jsonlog/" + name + '%04d.%02d%02d.%02d%02d.%02d.dat' % (d.year, d.month, d.day, d.hour, d.minute, d.second)

    import pickle
    fd = file(filenamedat,"w") 
    pickle.dump(information,fd)
    fd.close()

    #fh = file(filename,"w") 
    #information = strInfo(information)
    #sttr = json.dumps(information)
    #print sttr
    #fh.write(sttr)
    #fh.close()

    #resetInfo()
    information={}



def evaluateMethod(data,Mask,method,withRelation,exectimes,logger,unobserved,alpha,ranks):
    information = {}
    information["dataname"] = dataname
    information["datasize"] = data["X"].shape

    information = EvaluationCPWopt.EvaluateCompletion(data,Mask,method,withRelation,exectimes,logger,information,unobserved,alpha,ranks)
    print information
    if withRelation:
        flushInfo(information, dataname + "_" + method+"_with_")
    else:
        flushInfo(information, dataname + "_" + method+"_without_")


def evaluate(Mask,unobserved):
    prefix = Mask + "_" + dataname

    method = "CPWOPTProd"
    logger = Logger(prefix+"_"+method)
    evaluateMethod(data,Mask,method,True,exectimes,logger,unobserved,alpha,ranks)

    method = "CPWOPT"
    logger = Logger(prefix+"_"+method)
    evaluateMethod(data,Mask,method,True,exectimes,logger,unobserved,alpha,ranks)
    evaluateMethod(data,Mask,method,False,exectimes,logger,unobserved,alpha,ranks)

    #method = "CP"
    #logger = Logger(prefix+"_"+method)
    #evaluateMethod(data,Mask,method,True,exectimes,logger,unobserved,alpha,ranks)

    #method = "CPProd"
    #logger = Logger(prefix+"_"+method)
    #evaluateMethod(data,Mask,method,True,exectimes,logger,unobserved,alpha,ranks)


if __name__ == "__main__":
    #method = "CPWOPT"
    #method = "CP"
    #method = "CPProd"
    #method = "Tucker"
    #method = "TuckerProd"
    #alpha =[pow(10,x) for x in [-3]] #for L

    #----------------loading data----------------
    data = benchmark.Flow_Injection()
    dataname = "Flow"
    ranks=[4]
    alpha =[pow(10,x) for x in [-4,-3,-2,-1]] #for L
    #data = benchmark.Artificial()
    #dataname = "Artificial_1001"
    #ranks=[2]
    #alpha =[pow(10,x) for x in [-4,-3,-2,-1]] #for L

    exectimes = 5

    masking = "Random"
    #masking = "Fiber"
    #masking = "Slice"
    #evaluate("Random",[0.9])
    evaluate("Random",[0.75,0.9,0.95,0.99])







