
class Completion:
    def estimator(param, trainingData):
        pass
    def lossFunction():
        pass


class CPWOPT(Completion):
    def estimator(param,trainingData):
        #trainingData must be given as list of coordinate tuples
        #rank_estimate = param
        print param
        print len(trainingData) , " / " , elems
        (alpha,r)=param 
        rank_estimate = r
        print "alpha:",alpha
        print "rank_estimate",rank_estimate

        n,m,l=shape

        def getobslist(trainingData):
            for index in trainingData:
                i=index / (m*l)
                index = index - i*m*l
                j=index/l
                index = index - j*l
                k=index
                yield i;yield j;yield k
        
        #ObservedList = array(reduce(lambda a,b:a+b,[coord(ind) for ind in trainingData]))
        ObservedList = array(list(getobslist(trainingData)))
        Xobs = CPWOPT.CompressSparseTensorToVector(X,ObservedList)
        print "LEN",Xobs.shape
        print "Xobs ",type(Xobs)
        print "shape ",type(shape)
        print "trainingData ",type(trainingData)

        result = CPWOPT.CompletionGradient(Xobs,shape, ObservedList,rank_estimate,L,alpha,X)
        return result


