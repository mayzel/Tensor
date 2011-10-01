import threading
import Queue

__all__ = ['tmap','treduce','tfilter', 'parallel','parallel']

class tmap:
    def __init__(self, func=None, data=None, numthreads=2):
        if not numthreads > 0:
            raise AssertionError("numthreads should be greater than 0")

        if func:
            self.handle_input=func
        if data:
            self.get_input = lambda : data

        self._numthreads=numthreads
        self.threads = []
        self.run()

    def __iter__(self):
        return self

    def next(self):
        still_running, input, output = self.DQ.get()
        if not still_running:
            raise StopIteration

        return output

    def get_input(self):
        raise NotImplementedError, "You must implement get_input as a function that returns an iterable"

    def handle_input(self, input):
        raise NotImplementedError, "You must implement handle_input as a function that returns anything"

    def _handle_input(self):
        while 1:
            work_todo, input = self.Q.get()
            if not work_todo:
                break                       

            self.DQ.put((True, input, self.handle_input(input)))

    def cleanup(self):
        """wait for all threads to stop and tell the main iter to stop"""
        for t in self.threads:
            t.join()
        self.DQ.put((False,None,None))

    def run(self):
        self.Q=Queue.Queue()
        self.DQ=Queue.Queue()
        for x in range(self._numthreads):
            t=threading.Thread(target=self._handle_input)
            t.start()
            self.threads.append(t)

        try:
            for x in self.get_input():
                self.Q.put((True, x))
        except NotImplementedError, e:
            print e
        for x in range(self._numthreads):
            self.Q.put((False, None))

        threading.Thread(target=self.cleanup).start()

class tfilter(tmap):
    def next(self):
        while 1:
            still_running, input, output = self.DQ.get()
            if not still_running:
                raise StopIteration

            if output == True:
                return input
        
class _treduce(tmap):
    def __init__(self, func=None, data=None, numthreads=2):
        if not numthreads > 0:
            raise AssertionError("numthreads should be greater than 0")

        if func:
            self.handle_input=func
        if data:
            self.get_input = lambda : data

        self._numthreads=numthreads
        self.threads = []

        self.mutex = threading.Lock()
        self.inputFinished=False
        self.run()

    def _handle_input(self):
        while 1:
            self.mutex.acquire()
            if self.inputFinished and self.Q.qsize() <= 1:
                self.mutex.release()
                break
            a,b = self.Q.get(),self.Q.get()
            self.mutex.release()
            self.Q.put(self.handle_input(a,b))

    def cleanup(self):
        """wait for all threads to stop and tell the main iter to stop"""
        for t in self.threads:
            t.join()

    def run(self):
        self.Q=Queue.Queue()
        for x in range(self._numthreads):
            t=threading.Thread(target=self._handle_input)
            t.start()
            self.threads.append(t)
        try:
            for a in self.get_input():
                self.Q.put(a)
        except NotImplementedError, e:
            print e
        self.inputFinished=True
        self.cleanup()

def treduce(func, biglist, numthreads=5):
    return _treduce(func, biglist, numthreads).Q.get()

class _parallel:
    def __init__(self, *settings):
        self.results = {}
        self.threads = []
        self.settings = settings
        self.run()

    def cleanup(self):
        for t in self.threads:
            t.join()
        self.results = map(lambda r: r[1], sorted(self.results.iteritems()))

    def run(self):
        for i, (func, args, kwargs) in enumerate(self.settings):
            t=threading.Thread(target=self._process_func, args=[i,func]+args, kwargs=kwargs)
            t.start()
            self.threads.append(t)
        self.cleanup()
        
    def _process_func(self,*args,**kwargs):
        i, func, args = args[0], args[1], args[2:]
        self.results[i]=func(*args,**kwargs)

def parallel(*settings):
    return _parallel(*settings).results

import time,random

def test_tmap(numThreads):
    def double(x):
        time.sleep(random.random()) #something time-consuming procedure
        return x*2
    return tmap(double,range(1,32),numThreads)

def test_treduce(numThreads):
    def add(x,y):
        time.sleep(random.random()) #something time-consuming procedure
        return x+y
    return treduce(add,range(1,32),numThreads)

def test_tfilter(numThreads):
    def prime(n):
        time.sleep(random.random()) #something time-consuming procedure
        return n % 2 == 0
    return tfilter(prime, range(1,32),numThreads)

def test_parallel():
    def hoge(a): 
        time.sleep(random.random()) #something time-consuming procedure #1
        return  a**2
    def fuga(a): #something time-consuming procedure #2
        time.sleep(random.random())
        return a**5
    return parallel((hoge,[5],{}),(fuga,[2],{}))


def mapreducetest():
    import google
    def search(args):
        return google.doGoogleSearch(*args).results

    def aggregate(a,b):
        a.extend(b)
        return a
    
    return treduce(aggregate, tmap(search, (('hoge',0),('hoge',10))))

def test():
    numThreadsSetting = (1,2,4,16,32,64,128)
    print '### testing tmap() ###'
    for n in numThreadsSetting:
        start = time.time()
        print 'result:',
        for r in test_tmap(n):
            print r,
        print
        print n, 'thread(s),', time.time() - start, 'sec'
        print

    print '### testing treduce() ###'
    for n in numThreadsSetting:
        start = time.time()
        print 'result:',
        print test_treduce(n)
        print n, 'thread(s),', time.time() - start, 'sec'
        print

    print '### testing tfilter() ###'
    for n in numThreadsSetting:
        start = time.time()
        print 'result:',
        for r in test_tfilter(n):
            print r,
        print
        print n, 'thread(s),', time.time() - start, 'sec'
        print

    print '### testing parallel() ###'
    print test_parallel()

if __name__ == '__main__':
    test()

