import forkmap



@forkmap.parallelizable(4)
def mmm(n):
    while True:
        a=0
        print n
        for i in xrange(1000000):
            a=i*i



forkmap.map(mmm,xrange(4))

