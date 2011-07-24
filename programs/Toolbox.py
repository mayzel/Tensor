# coding: utf-8
from numpy import *
from numpy.linalg import *

import algorithm as alg

from logger import *

import random

#配列をシャッフルする。破壊的に書き換えます
def ShuffleArray(seq):

    def swap(arr,a,b):
        val = arr[a]
        arr[a] = arr[b]
        arr[b] = val

    N = len(seq)
    for i in xrange(N):
        a = random.randint(0,N - i - 1)
        swap(seq,i,i+a)

    return seq 

def Take(gen,n):
    i = 0
    for item in gen:
        yield item
        i += 1
        if i == n:
            break

def GenerateRandomSeparation(targetIndeces,size):
    length = len(targetIndeces)
    while True:
        mask = append(repeat(True,size),repeat(False,length-size))
        mask = ShuffleArray(mask)
        subsets = [targetIndeces[i] for i in xrange(len(mask)) if mask[i]]

        complement = [targetIndeces[i] for i in xrange(len(mask)) if not mask[i]]
        yield subsets,complement


def GenerateMaskingSparseTensor(size,rate):
	(n,m,l) = size
	total = n*m*l
	obsnum = int(total * rate)

	coords = set([])
	for ind in xrange(obsnum):
		i = random.randint(n)
		j = random.randint(m)
		k = random.randint(l)
		if not (i,j,k) in coords:
			coords.add((i,j,k))

	return array(list(coords))




def ToArray(gen):
    if isinstance(gen,list):
        return gen
    else:
        return [item for item in gen]


