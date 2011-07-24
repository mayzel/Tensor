
#coding: utf-8

from numpy import *

import itertools as it

import Toolbox
import const

class StreamMemorizer:
    pass


#観測された場所をランダムに設定、無限に返す。　１：観測　０：未観測
def RandomObservedTensorStream(shape,unobservedRate):

    #ランダムなマスクをひとつ生成。
    def createMask(shape,unobservedRate):
        elems = prod(shape)
        maskedElems = int(elems * unobservedRate)
        mask = []
        mask.extend(repeat(0,maskedElems))
        mask.extend(repeat(1,elems - maskedElems))

        return array(Toolbox.ShuffleArray(mask)).reshape(shape)


    #無限生成・the Random Mask
    while True:
        yield createMask(shape,unobservedRate)





