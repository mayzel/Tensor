#coding: utf-8
information={}

def resetInfo():
    information = {}
def strInfo(info):
    result = {}
    for k,v in info.items():
        newv = None
        if type(v) == dict:
            newv = strInfo(v)
        elif type(v) == tuple or type(v) == list:
            newv=""
            for vi in v:
                newv += str(vi) + ", "
        else:
            newv = str(v)

        #newv = v

        newk = str(k)
        result[newk] = newv

    return result
