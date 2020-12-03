#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append(".")

import pickle
def data_save(data,filename):
    f = open(filename,"wb") # .dat
    pickle.dump(data,f)
    f.close()
    
def data_load(filename):
    return pickle.load(open(filename,"rb")) 

# Tools
def PowerSetsBinary(items): 
    '''generate all combination of N items -- all 2^N
    e.g. PowerSetsBinary('ABC'), PowerSetsBinary(['A1','B2','C4'])
    '''
    N = len(items) 
    #enumerate the 2**N possible combinations 
    for i in range(2**N): 
        combo = [] 
        for j in range(N): 
          #test jth bit of integer i 
          if(i >> j ) % 2 == 1: 
            combo.append(items[j]) 
        yield combo