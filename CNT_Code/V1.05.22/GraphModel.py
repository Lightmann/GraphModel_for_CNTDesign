#!/usr/bin/env python
# coding: utf-8
# version: 2020.03.27


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# from GeneralTools import *
from RegressionTools import *
from GraphTools import *
# from CNT_Dataset import *


# Graph model

class GraphModel(object):
    """
    digraph based model.
    
    """
    
    def __init__(self, G=None, idx=None, idy='Y',
                 renamer=None,
                 assign_submodel=None,
                 **kwargs
                ):
        '''
        G : a `networkx` graph
            after fit, each non-input node will have attr 'model' and 'estimators'
        idx : input of the graph
        idy : output of the graph (one node only)
        renamer : if the input node is a vector, assin the vector names (used in pandas)
        assign_submodel : a function to give submodel in the graph
                        default is linear model
        '''
        self.G = G
        if idx is None:
            idx = [n for n in G if len([m for m in G.predecessors(n)])==0] # inputs node
        self.idx = idx
        self.idy = idy
        self.renamer = renamer
        self.assign_submodel = assign_submodel
        self._xy_array = None
        
    @property
    def xy_array(self):
        '''
        the list of submodels in the graph model
        xy_array like [[['X'], 'B'], [['X'], 'A'], [['A', 'B'], 'Y']]
        '''
        
        if self._xy_array is None:
            self._xy_array = nx_get_task_list(self.G,target=self.idy,renamer=self.renamer,sort=True)
            
        return self._xy_array

    def _assign_submodel(self,data,idx,idy,*args):
        return LinearRegression()
        
    def fit(self,data,n_estimator=10,ratio=0.9,reuse=False):
        '''
        fit a number of models with randomly chosen training data
        n_estimator: number of models
        ratio: chosen data size
        reuse : if the submodel exists, reuse it
        
        will use self.xy_array like [[['X'], 'B'], [['X'], 'A'], [['A', 'B'], 'Y']]
        after fit, the node in graph will have 'model' and 'estimators'
        
        '''
    
        G = self.G
        xy_array = self.xy_array
        
        # copy from old code: def get_model_list(data,xy_array):
        data_back = data
        for idx,idy in xy_array: # main idea: use hole data to find the 'best' model, then use subdatasets to get many estimators
            data_all = data_back.dropna(axis=0,how='any',subset=idx+[idy]) # use the valid data
            data = shuffle(data_all) # shuffle
            ni = int(len(data)*ratio) # num of training data
            
            if not 'model' in G.nodes[idy].keys():
                G.nodes[idy]['model'] = None # default value
    
            if (G.nodes[idy]['model'] is None) or (not reuse): # set the model
                if self.assign_submodel is None: # use default model
                    model = self._assign_submodel(data,idx,idy)
                else:
                    model = self.assign_submodel(data,idx,idy) # e.g. CV-search

                G.nodes[idy]['model'] = model # set the model
                
            G.nodes[idy]['model'].fit(data[idx],data[idy]) # reuse but fit -- usefull for cross-validation
            
            assert ni>len(idx) # test
            estimators = []
            for _ in range(n_estimator): # different model weights <-- use different training data
                modeli = clone(G.nodes[idy]['model'])
                data = shuffle(data_all).iloc[:ni,:]  # use part data
                modeli.fit(data[idx],data[idy])
                estimators.append(modeli)
            G.nodes[idy]['estimators'] = estimators
    
    def predict(self,
                X=None,       # numpy obj
                data=None,    # pandas obj -- suggested
                n_predict=10, # predict 10 times
                random=True,  # randomly chosen the estimator
                ensemble=False,# ensemle the predict by randomly chosen estimators
                return_pd=True # return a pandas obj
               ):
        '''
        predict by graph model. 
        n_predict: times of call
        '''
        
        G = self.G
        xy_array = self.xy_array # like [[['X'], 'B'], [['X'], 'A'], [['A', 'B'], 'Y']]
        
        if data is None:
            data = pd.DataFrame(data=X,columns=self.renamer(self.idx))
        data_back = data
        data_array = []
        # choose the estimators for each call
        n_estimator = len(G.nodes[self.idy]['estimators']) # assume all has the same num of estimators
        tmp_order = np.random.choice(n_estimator,n_predict,replace=(n_predict > n_estimator))
        for call in range(n_predict):
            data = data_back.copy()
            data['call'] = call+1
            for idx,idy in xy_array:
                if random:
                    estimators = G.nodes[idy]['estimators']
                    model = estimators[tmp_order[call]]
                else:
                    model = G.nodes[idy]['model']
                data[idy] = model.predict(data[idx])
            data_array.append(data)
            
        data = pd.concat(data_array)
        if return_pd:
            return data
        else:
            return data[self.idy].values
    
    def predict_onebyone(self,X=None,data=None,**kwargs):
        '''
        predict by graph model. see predict(...)
        '''
        
        G = self.G
        xy_array = self.xy_array # like [[['X'], 'B'], [['X'], 'A'], [['A', 'B'], 'Y']]
        
        if data is None:
            data = pd.DataFrame(data=X,columns=self.renamer(self.idx))
            
        kwargs.update({'return_pd':True})
        data_array = []
        for i in range(len(data)):
            data_array.append(self.predict(data=data.iloc[i:i+1,:], **kwargs))
            
        return pd.concat(data_array)
    
    def evaluate(self,X,y):
        return

