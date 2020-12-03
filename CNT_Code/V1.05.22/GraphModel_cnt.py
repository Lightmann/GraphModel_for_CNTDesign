#!/usr/bin/env python
# coding: utf-8
# version: 2020.03.22


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# from GeneralTools import *
from RegressionTools import *
from GraphTools import *
from GraphModel import *
# from CNT_Dataset import *


# Graph model

class GraphModel_cnt(GraphModel):
    """
    digraph based model.
    
    """
    
    def __init__(self,**kwargs):
        '''
        G : a `networkx` graph
            after fit, each non-input node will have attr 'model' and 'estimators'
        idx : input of the graph
        idy : output of the graph (one node only)
        renamer : if the input node is a vector, assin the vector names (used in pandas)
        assign_submodel : a function to give submodel in the graph
                        default is linear model
                        
        gb_estimators=200, -- in Gradient-Boosting
        n_iter_search=20,submodel_cv=5 -- in CV search
        '''
        
        self.cv = kwargs.pop("submodel_cv", 5)
        self.kwargs = kwargs
        
        super(GraphModel_cnt, self).__init__(**kwargs)
        

    def _assign_submodel(self,data,idx,idy,*args):
        return assign_submodel_cnt(data,idx,idy,
                                   cv=self.cv,
                                   **self.kwargs
                                  )

# 

from sklearn.model_selection import cross_val_score

def assign_submodel_cnt(data,idx,idy,cv=5,**kwargs):
    '''
    assign submodel for graph model

    Parameters
    ----------
    data : pandas df
        use to train submodel.
    idx : list
        node list of a graph.
    idy : string
        output node of a graph.
    cv : cross-validation, optional
        The default is 5.
    **kwargs : other parameters for model selection
        for example, gb_estimators=200,n_iter_search=20.

    Returns
    -------
    TYPE
        a regression model such as model = LinearRegression().

    '''
    
    if idx == ['T','S'] or idx==['S','T']: # (T,S)->Y, linear is the ground truth
        model = LinearRegression()
        return model

    data_all = data.dropna(axis=0,how='any',subset=idx+[idy]) # use valid data
    data = shuffle(data_all)
    X,y = data[idx],data[idy]
    groups = None
    if 'class' in data.columns:
        groups = data['class']
    
    model_array = [LinearRegression(),
                   RSCV_Huber_cnt(X,y,groups,cv=cv,**kwargs),
                   RSCV_GB_cnt(X,y,groups,cv=cv,**kwargs),
                  ]
    
    score_array = [cross_val_score(model,X,y,groups,cv=cv).mean() for model in model_array]
    print(idx,idy,score_array) # for debug
    
    return model_array[np.argmax(score_array)]


def RSCV_GB_cnt(X,y,groups=None,cv=5,gb_estimators=200,n_iter_search=20):
    '''
    Random search cross-validation GradientBoostingRegressor -- for CNT-dataset

    Parameters
    ----------
    X : numpy
        inputs of a model.
    y : numpy
        target of a model.
    groups : TYPE, optional
        DESCRIPTION. The default is None.
    cv : TYPE, optional
        DESCRIPTION. The default is 5.
    gb_estimators : TYPE, optional
        DESCRIPTION. The default is 200.
    n_iter_search : TYPE, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    a model = GradientBoostingRegressor(...)

    '''
    param_dist = {"n_estimators":np.arange(40,gb_estimators+1,40).astype(int),
                  "max_depth": [2,3,4,5,6,None],
                  'min_samples_split': [2,3,4,5]}

    params = {'learning_rate': 0.01, 'loss': 'ls'} # 'huber
    grad_boost = ensemble.GradientBoostingRegressor(**params)
    # run randomized search
    random_search = RandomizedSearchCV(grad_boost, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=cv, iid=False)
    random_search.fit(X, y,groups=groups)
    grad_boost.set_params(**random_search.best_params_);
    return grad_boost

def RSCV_Huber_cnt(X,y,groups=None,cv=5,n_iter_search=20):
    try:
        return _RSCV_Huber_cnt(X,y,groups,cv,n_iter_search,tol=1e-5)
    except:
        try:
            return _RSCV_Huber_cnt(X,y,groups,cv,n_iter_search,tol=1e-4)
        except:
            return _RSCV_Huber_cnt(X,y,groups,cv,n_iter_search,tol=1e-3)

def _RSCV_Huber_cnt(X,y,groups=None,cv=5,n_iter_search=20,tol=1e-5):
    param_dist = {"alpha":np.logspace(-6,0,7),
                  "epsilon": np.linspace(1.1,3.1,11),
                  'max_iter': [1000],
                  # 'tol':[1e-05], # default, but may have convergence problem
                  'tol':[1e-04],
                  #'warm_start': [False,True]
                 }

    model = HuberRegressor(fit_intercept=True)
    # run randomized search
    random_search = RandomizedSearchCV(model, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=cv, iid=False)

    random_search.fit(X, y,groups=groups)
    model.set_params(**random_search.best_params_);
    return model


