#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, KFold


# basic metrics

from sklearn.metrics import mean_squared_error, r2_score

def scaled_rmse(y_true, y_pred):
    """
    Rescaled RMSE
    
    Parameters
        
        y_true: vector of labels
        y_pred: vector of predictions
    """
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse / np.mean(y_true**2))

# score-tools

def KL_Gaussian(y_true,sigma_true,y_pred,sigma_pred,eps=1e-6):
    ''' KL -- for one line
    '''
    sigma_pred = np.max([sigma_pred,eps])# epsilon
    sigma_true = np.max([sigma_true,eps])# epsilon
    sigma = sigma_true
    
    return 0.5*(sigma**2 + (y_true-y_pred)**2) / sigma_pred**2 - 0.5 - np.log(sigma/sigma_pred)

def KL_Gaussian_var(y_true,sigma_true,y_pred,sigma_pred,eps=1e-6):
    ''' min KL -- for one line
    '''
    sigma_pred = np.max([sigma_pred,eps])# epsilon
    sigma_true = np.max([sigma_true,eps])# epsilon
    
    sigma = np.min([sigma_pred,sigma_true]) # if sigma pred < true, good
    
    return 0.5*(sigma**2 + (y_true-y_pred)**2) / sigma_pred**2 - 0.5 - np.log(sigma/sigma_pred)

def KL_Gaussian_inv(y_true,sigma_true,y_pred,sigma_pred,eps=1e-6):
    ''' KL -- for one line
    '''
    sigma_pred = np.max([sigma_pred,eps])# epsilon
    sigma_true = np.max([sigma_true,eps])# epsilon
    sigma = sigma_true
    
    return 0.5*(sigma_pred**2 + (y_true-y_pred)**2) / sigma**2 - 0.5 - np.log(sigma_pred/sigma) # 2020.05.04

def KL_Gaussian_var_inv(y_true,sigma_true,y_pred,sigma_pred,eps=1e-6):
    ''' min KL -- for one line
    '''
    sigma_pred = np.max([sigma_pred,eps])# epsilon
    sigma_true = np.max([sigma_true,eps])# epsilon
    
    sigma = np.min([np.sqrt(sigma_pred**2 + (y_true-y_pred)**2),sigma_true]) # 
    return 0.5*(sigma_pred**2 + (y_true-y_pred)**2) / sigma**2 - 0.5 - np.log(sigma_pred/sigma) # 2020.05.04

def cross_val_score_gm(gm,data,data_supp=None,cv=5,
                       n_estimator=10,ratio=0.9,reuse=True,
                       n_predict=20,
                       metrics=[r2_score, mean_squared_error, scaled_rmse],
                       metrics_distr=[KL_Gaussian_inv,KL_Gaussian_var_inv,KL_Gaussian,KL_Gaussian_var], # by mean and std
                       return_prediction=False, # return data obj
                       **kwargs
                      ):
    '''gm : a graph model
    e.g. res = cross_val_score_gm(gm,data,reuse=True,cv=3,n_estimator=10)    
    
    '''
    
    assert not (n_predict<2 and len(metrics_distr)>0)
    
    idy = gm.idy
    data[idy+'_pred_mean'] = np.nan # initialize
    data[idy+'_pred_median'] = np.nan
    for md in metrics_distr:
        data[md.__name__] = np.nan
    if len(metrics_distr)>0:
        sigma = kwargs.pop('sigma',0.1) # default for cnt
    
    #from sklearn.model_selection import StratifiedKFold, KFold
    skf = StratifiedKFold(n_splits=cv,shuffle=False) # data has shuffled
    
    gm.fit(data,n_estimator=0,ratio=ratio,reuse=reuse) # fit the graph model
    
    groups = None
    if 'class' in data.columns:
        groups = data['class']
    groups = data['class'] # debug: make sure 'class' is used
    for train_index, test_index in skf.split(data, groups):
    
        index = data.index
        print('*',end=',')

        data_test = data.iloc[test_index,:]
        if data_supp is None:
            data_train = data.iloc[train_index,:]
        else:
            data_train = pd.concat([data_supp,data.iloc[train_index,:]])
        data_train = shuffle(data_train)

        gm.fit(data_train,n_estimator=n_estimator,ratio=ratio,reuse=True) # fit the graph model
        random = (n_estimator>1)
        
        idx = gm.renamer(gm.idx)
        data_pred1 = gm.predict(X=data_test[idx],random=False,n_predict=1,return_pd=True) 
        data_preds = gm.predict(X=data_test[idx],random=random,n_predict=n_predict,return_pd=True) 
        
        g = data_preds[idy].groupby(data_preds.index,sort=False)
        data.loc[test_index,idy+'_pred'] = data_pred1[idy]
        data.loc[test_index,idy+'_pred_mean'] = g.mean()
        data.loc[test_index,idy+'_pred_median'] = g.median()
        if n_predict>1:
            data.loc[test_index,idy+'_pred_std'] = g.std()
            data.loc[test_index,idy+'_pred_25'] = g.quantile(.25)
            data.loc[test_index,idy+'_pred_75'] = g.quantile(.75)
        
    print('.')
    for md in metrics_distr:
        data[md.__name__] = data.apply(lambda r: md(r[idy],sigma,r[idy+'_pred_mean'],r[idy+'_pred_std']),axis=1)
    for md in metrics_distr:
        df = data
        df[idy+'_pred_mean(p)'] = (df[idy+'_pred_75'] + df[idy+'_pred_25'])/2 # another estimate of mean and std
        df[idy+'_pred_std(p)'] = (df[idy+'_pred_75'] - df[idy+'_pred_25'])/2/0.675
        data[md.__name__+'(p)'] = df.apply(lambda r: md(r[idy],sigma,r[idy+'_pred_mean(p)'],r[idy+'_pred_std(p)']),axis=1)

    # scores and names
    metric_value = [m(data[idy],data[idy+'_pred']) for m in metrics]
    names = [m.__name__ for m in metrics]
    metric_value += [m(data[idy],data[idy+'_pred_mean']) for m in metrics]
    names += [m.__name__ + ' (mean pred)' for m in metrics]
    metric_value += [m(data[idy],data[idy+'_pred_median']) for m in metrics]
    names += [m.__name__ + ' (median pred)' for m in metrics]
    
    metric_value += [data[md.__name__].mean() for md in metrics_distr]
    names += [md.__name__ + ' (ps mean)' for md in metrics_distr]
    metric_value += [data[md.__name__].median() for md in metrics_distr]
    names += [md.__name__ + ' (ps median)' for md in metrics_distr]
    metric_value += [data[md.__name__+'(p)'].mean() for md in metrics_distr]
    names += [md.__name__ + '(p) (ps mean)' for md in metrics_distr]
    metric_value += [data[md.__name__+'(p)'].median() for md in metrics_distr]
    names += [md.__name__ + '(p) (ps median)' for md in metrics_distr]
    
    pd_metrics = pd.DataFrame(np.array(metric_value).reshape([1,-1]),columns=names)
    if return_prediction:
        return pd_metrics,data
    return pd_metrics