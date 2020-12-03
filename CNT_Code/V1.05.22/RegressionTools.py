#!/usr/bin/env python
# coding: utf-8
# version: 2020.03.27

from __future__ import print_function

from sklearn import ensemble
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.base import clone

from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error


from sklearn import datasets
from sklearn.linear_model import HuberRegressor, LinearRegression

from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from time import time

import pickle
def data_save(data,filename):
    f = open(filename,"wb") # .dat
    pickle.dump(data,f)
    f.close()
    
def data_load(filename):
    return pickle.load(open(filename,"rb"))


def scaled_rmse(y_true, y_pred):
    """
    Rescaled RMSE
    
    Parameters
        
        y_true: vector of labels
        y_pred: vector of predictions
    """
    mse = mean_squared_error(y_true, y_pred)
    return sqrt(mse / np.mean(y_true**2))


def score_model(model, data, metric):
    """
    Score model on data according to metric
    
    Parameters
        
        model: sklearn model instance
        data: tuple of inputs and labels
        metric: callable metric function metric(y_true, y_pred)
    """
    x, y_true = data
    y_pred = model.predict(x)
    return metric.__name__, metric(y_true, y_pred)


def train_and_evaluate(model,
                       train_data,
                       test_data,
                       metrics=[r2_score, mean_squared_error, scaled_rmse]):
    """
    Evaluates the train/test performance of model
    
    Parameters
        
        model: sklearn model instance
        train_data: tuple of training inputs and labels
        test_data: tuple of testing inputs and labels
        metrics: a list of callable metrics to evaluate
    """

    # Train model
    model.fit(*train_data)

    # Evaluate Model
    print(f'{"="*100} \n Evaluating: \n {model} \n{"="*100}')
    for m in metrics:
        m_name, m_value_train = score_model(model=model,
                                            data=train_data,
                                            metric=m)
        _, m_value_test = score_model(model=model, data=test_data, metric=m)
        print(
            f'{m_name:<20} {m_value_train:>10.3f} (train) {m_value_test:>10.3f} (test)'
        )

    # Plot data
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.scatterplot(
        x=train_data[1],
        y=model.predict(train_data[0]),
        ax=ax[0],
        alpha=0.5,
    )
    ax[0].set_title('Train')
    sns.scatterplot(
        x=test_data[1],
        y=model.predict(test_data[0]),
        ax=ax[1],
        alpha=0.5,
    )
    ax[1].set_title('Test')

    for a in ax:
        a.set_xlabel('y_true')
        a.set_ylabel('y_pred')
        a.plot(a.get_xlim(), a.get_xlim(), ls='--', c='k')
    
def train_and_evaluate_cv(model,
                       data,cv=3,
                       metrics=[r2_score, mean_squared_error, scaled_rmse],
                       ax=None):
    """
    Evaluates the train/test performance of model
    
    Parameters
        
        model: sklearn model instance
        data: tuple of inputs and labels
        cv: cv
        metrics: a list of callable metrics to evaluate
    """

    # Train model
    predicted = cross_val_predict(model, data[0], data[1], cv=cv)

    # Evaluate Model
    print(f'{"="*100} \n Evaluating: \n {model} \n{"="*100}')
    for m in metrics:          
        m_name = m.__name__
        m_value = m(data[1], predicted)
        print(
            f'{m_name:<20} {m_value:>10.3f} (cv = {cv})'
        )

    # Plot data
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.gca()
    sns.scatterplot(
        x=data[1],
        y=predicted,
        ax=ax,
        alpha=0.5,
    )
    ax.set_xlabel('measurement')
    ax.set_ylabel('predict')
    ax.set_title(model.__class__.__name__)
    ax.plot(ax.get_xlim(), ax.get_xlim(), ls='--', c='k')
          
    return predicted
          
def RSCV_GB(X,y,cv=5):
    param_dist = {"n_estimators":range(50,500,10),
                  "max_depth": [2,3, 4, 5,6,None],
                  'min_samples_split': [2,3,4,5]}

    params = {'learning_rate': 0.01, 'loss': 'ls'}
    grad_boost = ensemble.GradientBoostingRegressor(**params)
    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(grad_boost, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=cv, iid=False)
    start = time()
    random_search.fit(X, y)
    grad_boost.set_params(**random_search.best_params_);
    return grad_boost

from sklearn.neural_network import MLPRegressor
# mlp = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 10), random_state=10)
def RSCV_MLP(X,y):
    param_dist = {"hidden_layer_sizes":[(10,10),(15,15),(20,20),(25,25)],
                  "alpha": [1e-5,1e-4],
                  "max_iter":[200,300,400]
                 }

    model = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 10))
    
    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(model, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=5, iid=False)
    start = time()
    random_search.fit(X, y)
    model.set_params(**random_search.best_params_);
    return model
          
def RSCV_Huber(X,y,fit_intercept=True):
    param_dist = {"alpha":np.logspace(-6,3,10),
                  "epsilon": np.linspace(1.1,3,10),
                  'max_iter': [1000],
                 }

    model = HuberRegressor(fit_intercept=fit_intercept)
    # run randomized search
    n_iter_search = 100
    random_search = RandomizedSearchCV(model, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=3, iid=False)
    start = time()
    random_search.fit(X, y)
    model.set_params(**random_search.best_params_);
    return model
