#!/usr/bin/env python
# coding: utf-8
# 2020-07-01

import argparse
# Params
parser = argparse.ArgumentParser(description='BO')
# parser.add_argument('--model', default='DnCNN_wBeta_woMean_BF', type=str, help='choose a type of model')
# parser.add_argument('--batch_size', default=128, type=int, help='batch size')
# parser.add_argument('--train_data', default='Train400', type=str, help='path of train data')
# parser.add_argument('--sigma', default=[0,80], help='noise level')
# parser.add_argument('--epoch', default=180, type=int, help='number of train epoches')
# parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
# parser.add_argument('--gpu_idx', default=0, type=float, help='gpu idx')

parser.add_argument('--itag', default=0, type=int, help='itag')
parser.add_argument('--n_start', default=20, type=int, help='number of start points')


#
args = parser.parse_args()

itag = args.itag
n_start = args.n_start

# itag = 0
# n_start = 20





# In[2]:

from GPyOpt.methods import BayesianOptimization
import os
import time as tm
from openpyxl import load_workbook


# ## Design the call function (basic)
# 
# Aim: given a batch of x=content, fill the properties by graph models.


import sys
for path in ['../','../../','../../../','../../../../']:
    sys.path.append(path+"CNT_Code/V1.05.22")
    
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

random.seed(itag)
np.random.seed(itag)

from RegressionTools import *
from GraphTools import *
from GeneralTools import *
# from GraphModel import *
from GraphModel_cnt import *
from CNT_Dataset import *
from ModelSelection import *


# In[5]:


# data_org = pd.read_excel('data_all_back_A.xlsx') # 2020.06.06 use data back -- no log(*)
# graph_array = data_load('graph_array_368.dat')
gm_array = data_load('gm_array.dat')

# gm = data_load('gm.dat')


idx = ['P3HT vol (µL)','D1 vol (µL)', 'D2 vol (µL)', 'D6 vol (µL)', 'D8 vol (µL)',]
# idy = 'Conductivity (modified predict)'
idy = 'Y'; targy = 'Cond'
idz = 'R'; targz = 'Abs'

id_targ = idy; this_targ = targy
# id_targ = idz; this_targ = targz


# In[9]:


def _call(x):
    res_array = []
    for i in range(len(x)):
        gm = np.random.choice(gm_array)
        data_x = pd.DataFrame(x[i:i+1],columns=idx)
        for s in ['A','R','S','T','Y']: data_x[s] = np.nan;
        res = gm.predict(data=data_x,n_predict=1,ensemble=False,random=True)
        res['predicted by graph'] = nx2str(gm.G)
        res_array.append(res)
    return pd.concat(res_array)


# ## ExperimentGroundexperiment


# use the "_call" function to do artifical experiments
class ExperimentGroundexperiment():
    
    def __init__(self,target=None,tag = 'demo',filename=None):
        
        self.columns = idx + ['A','R','S','T','Y'] + ['call','predicted by graph','run','target']
        self.target = target
        
        
        self.data = pd.DataFrame(columns=self.columns)
        self.data_history = []
        self.run = 0
        
        self.filename = 'Artificial_BO_%s_history_%s.xlsx'%(tag,self.target)
        
    def Call(self,x): # x is a numpy instance
        
        data_x = pd.DataFrame(x,columns=idx)
        res = _call(data_x) # res is a pandas
        self.update_data(x,res) # update the pandas data history
        self.save() # save the new call
        #print(self.data)

        if self.target is targy:
            return res[idy].values
        if self.target is targz:
            return res[idz].values
        
    def update_data(self,x,res): # if res is a pandas
        
        res['run'] = self.run
        res['target'] = self.target
        self.run += 1
        self.data = self.data.append(res)
        
    def save(self): # save the call history to xlsx
        
        self.data_history.append(self.data.copy())
        sheet_name='%d'%(self.run)
        
        from openpyxl import load_workbook
        
        with pd.ExcelWriter(self.filename) as writer:
            for i,data in enumerate(self.data_history):
                data.to_excel(writer,sheet_name='%d'%(i))


# ## BO

# ### Configurations for artificial experiments

# In[13]:


# itag = 0
# n_start = 20
experiment = ExperimentGroundexperiment(target=this_targ,
                                        tag="20200701_%dT%d"%(n_start,itag))


# ### Generater initial dataset
# 
# There are many sample strategies. We use the `random_sample_content`

# In[14]:


# here we use this one: uniformly sample from simplex
def random_sample_content(n_start=100):
    x = 100 * np.hstack([np.zeros([n_start,1]),np.random.rand(n_start,4),np.ones([n_start,1])])
    x.sort(axis=1)
    x = x[:,1:] - x[:,:-1]
    # x.round(1)
    return x


x = random_sample_content(n_start)
# experiment = ExperimentGroundexperiment(...)
y = experiment.Call(x).reshape(-1,1)
experiment.data # pandas


# ### Configuration for BO

# In[17]:


bds = [ {'name':idx[0],'type':'continuous','domain':(0.05,100)},
        {'name':idx[1],'type':'continuous','domain':(0.05,100)},
        {'name':idx[2],'type':'continuous','domain':(0.05,100)},
        {'name':idx[3],'type':'continuous','domain':(0.05,100)},
        {'name':idx[4],'type':'continuous','domain':(0.05,100)}
        ]
        
constraints = [
    {
        'name': 'constr_1',
        'constraint': '(x[:,0] + x[:,1] + x[:,2] + x[:,3] + x[:,4]) - 100 - 0.1'
    },
    {
        'name': 'constr_2',
        'constraint': '100 - (x[:,0] + x[:,1] + x[:,2] + x[:,3] + x[:,4]) - 0.1'
    }
    ]    
    

        
batch_size = 9
num_cores = 4


# ### Run BO
# 
# `experiment.data` contains all X and Y information.

# In[ ]:


# func = experiment.Call
func = None

print(tm.ctime())
for Run in range(20):
    
    X = experiment.data[idx].values
    Y = experiment.data[[id_targ]].values.reshape([-1,1])

    batch_optimizer = BayesianOptimization(f= func,
                                           domain = bds,
                                           model_type='GP',
                                           #initial_design_numdata=20,
                                           initial_design_numdata=0,
                                           constraints = constraints,
                                           acquisition_type ='EI',
                                           acquisition_jitter = 0.5,
                                           X=X,
                                           Y= -1*Y,
                                           exact_feval = False,  
                                           evaluator_type = 'local_penalization',
                                           batch_size = batch_size,
                                           num_cores = num_cores,
                                           maximize= False)

    batch_x_next = batch_optimizer.suggest_next_locations()

    experiment.Call(batch_x_next)

    df = pd.DataFrame(batch_x_next)
    df.columns = idx
    print(tm.ctime())


# In[ ]:


# print(df)


# In[ ]:


# batch_optimizer.run_optimization()
# vars(batch_optimizer.model.model).keys()
# batch_optimizer.model
# experiment.data

