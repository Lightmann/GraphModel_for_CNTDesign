#!/usr/bin/env python
# coding: utf-8
# version: 2020.03.29 --> 2020.05.23

import sys
for path in ['../','../../','../../../','../../../../']:
    sys.path.append(path+"CNT_Code/V1.05.22")
    
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from RegressionTools import *
from GraphTools import *
from GeneralTools import *
# from GraphModel import *
from GraphModel_cnt import *
from CNT_Dataset import *
from ModelSelection import *

import random

random.seed(1)
np.random.seed(1)

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONHASHSEED'] = str(1)


# --------------------------------------------------------------------------------------------------------
# args
# --------------------------------------------------------------------------------------------------------


import argparse
# Params
parser = argparse.ArgumentParser(description='Model selection')
parser.add_argument('--id_graph', default=0, type=int, help='id_graph')

args = parser.parse_args()
id_graph = args.id_graph

# --------------------------------------------------------------------------------------------------------

GB_ESTIMATORS = 400
GB_SEARCH_ITER = 80
CV = 4
RATIO = 0.9
MULTI_ESTIMATOR = 50
MULTI_PREDICTION = 50
SCORE_RUN = 20

# --------------------------------------------------------------------------------------------------------


data_org = pd.read_excel('data_all_back_A.xlsx')
graph_array = data_load('graph_array_368.dat')

# generate resampling data first -- keep same for all graph models
data_array = []
for i in range(SCORE_RUN):
    data = random_resample(data_org,shuffle=True)
    data_array.append(data)

G = graph_array[id_graph]
gm = GraphModel_cnt(G=G,idy='Y',renamer=cnt_renamer,
                    assign_submodel=assign_submodel_cnt,
                    gb_estimators=GB_ESTIMATORS,n_iter_search=GB_SEARCH_ITER,submodel_cv=CV,
                   )

score_array = []
data_pred_array = []
for i in range(SCORE_RUN):
    data = data_array[i] # prepared
    data = data.dropna(subset=gm.renamer(idAll),how='any').reset_index(drop=True) # all-valid
    score,data_pred = cross_val_score_gm(gm,data,reuse=False,cv=CV,return_prediction=True,
                               ratio=RATIO,
                               n_estimator=MULTI_ESTIMATOR,
                               sigma=0.3,
                               n_predict=MULTI_PREDICTION)
    score['model'] = nx2str(gm.G,model=True)
    data_pred['model'] = nx2str(gm.G,model=True)
    score_array.append(score)
    data_pred_array.append(data_pred)

    print(gm.G._node)
    
# save scores

scores = pd.concat(score_array)
data_preds = pd.concat(data_pred_array)
scores.to_excel('scores/score_graph_%d.xlsx'%id_graph)
with pd.ExcelWriter('predictions/predictions_graph_%d.xlsx'%id_graph) as writer:
    for i,df in enumerate(data_pred_array):
        df.to_excel(writer,sheet_name='%d'%i)





