#!/usr/bin/env python
# coding: utf-8
# version: 2020.03.29

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

# --------------------------------------------------------------------------------------------------------
# args
# --------------------------------------------------------------------------------------------------------


import argparse
# Params
parser = argparse.ArgumentParser(description='Model selection')
parser.add_argument('--inputs', default='C', type=str, help='map inputs')

args = parser.parse_args()
inputs = args.inputs

# --------------------------------------------------------------------------------------------------------

graph_file = {
    'C':'graph_array_C_368.dat',
    'CA':'graph_array_CA_480.dat',
    'CR':'graph_array_CR_480.dat',
    'CS':'graph_array_CS_265.dat',
    'CAR':'graph_array_CAR_104.dat',
    'CAS':'graph_array_CAS_63.dat',
    'CRS':'graph_array_CRS_63.dat',
    'CARS':'graph_array_CARS_3.dat',
}
print(inputs)
print(graph_file[inputs])

data_org = pd.read_excel('data_all.xlsx') 
graph_array = data_load(graph_file[inputs])


length_array,nodes_array,node_array,edges_array,str_array = [],[],[],[],[]
for G in graph_array:
    length_array.append(nx_path_length(G))
    nodes_array.append(len(G.nodes))
    node_array.append(''.join(G.nodes))
    edges_array.append(len(G.edges))
    str_array.append(nx2str(G))

scores_all_array = []
for i in range(len(graph_array)):
    score = pd.read_excel('scores/score_graph_%s_%d.xlsx'%(inputs,i),index_col=0,)
    score['ID'] = i
    scores_all_array.append(score)

scores_all = pd.concat(scores_all_array,axis=0)
scores = scores_all.groupby('ID').mean()

scores['r2_score_std'] = scores_all.groupby('ID').std()['r2_score']


df = scores
df['#node'] = nodes_array
df['#edge'] = edges_array
df['path length'] = length_array
df['node'] = node_array
df['str'] = str_array

df['inputs'] = inputs

# save scores

scores_all.to_excel('scores_%s_all.xlsx'%inputs)
scores.to_excel('scores_%s.xlsx'%inputs)
scores

