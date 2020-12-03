#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx 

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


# graphs

graph_array = data_load('graph_array_368.dat')

n_graph = len(graph_array)
length_array,nodes_array,node_array,edges_array,str_array = [],[],[],[],[]
for G in graph_array:
    length_array.append(nx_path_length(G))
    nodes_array.append(len(G.nodes))
    node_array.append(''.join(G.nodes))
    edges_array.append(len(G.edges))
    str_array.append(nx2str(G))

len(G.edges)


# collect scores

scores_all_array = []
for i in range(n_graph):
    score = pd.read_excel('scores/score_graph_%d.xlsx'%i,index_col=0,)
    score['ID'] = i
    scores_all_array.append(score)

scores_all = pd.concat(scores_all_array,axis=0)
scores = scores_all.groupby('ID').mean()


df = scores
df['#node'] = nodes_array
df['#edge'] = edges_array
df['path length'] = length_array
df['node'] = node_array
df['str'] = str_array

score_array = scores['r2_score'].values


# save

scores_all.to_excel('scores_all.xlsx')
scores.to_excel('scores.xlsx')
scores



