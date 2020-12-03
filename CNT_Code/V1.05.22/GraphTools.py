#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import networkx as nx 

''' demo:

G = str2nx('C->RS, CR->T, ST->Y')
nx.draw_networkx(G,nx.circular_layout(G))
plt.axis('equal')

nx2str(G)

nx_get_task_list(G,sort=True)

nx.set_node_attributes(G,{},'model')

nx.get_node_attributes(G,'model')

'''

# create graph model

def str2nx(s,G=None,methods=None,models=None):
    '''create the graph model by a simple string
    example:
    import networkx as nx     
    G = str2nx('C->RS, CR->T, ST->Y')
    G = str2nx('C->RS, CR->T, ST->Y',models={'S':Linear,'Y':Linear,'R':GB})
    nx.draw_networkx(G,nx.circular_layout(G))
    '''
    s = s.replace(' ','')
    if G is None:
        G = nx.DiGraph()
    for m in s.split(','):
        i,o = m.split('->')
        edges = [(ii,oi) for ii in i for oi in o]
        G.add_edges_from(edges)
    
    #nx.get_node_attributes(G,'model')
    if methods is not None:
        nx.set_node_attributes(G,methods,'method') # initial
    if models is not None:
        #nx.set_node_attributes(G,{'C':'GB'},'model')
        nx.set_node_attributes(G,models,'model') # G.nodes['C']['model'] = ...
        
    return G

def nx2str(G,target='Y',model=False):
    '''suppose the graph G is a valid model, return its string
    G = str2nx('C->RS, CR->T, ST->Y')
    nx2str(G) # 'C->R, CR->T, C->S, ST->Y'
    '''
    xy_array = nx_get_task_list(G,target)
    s = []
    for x,y in xy_array[::-1]:
        if s != []:
            s += ', '
        s += ''.join(x) +'->' + y
        if model:
            if 'model' in G.nodes[y].keys():
                s += '(%s)'%G.nodes[y]['model'].__class__.__name__
    s = ''.join(s)
    return s

def list2nx(s,G=None,methods=None,models=None):
    return None

def nx_get_task_list(G,target='Y',renamer=None,sort=False):
    '''suppose the graph G is a valid model, return a list of sub-models
    renamer is a function: e.g. change 'C' as ['C1','C2'] -- xs_regular
    G = str2nx('C->RS, CR->T, ST->Y')
    get_task_list(G) # [[['S', 'T'], 'Y'], [['C'], 'S'], [['C', 'R'], 'T'], [['C'], 'R']]
    nx2str(G) # 'C->R, CR->T, C->S, ST->Y'
    '''
    dist = {n:len(max(nx.all_simple_paths(G, n, target), key=lambda x: len(x)))-1 for n in G.nodes if n != target}
    dist.update({target:0}) # e.g. {'C': 3, 'A': 1, 'R': 2, 'S': 1, 'Y': 0}
    
    node_array = sorted(dist.keys(), key= lambda k : dist[k]) # ['Y', 'A', 'S', 'R', 'C']
    
    xy_array = []
    for node in node_array: # last one is 'C'
        xs = [x for x in G.predecessors(node)]
        if len(xs)==0: continue; # omit -- inputs
        if renamer is not None:
            xs = renamer(xs)
        xy_array.append([xs,node])
    if sort:
        return xy_array[::-1] # 'ST->Y' is the last one
    return xy_array # 'ST->Y' is the first one

# plot tools

def nx_plot_cnt(G):
    
    G = G.copy()
    G.add_nodes_from(['Y','S','R','C','A','T'])
    
    G0 = nx.Graph()
    G0.add_nodes_from(['Y','S','R','C','A','T'], size=10)
    pos = nx.circular_layout(G0)
    nx.draw_networkx(G, pos,with_label = True,node_color ='green',node_size = 1000,font_size=16) 
    
# valid graph

def valid_graph(G,inputs=['C'],output='Y'):
    
    # input is 'C' and output is 'Y'
    if not set(inputs+[output]).issubset(G.nodes):
        return False
    for n in inputs:
        if G.in_degree[n] > 0 or G.out_degree[n] == 0:
            return False
    if G.in_degree[output] == 0 or G.out_degree[output] >0:
        return False
    
    # other are inter media nodes
    for node in G.nodes:
        if node not in inputs+[output]:
            if G.in_degree[node]==0 or G.out_degree[node]==0:
                return False
            
    # no circle
    try:
        nx.find_cycle(G, orientation='original')
        return False
    except:
        pass
    
    # upto now: 45791 graphs
    
    # T only to Y
    if 'T' in G.nodes:
        if G.out_degree['T'] > 1:
            return False
        if not G.has_edge('T','Y'):
            return False
        
        # (T,S) only to Y
        if 'S' in G.nodes:
            if G.in_degree['Y'] != 2:
                return False
            if not G.has_edge('S','Y'):
                return False
        
    # upto noew: 1600 graphs -- will be futher reduced
    
    return True

def nx_path_length(G):
    '''maximal path length
    '''
    return nx.algorithms.dag.dag_longest_path_length(G)

# Refs
# https://stackoverflow.com/questions/56362785/calculate-the-longest-path-between-two-nodes-networkx
