#!/usr/bin/env python
# coding: utf-8
# version: 2020.03.27

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform

# from sklearn.utils import shuffle
import sklearn.utils as sku


## basic info

idC = ['P3HT vol (µL)','D1 vol (µL)', 'D2 vol (µL)', 'D6 vol (µL)', 'D8 vol (µL)',]
idAll = ['C','A','R','T','S','Y']

sigma_y = 0.3

## functions

def y_class(y):
    '''
    assign grooup class for 'log(conductivity)' -- used in cross-validation and train_test_split
    '''
    if y<2.5:
        return 0
    elif y<4:
        return 1
    elif y<5:
        return 2
    elif y<6:
        return 3
    elif y<7:
        return 4
    else:
        return -1
    
def invp_dist(x,d0=1.0):
    '''
    a function for resampling data_org
    d0 is a number control the reference distance -- default 1.0
    (median distance is 2.51)
    '''
    return np.exp(-x**2 / (2*d0**2))

def fill_columns(data,addition=True,d0=1.0):
    
    data = data.dropna(axis=0,how='all',)    
    
    data['Absorbance @525nm (log)'] = np.log(data['Absorbance @525nm (doped)'])
    data['Absorption Ratio (log)'] = np.log(data['Absorption Ratio'])
    data['Sheet Resistance (ohm/sq) (log)'] = np.log(data['Sheet Resistance (ohm/sq)'])
    data['Thickness (nm, measured) (log)'] = np.log(data['Thickness (nm, measured)'])
    data['Conductivity'] = 10000000/(data['Sheet Resistance (ohm/sq)'] * data['Thickness (nm, measured)'])
    data['Conductivity (log)'] = np.log(data['Conductivity'])

    data['A'] = data['Absorbance @525nm (log)'] # will change back to 'Absorbance @525nm'
    data['R'] = data['Absorption Ratio (log)']
    data['T'] = data['Thickness (nm, measured) (log)']
    data['S'] = data['Sheet Resistance (ohm/sq) (log)']
    data['Y'] = data['Conductivity (log)']
    
    if addition:
        return fill_columns_addition(data,d0)
    
    return data

def fill_columns_addition(data_all,d0=1.0):
    ''' add p-content -- used in resampling process
    '''
    
    df = data_all
    df['ID'] = df['Run']*100 + df['Wafer No.']*10 + df['Sample No.'] # test -- special 
    df['p-droplet'] = df[idAll[1:]].apply(lambda r:10**np.sum(r==r),axis=1) # test -- high p for ...
    df['p-droplet'] = df['p-droplet']/df.groupby('ID')['p-droplet'].transform('sum')
    
    data = random_chosen_droplet(data_all)

    dist = euclidean_distances(data[idC],data[idC])
    invp = np.sum(invp_dist(dist,d0),axis=0)
    p = 1/invp
    data['p-content'] = p
    data_all = pd.merge(data_all,data[['ID','p-content']],on='ID')
    data_all['d0'] = d0
    return data_all

def valid_supp_split(data,valid_split=True,subset=idAll[1:]):
    
    if not valid_split:
        return data
    
    data_allvalid = data.dropna(axis=0,how='any',subset=subset)
    data_supp = data.drop(index=data_allvalid.index)
    
    return data,data_allvalid,data_supp

def random_chosen_droplet(df,valid_split=False):
    # random choen samples -- one for each group
    # e.g.: data,data_allvalid,data_supp = random_chosen(data_org)
    fn = lambda obj: obj.loc[np.random.choice(obj.index, 1, True,p=obj['p-droplet']),:]
    data = df.groupby('ID',sort=False,as_index=False).apply(fn)

    return valid_supp_split(data,valid_split)

def random_chosen_content(data,valid_split=False):
    '''random chosen rows by probability p-content
    '''
    data['rand'] = np.random.rand(len(data))
    data = data[data['rand']<=data['p-content']]
    
    return valid_supp_split(data,valid_split)


def random_resample(data,onedroplet=True,valid_split=False,shuffle=False):
    '''random chosen rows: random_chosen_droplet + random_chosen_content
    '''
    data = random_chosen_droplet(data)
    data = random_chosen_content(data)
    
    if shuffle:
        data = sku.shuffle(data)
    
    return valid_supp_split(data,valid_split)

# name

def cnt_renamer(xlabels): # sort, and then change C
    '''idC = ['P3HT vol (µL)', 'D1 vol (µL)', 'D2 vol (µL)', 'D6 vol (µL)', 'D8 vol (µL)']
    idAll = ['C','A','R','T','S','Y']
    '''
    idx = []
    for x in idAll:
        if x not in xlabels:
            continue
        if x =='C':
            idx += idC;
        else:
            idx.append(x);
    assert len(idx) > 0
    return idx

