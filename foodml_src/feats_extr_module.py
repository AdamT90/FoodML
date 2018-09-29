# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 03:11:21 2018

@author: AdamT
"""

import features_extraction_funcs as feex
from sklearn.model_selection import train_test_split
import pandas as pd

def extract_feats_from_pickle(input_path, output_path, join_cols = [], label_column='ProdName'):
    in_DF = pd.read_pickle(input_path)
    feats_DF = feex.extractVarsForML(in_DF)
    feats_DF = pd.concat([feats_DF, in_DF[join_cols], in_DF[label_column]], axis=1)
    feats_DF.to_pickle(output_path)
    return feats_DF

def split_dataset(feats_DF,label_column='ProdName',cols_to_drop=[]):
    y = feats_DF[label_column]
    X = feats_DF.drop([label_column],axis=1)
    X = X.drop(cols_to_drop,axis=1)
    return train_test_split(X, y, test_size = 0.2, random_state = 666)#because hail satan
    
def prepareTrainingData(output_path, input_path='', label_column='ProdName', join_cols = [], extract=False):
    if extract:
        feats_DF = extract_feats_from_pickle(input_path, output_path, label_column)
    else:
        feats_DF = pd.read_pickle(output_path)
    return split_dataset(feats_DF,label_column)