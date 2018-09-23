# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 03:40:08 2018

@author: AdamT
"""


#utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV

#preprocessing
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import scale
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA

#evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB


def checkmodels(models,data):
    X_train, X_test, y_train, y_test = data
    outinf = []
    for m in models:
        mtype, params = m
        foundm = GridSearchCV(mtype, params, cv=6)
        print(mtype, "////", params)
        foundm.fit(X_train, y_train)
        bestp = foundm.best_params_
        acctest = accuracy_score(y_test,foundm.best_estimator_.predict(X_test))
        acctrain = accuracy_score(y_train,foundm.best_estimator_.predict(X_train))
        outinf.append([foundm.best_estimator_,bestp,acctest,acctrain])
    return outinf