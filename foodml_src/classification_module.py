# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 03:40:08 2018

@author: AdamT
"""

# utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# preprocessing
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import scale
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA

# evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB


class ModelsBenchmark:

    def __init__(self, _names=[], _models=[], _params=[], _transformator_names=[], _transformator_defs=[],
                 _models_info_dict={}, _best_models_dict={}, _best_models_results={},
                 _transformators={}, _transformators_components={}):
        self.model_names = _names
        self.models = _models
        self.model_params = _params
        self.transformator_names = _transformator_names
        self.transformator_defs = _transformator_defs
        self.models_info_dict = _models_info_dict
        self.best_models_dict = _best_models_dict
        self.best_models_results = _best_models_results
        self.transformators = _transformators
        self.transformators_components = _transformators_components

    ########SAVING FROM FILES AND LOADING FROM FILES##########
    def saveTransformatorsToPickles(self, directory, postfix):
        transformators_filename = directory + "_transformators_" + postfix + '.sav'
        transformators_components_filename = directory + "_transformators_components_" + postfix + '.sav'
        pickle.dump(self.transformators, open(transformators_filename, 'wb'))
        pickle.dump(self.transformators_components, open(transformators_components_filename, 'wb'))

    def saveModelsToPickles(self, directory, postfix):
        best_models_filename = directory + "_best_models_" + postfix + '.sav'
        models_info_filename = directory + "_models_info_" + postfix + '.sav'
        pickle.dump(self.best_models_dict, open(best_models_filename, 'wb'))
        pickle.dump(self.models_info_dict, open(models_info_filename, 'wb'))

    def saveToPickles(self, directory, postfix):
        self.saveTransformatorsToPickles(directory, postfix)
        self.saveModelsToPickles(directory, postfix)

    def loadTransformatorsFromPickles(self, directory, postfix):
        transformators_filename = directory + "_transformators_" + postfix + '.sav'
        transformators_components_filename = directory + "_transformators_components_" + postfix + '.sav'
        self.transformators = pickle.load(open(transformators_filename, 'rb'))
        self.transformators_components = pickle.load(open(transformators_components_filename, 'rb'))

    def loadModelsFromPickles(self, directory, postfix):
        best_models_filename = directory + "_best_models_" + postfix + '.sav'
        models_info_filename = directory + "_models_info_" + postfix + '.sav'
        self.best_models_dict = pickle.load(open(best_models_filename, 'rb'))
        self.models_info_dict = pickle.load(open(models_info_filename, 'rb'))

    def loadFromPickles(self, directory, postfix):
        self.loadTransformatorsFromPickles(directory, postfix)
        self.loadModelsFromPickles(directory, postfix)

    #######################################################

    ####################EXPORTING TO TUPLES################
    def exportModelDefsToTuple(self):
        return (self.model_names, self.models, self.model_params)

    def exportTransformatorDefsToTuple(self):
        return (self.transformator_names, self.transformator_defs)

    def exportModelResultsToTuple(self):
        return (self.models_info_dict, self.best_models_dict, self.best_models_results)

    def exportTransformatorResultsToTuple(self):
        return (self.transformators, self.transformators_components)

    def exportDefsToTuple(self):
        return self.exportModelDefsToTuple() + self.exportTransformatorDefsToTuple()

    def exportResultsToTuple(self):
        return self.exportModelResultsToTuple() + self.exportTransformatorResultsToTuple()

    def exportAllToTuple(self):
        return self.exportDefsToTuple() + self.exportResultsToTuple()

    #######################################################

    def loadModels(self, _names=[], _models=[], _params=[], _transformator_names=[], _transformator_defs=[],
                 _models_info_dict={}, _best_models_dict={}, _best_models_results={},
                 _transformators={}, _transformators_components={}):
        self.model_names = self.model_names.append(_names)
        self.models =  self.models.append(_models)
        self.model_params = self.model_params.append(_params)
        self.transformator_names = self.transformator_names.append(_transformator_names)
        self.transformator_defs = self.transformator_defs.append(_transformator_defs)
        self.models_info_dict.update(_models_info_dict)
        self.best_models_dict.update(_best_models_dict)
        self.best_models_results.update(_best_models_results)
        self.transformators.update(_transformators)
        self.transformators_components.update(_transformators_components)

    ##################FITTING TRANSFORMATORS###############
    def fitLoadedPCA(self, X, reset=False):
        self.fitPCA(self, self.transformator_names, self.transformator_defs, X, reset)

    def fitPCA(self, pca_names, pca_tranformers, X, reset=False):
        if reset:
            self.transformators = {}
            self.transformators_components = {}
        for name, transf in zip(pca_names, pca_tranformers):
            transf_pipe = make_pipeline(StandardScaler(), transf)
            transf_pipe.fit(X)
            self.transformators[name] = transf_pipe
            self.transformators_components[name] = (transf_pipe.named_steps['pca'].components_,
                                                    transf_pipe.named_steps['pca'].explained_variance_)

    #######################################################

    ######################FITTING MODELS###################  
    def findBestParams(self, X_train, y_train, names, models, params, transf_postfix, folds=4, iters=20,
                       randomized=True, scoring=None, n_jobs=-1, verbosity=0, reset=False):
        if reset:
            self.models_info_dict = {}
            self.best_models_dict = {}
        for model_name, model, params in zip(names, models, params):
            if verbosity > 0:
                print('Finding best parameters for ' + model_name + ':')
            model_scanner = None
            if randomized:
                model_scanner = RandomizedSearchCV(model, params, n_iter=iters,
                                                   scoring=scoring, n_jobs=n_jobs,
                                                   verbose=verbosity, cv=folds)
            else:
                model_scanner = GridSearchCV(model, params, scoring=scoring,
                                             n_jobs=n_jobs, verbose=verbosity, cv=folds)

            model_scanner.fit(X_train, y_train)
            model_info = \
                model_scanner.cv_results_, model_scanner.best_params_, \
                model_scanner.best_index_, model_scanner.best_score_
            self.models_info_dict[model_name + '_' + transf_postfix] = model_info
            self.best_models_dict[model_name + '_' + transf_postfix] = model_scanner.best_estimator_
            if verbosity > 0:
                print('Evaluating ' + model_name + '_' + transf_postfix + ':')
            labels = np.unique(y_train)
            self.best_models_results[model_name + '_' + transf_postfix] = self.evaluate_model(X_train, y_train,
                                                                       labels, model_scanner.best_estimator_)
        return self.models_info_dict, self.best_models_dict

    def evaluate_model(self, X, y_true, labels, model):
        y_pred = model.predict(X)
        accuracy_res = accuracy_score(y_true, y_pred)
        confusion_res = confusion_matrix(y_true, y_pred)
        # rocauc_res = roc_auc_score(y_true, y_pred)
        classrep_res = classification_report(y_true, y_pred, target_names=labels)
        return accuracy_res, confusion_res, classrep_res

    def runWithFixedParams(self, X_train, y_train, labels, names, models, print_shapes=False):
        default_model_results = {}
        for transf_name, transformer in self.transformators.items():
            X_transf = transformer.transform(X_train)
            if print_shapes:
                print('Before PCA, ' + 'x: ' + str(X_train.shape[0]) + "; y: " + str(X_train.shape[1]))
                print('After PCA, ' + 'x: ' + str(X_transf.shape[0]) + "; y: " + str(X_transf.shape[1]))
            for model_name, model in zip(names, models):
                model.fit(X_transf, y_train)
                default_model_results[model_name + '_' + transf_name] = self.evaluate_model(X_transf, y_train, labels,
                                                                                            model)
        return default_model_results, model

    def fitModels(self, X, y, models):
        for model_name in models:
            models[model_name].fit(X, y)

    def runBenchmark(self, X_train, y_train, X_val=None, y_val=None,
                     labels_val=None, vaildate=False,
                     folds=4, iters=20, randomized=True,
                     scoring=None, n_jobs=-1, verbosity=0, reset=False, print_shapes=False):
        if reset:
            self.best_models_results = {}
        for transf_name, transformer in self.transformators.items():
            X_transf = transformer.transform(X_train)
            if print_shapes:
                print('Before PCA, ' + 'x: ' + str(X_train.shape[0]) + "; y: " + str(X_train.shape[1]))
                print('After PCA, ' + 'x: ' + str(X_transf.shape[0]) + "; y: " + str(X_transf.shape[1]))
                self.findBestParams(X_transf, y_train, self.model_names,
                                    self.models, self.model_params, transf_name, folds, iters,
                                    randomized, scoring, n_jobs, verbosity)
            #for model_name, model in self.best_models_dict.items():
            #    if verbosity > 0:
            #        print('Evaluating ' + model_name + ':')
            #    self.best_models_results[model_name] = self.evaluate_model(X_transf, y_train,
            #                                                               labels_train, model)
        return self.best_models_results

    def fixedParamsBenchmark(self, X_train, y_train, labels, print_shapes=False):
        return self.runWithFixedtParams(X_train, y_train, labels, self.model_names, self.models,
                                        print_shapes=print_shapes)
    #######################################################

class ResultsReport:
    def __init__(self):
        self.PCA_featuresDF = pd.DataFrame()
        self.features_importancesDF = pd.DataFrame()
        self.models_summaryDF = pd.DataFrame()
        self.class_reportsDict = {}
        self.conf_matrixesDixt = {}

    def print_class_reports(self, model_names):
        for name in model_names:
            print('Classification Report for ' + name + ':')
            print(self.class_reportsDict[name])
            print('###############################################\n')

    def print_conf_matrixes(self, model_names):
        for name in model_names:
            print('Classification Report for ' + name + ':')
            print(self.conf_matrixesDixt[name])
            print('###############################################\n')

