# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 03:40:08 2018

@author: AdamT
"""

# utils
import numpy as np
import sys, traceback
import pickle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


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
        best_results_filename = directory + "_best_results_" + postfix + '.sav'
        models_info_filename = directory + "_models_info_" + postfix + '.sav'
        pickle.dump(self.best_models_dict, open(best_models_filename, 'wb'))
        pickle.dump(self.best_models_results, open(best_results_filename, 'wb'))
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
        best_results_filename = directory + "_best_results_" + postfix + '.sav'
        models_info_filename = directory + "_models_info_" + postfix + '.sav'
        self.best_models_dict = pickle.load(open(best_models_filename, 'rb'))
        self.models_info_dict = pickle.load(open(models_info_filename, 'rb'))
        self.best_models_results = pickle.load(open(best_results_filename, 'rb'))

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

    def loadModels(self, _names=[], _models=[], _params=[],
                 _models_info_dict={}, _best_models_dict={}, _best_models_results={}):
        self.model_names = self.model_names.append(_names)
        self.models =  self.models.append(_models)
        self.model_params = self.model_params.append(_params)
        self.models_info_dict.update(_models_info_dict)
        self.best_models_dict.update(_best_models_dict)
        self.best_models_results.update(_best_models_results)

    def loadTransformators(self,  _transformator_names=[], _transformator_defs=[], _transformators={}, _transformators_components={}):
        self.transformator_names = self.transformator_names.append(_transformator_names)
        self.transformator_defs = self.transformator_defs.append(_transformator_defs)
        self.transformators.update(_transformators)
        self.transformators_components.update(_transformators_components)

    def loadTransformatorsAndModels(self, _names=[], _models=[], _params=[], _transformator_names=[], _transformator_defs=[],
                 _models_info_dict={}, _best_models_dict={}, _best_models_results={},
                 _transformators={}, _transformators_components={}):
        self.loadTransformators(_transformator_names, _transformator_defs, _transformators,
                           _transformators_components)
        self.loadModels(_names, _models, _params,
                 _models_info_dict, _best_models_dict, _best_models_results)

    ##################FITTING TRANSFORMATORS###############
    def fitLoadedTransformators(self, X, reset=False):
        self.fitTransformators(self, self.transformator_names, self.transformator_defs, X, reset)

    def fitTransformators(self, names, transformators, X, reset=False):
        if reset:
            self.transformators = {}
            self.transformators_components = {}
        for name, transf in zip(names, transformators):
            transf_pipe = transf
            transf_pipe.fit(X)
            self.transformators[name] = transf_pipe
            self.transformators_components[name] = (transf_pipe.named_steps['pca'].components_,
                                                    transf_pipe.named_steps['pca'].explained_variance_ratio_)

    #######################################################

    ######################FITTING MODELS###################  
    def findBestParams(self, X_train, y_train, names, models, params, transf_postfix, prio_score, folds=4, iters=20,
                       randomized=True, scoring=None, n_jobs=-1, verbosity=0):
        '''Run grid or randomized search'''
        for model_name, model, params in zip(names, models, params):
            if verbosity > 0:
                print('Finding best parameters for ' + model_name + '_' + transf_postfix + ':')
            model_scanner = None

            try:
                if randomized:
                    model_scanner = RandomizedSearchCV(model, params, n_iter=iters,
                                                   scoring=scoring, n_jobs=n_jobs,
                                                   verbose=verbosity, cv=folds, refit=prio_score)
                else:
                    model_scanner = GridSearchCV(model, params, scoring=scoring,
                                             n_jobs=n_jobs, verbose=verbosity, cv=folds, refit=prio_score)

                #get search results and store them
                model_scanner.fit(X_train, y_train)
                model_info = \
                    model_scanner.cv_results_, model_scanner.best_params_, \
                    model_scanner.best_index_, model_scanner.best_score_
                self.models_info_dict[model_name + '_' + transf_postfix] = model_info
                self.best_models_dict[model_name + '_' + transf_postfix] = model_scanner.best_estimator_

                #evaluate models
                if verbosity > 0:
                    print('Evaluating ' + model_name + '_' + transf_postfix + '...')
                    print()
                labels = np.unique(y_train)
                self.best_models_results[model_name + '_' + transf_postfix] = self.evaluate_model(X_train, y_train,
                                                                       labels, model_scanner.best_estimator_)

            except ValueError:
                print('Value Error occured!')
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)
            except TypeError:
                print('Type error occured!')
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)
            except OverflowError:
                print('Overflow error occured!')
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)
            except AttributeError:
                print('Attribute error occured!')
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)

        return self.models_info_dict, self.best_models_dict

    def evaluate_model(self, X, y_true, labels, model):
        '''models' evaluation'''
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        accuracy_res = accuracy_score(y_true, y_pred)
        confusion_res = confusion_matrix(y_true, y_pred)
        f1_score_res = f1_score(y_true, y_pred, labels=labels, average='weighted')
        log_loss_res = log_loss(y_true, y_proba, labels=labels, normalize=True)
        classrep_res = classification_report(y_true, y_pred, target_names=labels)
        return accuracy_res, f1_score_res, log_loss_res, confusion_res, classrep_res

    def runWithFixedParams(self, X_train, y_train, labels, names, models, print_shapes=False):
        '''fit and run model with every defined transformator'''
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

    def runBenchmarkOnLoaded(self, X_train, y_train, prio_score, folds=4, iters=20, randomized=True,
                     scoring=None, n_jobs=-1, verbosity=0, reset=False, print_shapes=False):
        self.runBenchmark(X_train, y_train, self.model_names, self.models,
                    self.model_params, self.transformators, prio_score, folds, iters, randomized,
                    scoring, n_jobs, verbosity, reset, print_shapes)

    def runBenchmark(self, X_train, y_train, model_names, models, model_params, transformators, prio_score,
                     folds=4, iters=20, randomized=True,
                     scoring=None, n_jobs=-1, verbosity=0, reset=False, print_shapes=False):
        '''search best parameters for set of models'''

        #models are not included in pipelines with transformators,
        #because we want to avoid waiting for the same transformator to fit multiple times to the same dataset
        for transf_name, transformer in transformators.items():
            X_transf = transformer.transform(X_train)
            if print_shapes:
                print('Before PCA, ' + 'x: ' + str(X_train.shape[0]) + "; y: " + str(X_train.shape[1]))
                print('After PCA, ' + 'x: ' + str(X_transf.shape[0]) + "; y: " + str(X_transf.shape[1]))
                self.findBestParams(X_transf, y_train, model_names,
                                    models, model_params, transf_name, prio_score, folds, iters,
                                    randomized, scoring, n_jobs, verbosity)
        return self.best_models_results

    def fixedParamsBenchmark(self, X_train, y_train, labels, print_shapes=False):
        return self.runWithFixedtParams(X_train, y_train, labels, self.model_names, self.models,
                                        print_shapes=print_shapes)
    #######################################################

