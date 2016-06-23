#!/usr/bin/env python2

import pandas as pd
import numpy as np
import scipy as sp
import pdb
import sys

from time import time
import random

from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import log_loss

from sklearn.cross_validation import KFold, StratifiedShuffleSplit
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from numpy import var, average, mean, median
from math import log

# Extracted from https://www.kaggle.com/wiki/LogarithmicLoss
def multi_logloss(act, pred):
    N = len(pred)

    s = 0
    for i in range(N):
        s += act[i] * sp.log(pred[i])

    return sum(s) * (-1.0/N)

def prepare_data(data_file):
    for name, series in data_file.iteritems():
        if series.dtype == 'O':
            # Encode input values as an enumerated type or categorical variable
            data_file[name], tmp_indexer = pd.factorize(data_file[name])
            pass
        elif len(data_file[series.isnull()]) > 0:
            # NaN variables. It's like missing data
            data_file.loc[series.isnull(), name] = -999
    return data_file

def load_file(path, drop_attr=[]):
    # loads CSV file
    in_file = pd.read_csv(path)

    # drop selected columns
    in_file.drop(drop_attr, inplace=True, axis=1)

    #return prepare_data(in_file)
    return in_file


def data_fold(df, target, k, method='strat', shuffle=True):
    n_labels = df.axes[1].size

    print 'Folding data'
    if method == 'strat':
        m = StratifiedShuffleSplit(target, k, test_size=0.1)
    else:
        m = KFold(len(target), k, shuffle)

    #n_source = pd.get_dummies(df)
    #print n_source.columns
    n_source = df

    folds = []
    for train_indices, trial_indices in m:
        # This is the way to access values in a pandas DataFrame
        folds.append({'train': {'source': n_source.ix[train_indices, :],
            'target': target[train_indices]},
                      'trial': {'source': n_source.ix[trial_indices, :],
                          'target': target[trial_indices]}})
    return folds


def learn_and_test(cls, folds, cls_name):
    worst = auc_total = idx = 0
    best = 999999999.0
    loss = []

    lb = LabelBinarizer()
    #lb.fit(['Adoption','Died','Euthanasia','Return_to_owner','Transfer'])
    lb.fit(folds[0]['trial']['target'])

    for fold in folds:
        print '\nFold', idx+1
        print 'Fitting'
        if cls_name == 'xgboost':
            cls = modelfit(cls, fold['train']['source'],
                    fold['train']['target'], useTrainCV=False)
        cls.fit(fold['train']['source'], fold['train']['target'])

        print 'Predicting'
        pred_classes = cls.predict_proba(fold['trial']['source'])
        actual_classes = lb.transform(fold['trial']['target'])

        #ll = multi_logloss(actual_classes, pred_classes)
        ll = log_loss(actual_classes, pred_classes)
        loss.append(ll)
        print ll

        best = (loss[idx] if loss[idx] <  best else best)
        worst = (loss[idx] if loss[idx] > worst else worst)
        idx += 1

    return {'best': str(best).zfill(15),
            'worst': str(worst).zfill(15),
            'mean': str(mean(loss)).zfill(15),
            'variance': str(var(loss)).zfill(15)}


def export_run(f, r, R=0):
    print "Writing results"
    f.write(str(R).zfill(4) + ',' +
            str(r['best']) + ',' +
            str(r['worst']) + ',' +
            str(r['mean']) + ',' +
            str(r['variance']) + ',' +
            str(r['time']) + "\n")
    f.flush()


def modelfit(alg, dtrain, target, useTrainCV=True, cv_folds=10):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgb_param['num_class'] = 5
        xgb_param['eval_metrics'] = 'mlogloss'
        xgtrain = xgb.DMatrix(dtrain.values, label=target.values)
        #cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=300, nfold=cv_folds,
            metrics=["mlogloss"], show_progress=True)

        min_ll = cvresult['test-mlogloss-mean'].min()
        print min_ll
        est = cvresult.loc[cvresult['test-mlogloss-mean'] == min_ll]['test-mlogloss-mean']
        print est
        alg.set_params(n_estimators=est)

    return alg

def run_cls(cls, source, target, cls_name):
    print "Preparing folds"
    folds = data_fold(source, target, 10)

    print "Learning and testing folds"
    st = time()
    result =  learn_and_test(cls, folds, cls_name)
    et = time() - st
    result.update({'time': str(et)})

    return result

def main(classifier_name):
    # relative to each machine
    data_path = '../../data/'
    train_file = data_path + 'transformed_train.csv'
    test_file = data_path + 'transformed_test.csv'


    print "Loading data"
    # loading test set
    test_set = pd.read_csv(test_file)
    # loading train set
    train_set = pd.read_csv(train_file)

    # feature selection
    # Feature selection by Weka without transformation
    cls_col01 = ['AgeuponOutcome', 'IsNamed', 'IsIntact', 'BreedA', 'AnimalType']

    # All available columns
    cls_col02 = ['AnimalID', 'Name', 'DateTime', 'OutcomeType', 'OutcomeSubtype',
            'AnimalType', 'SexuponOutcome', 'AgeuponOutcome', 'Breed', 'Color',
            'IsNamed', 'Date', 'Time', 'Gender', 'IsIntact', 'AgeInDays',
            'AgeInCategory', 'BreedA', 'BreedB', 'IsMix', 'ColorA', 'ColorB',
            'OutcomeTypeEncoded', 'Adoption', 'Died', 'Euthanasia',
            'Return_to_owner', 'Transfer']

    # Columns from test dataset
    cls_col03 = ['ID', 'Name', 'DateTime', 'AnimalType', 'SexuponOutcome',
    'AgeuponOutcome', 'Breed', 'Color', 'IsNamed', 'Date', 'Time', 'Gender',
    'IsIntact', 'AgeInDays', 'AgeInCategory', 'BreedA', 'BreedB', 'IsMix',
    'ColorA', 'ColorB']

    # Sent to kaggle (best logloss 1.22 naive bayes)
    cls_col04 = ['DateTime', 'AgeuponOutcome', 'Gender', 'IsIntact', 'IsNamed',
    'BreedA', 'BreedB', 'AnimalType', 'ColorA', 'ColorB']

    # Available columns after transformation
    cls_col05 = ['DateTime', 'AnimalType', 'SexuponOutcome',
        'AgeuponOutcome', 'Breed', 'Color', 'IsNamed', 'Date', 'Time', 'Gender',
        'IsIntact', 'AgeInDays', 'AgeInCategory', 'BreedA', 'BreedB', 'IsMix',
        'ColorA', 'ColorB']

    # Useful columns
    cls_col06 = ['AnimalType',
        'IsNamed', 'Date', 'Time', 'Gender',
        'IsIntact', 'AgeInDays', 'AgeInCategory', 'BreedA', 'BreedB', 'IsMix',
        'ColorA', 'ColorB']

    # Feature selection by Weka after transformation 1.09 1.11
    cls_col07 = ['IsNamed', 'Time', 'IsIntact', 'AgeInDays',
            'BreedA', 'AgeInCategory', 'AnimalType']
    # 1.03 1.05
    cls_col08 = ['IsNamed', 'Time', 'IsIntact', 'AgeInDays']


    # preparing data to classify
    print "Preparing data"
    train_outcome = train_set['OutcomeTypeEncoded']

    cls_col = cls_col07
    train_set = train_set[cls_col]
    test_set = test_set[cls_col]
    full_set = prepare_data(pd.concat([train_set, test_set]))

    train_set = full_set[:len(train_outcome)]
    test_set = full_set[len(train_outcome):]

    # creating classifier
    if classifier_name == 'randomforest':
        print("Random Forest")
        cls = RandomForestClassifier(n_estimators = 1000, n_jobs=-1)
    elif classifier_name == 'naivebayes':
        print("Naive Bayes")
        cls = GaussianNB()
    elif classifier_name == 'xgboost':
        print("XGBoost")

        cls = XGBClassifier(
         learning_rate=0.2,
         n_estimators=63,
         max_depth=6,
         min_child_weight=1,
         gamma=0,
         subsample=0.75,
         colsample_bytree=0.85,
         objective='multi:softprob',
         scale_pos_weight=1,
         seed=27)

        #xgb_params = cls.get_xgb_params()
        #xgb_params['num_class'] = 5
    elif classifier_name == 'decisiontree':
        print("Decision Tree")
        pass
    elif classifier_name == 'knn':
        print("Knn")
        cls = KNeighborsClassifier(n_neighbors = 2000)

    result = run_cls(cls, train_set, train_outcome, classifier_name)

    #f = open('1-'+classifier_name + '.txt', 'w')
    #f.write("N, BEST, WORST, MEAN, MEDIAN, VARIANCE, TIME\n")
    #export_run(f, result)

    print('Saving output')
    output = pd.Series(result)
    output.to_csv('1-'+classifier_name+'.csv', header=False)


    '''
    print("Fitting")
    cls.fit(train_set.values, train_outcome.values)
    print("Classifying")
    predictions = cls.predict_proba(test_set.values)

    # saving output
    print("Saving output")
    output = pd.DataFrame(predictions, columns=['Adoption','Died','Euthanasia','Return_to_owner','Transfer'])
    output.index += 1
    output.to_csv("out.csv", index_label='ID')
    '''




if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("\nInput file was expected.\nExiting...\n")
        exit(1)
