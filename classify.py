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
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

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
    cls_col04 = ['DateTime', 'AgeUponOutcome', 'Gender', 'IsIntact', 'IsNamed',
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

    # Feature selection by Weka after transformation
    cls_col07 = ['IsNamed', 'Time', 'IsIntact', 'AgeInDays',
            'BreedA', 'AgeInCategory', 'AnimalType']
    cls_col08 = ['IsNamed', 'Time', 'IsIntact', 'AgeInDays']


    # preparing data to classify
    print "Preparing data"
    train_outcome = train_set['OutcomeTypeEncoded']

    cls_col = cls_col08
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
        pass
    elif classifier_name == 'decisiontree':
        print("Decision Tree")
        pass


    print("Fitting")
    cls.fit(train_set.values, train_outcome.values)
    #cls.fit(total_set[:len(train_set)], train_outcome.values)
    print("Classifying")
    predictions = cls.predict_proba(test_set.values)
    #predictions = cls.predict_proba(total_set[len(train_set):])

    # saving output
    print("Saving output")
    output = pd.DataFrame(predictions, columns=['Adoption','Died','Euthanasia','Return_to_owner','Transfer'])
    output.index += 1
    output.to_csv("out.csv", index_label='ID')




if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("\nInput file was expected.\nExiting...\n")
        exit(1)
