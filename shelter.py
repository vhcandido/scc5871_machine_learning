import pandas as pd
import numpy as np
import scipy as sp
import pdb
import sys

from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import log_loss

from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from numpy import var, average, mean, median

# Extracted from https://www.kaggle.com/wiki/LogarithmicLoss
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def prepare_data(data_file):
    for name, series in data_file.iteritems():
        if series.dtype == 'O':
            # Encode input values as an enumerated type or categorical variable
            data_file[name], tmp_indexer = pd.factorize(data_file[name])
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


def data_fold(df, target, k, shuffle=False):
    n_labels = df.axes[1].size
    source = df.values

    kf = KFold(len(source), k, shuffle)

    n_source = pd.get_dummies(df)

    folds = []
    for train_indices, trial_indices in kf:
        # This is the way to access values in a pandas DataFrame
        folds.append({'train': {'source': n_source.ix[train_indices, :],
                                'target': target[train_indices]},
                      'trial': {'source': n_source.ix[trial_indices, :],
                                'target': target[trial_indices]}})
    return folds


def learn_and_test(cls, folds):
    worst = auc_total = idx = 0
    best = 999999999.0
    loss = []

    lb = LabelBinarizer()
    lb.fit(['Adoption','Died','Euthanasia','Return_to_owner','Transfer'])

    for fold in folds:
        print 'Fold', idx+1
        print 'Fitting'
        cls.fit(fold['train']['source'], fold['train']['target'])

        print 'Predicting'
        pred_classes = cls.predict_proba(fold['trial']['source'])
        actual_classes = lb.transform(fold['trial']['target'])

        ll = logloss(actual_classes, pred_classes)
        ll2 = log_loss(actual_classes, pred_classes)
        loss.append(ll2)
        print ll, sum(ll), ll2

        best = (loss[idx] if loss[idx] <  best else best)
        worst = (loss[idx] if loss[idx] > worst else best)
        idx += 1

    return {'best': str(best).zfill(15), 'worst': str(worst).zfill(15), 'mean': str(mean(loss)).zfill(15),
            'variance': str(var(loss)).zfill(15), 'median': str(median(loss)).zfill(15),
            'average': str(average(loss)).zfill(15), 'log-loss': loss}


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
    '''
    cls_col = ['AgeuponOutcome', 'IsNamed', 'IsIntact', 'BreedA', 'AnimalType']
    cls_col = ['AnimalID', 'Name', 'DateTime', 'OutcomeType', 'OutcomeSubtype',
            'AnimalType', 'SexuponOutcome', 'AgeuponOutcome', 'Breed', 'Color',
            'IsNamed', 'Date', 'Time', 'Gender', 'IsIntact', 'AgeInDays',
            'AgeInCategory', 'BreedA', 'BreedB', 'IsMix', 'ColorA', 'ColorB',
            'OutcomeTypeEncoded', 'Adoption', 'Died', 'Euthanasia',
            'Return_to_owner', 'Transfer']
    cls_col = ['ID', 'Name', 'DateTime', 'AnimalType', 'SexuponOutcome',
    'AgeuponOutcome', 'Breed', 'Color', 'IsNamed', 'Date', 'Time', 'Gender',
    'IsIntact', 'AgeInDays', 'AgeInCategory', 'BreedA', 'BreedB', 'IsMix',
    'ColorA', 'ColorB']
    '''
    cls_col = ['DateTime', 'AnimalType', 'SexuponOutcome',
        'AgeuponOutcome', 'Breed', 'Color', 'IsNamed', 'Date', 'Time', 'Gender',
        'IsIntact', 'AgeInDays', 'AgeInCategory', 'BreedA', 'BreedB', 'IsMix',
        'ColorA', 'ColorB']
    # preparing data to classify
    print "Preparing data"
    train_outcome = train_set['OutcomeType']

    train_set = train_set[cls_col]
    test_set = test_set[cls_col]
    train_set = prepare_data(train_set)
    test_set = prepare_data(test_set)

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
    #cls.fit(train_set.values[0::,0:-5:], train_outcome.values[0::, -5::])
    folds = data_fold(train_set, train_outcome, 10)
    r = learn_and_test(cls, folds)
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
