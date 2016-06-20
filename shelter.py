import pandas as pd
import numpy as np
import pdb
import sys

from sklearn import preprocessing

from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

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
    in_file = in_file.drop(drop_attr, axis=1)

    #return prepare_data(in_file)
    return in_file


def main(classifier_name):
    # relative to each machine
    data_path = '../../data/'
    train_file = data_path + 'transformed_train.csv'
    test_file = data_path + 'transformed_test.csv'


    # loading test set
    test_set = load_file(test_file, drop_attr=['ID'])
    # loading train set
    train_set = load_file(train_file, drop_attr=['AnimalID', 'OutcomeSubtype'])

    ''' OLD
    train_outcome = train_set.OutcomeType
    #train_outcome = train_set.values[0::,-5::]
    '''
    '''
    lb = preprocessing.LabelBinarizer()
    #lb = preprocessing.LabelEncoder()
    lb.fit(['Adoption','Died','Euthanasia','Return_to_owner','Transfer'])
    train_outcome = lb.transform(train_set['OutcomeType'])
    '''
    # won't be used anymore
    train_outcome = train_set['OutcomeType']
    train_set.drop(['OutcomeType'], inplace=True, axis=1)

    # preparing data to classify
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

    # feature selection
    cls_col = ['AgeuponOutcome', 'IsNamed', 'IsIntact', 'BreedA', 'AnimalType']
    train_set = pd.DataFrame(train_set, columns=cls_col).values
    test_set = pd.DataFrame(test_set, columns=cls_col).values

    print("Classifying")
    cls.fit(train_set, train_outcome)
    predictions = cls.predict_proba(test_set)

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
