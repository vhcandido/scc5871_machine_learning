import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Extracted from https://www.kaggle.com/wiki/LogarithmicLoss
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def load_file(path, drop_attr=[]):
    # loads CSV file
    in_file = pd.read_csv(path)

    # drop selected columns
    in_file = in_file.drop(drop_attr, axis=1)

    return in_file


def main():
    # relative to each machine
    data_path = '../../data/'
    train_file = data_path + 'transformed_train.csv'
    test_file = data_path + 'transformed_test.csv'

    # loading train set
    train_set = load_file(train_file, drop_attr=['AnimalID', 'OutcomeSubtype'])
    train_outcome = train_set.OutcomeType
    train_set.drop(['OutcomeType'], inplace=True, axis=1)
    # loading test set
    test_set = load_file(test_file, drop_attr=['ID'])


    # creating classifier
    #cls = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    cls = RandomForestClassifier(n_estimators = 400, max_features='auto')
    cls.fit(train_set.values, train_outcome.values)

    predictions = cls.predict_proba(test.values)

    # saving output
    pd.DataFrame(predictions,columns=['Adoption','Died','Euthanasia','Return_to_owner','Transfer'])
    output.columns.names = ['ID']
    output.index.names = ['ID']
    output.index += 1
    output.to_csv("out.csv")




if __name__ == '__main__':
    main()
