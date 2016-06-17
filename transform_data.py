from datetime import datetime as dt
import pandas as pd
import pdb

def get_gender(sex):
    sex = str(sex)
    if 'Male' in sex: return 'Male'
    elif 'Female' in sex: return 'Female'
    return 'Unknown'

def get_isintact(sex):
    sex = str(sex)
    if 'Intact' in sex: return 'True'
    elif 'Neutered' in sex or 'Spayed' in sex: return 'False'
    return 'Unknown'

def get_datetime_seconds(date):
    input_date = dt.strptime(date, '%Y-%m-%d %H:%M:%S')
    epoch = dt.utcfromtimestamp(0)
    return int((input_date - epoch).total_seconds())

def get_age_days(age_str):
    age = None

    try:
	    age_split = age_str.split()
    except:
        return age
    if 'year' in age_str:
        age = 365 * int(age_split[0])
    elif 'month' in age_str:
        age = 30 * int(age_split[0])
    elif 'week' in age_str:
        age = 7 * int(age_split[0])
    elif 'day' in age_str:
        age = int(age_split[0])

    return age

def get_mix_breed(breed):
    b = 'False'
    if 'Mix' in breed or '/' in breed:
        b = 'True'
    return b

def get_colorA(color):
    c = color.split('/')
    return c[0].split()[0]

def get_colorB(color):
    c = color.split('/')
    if len(c) > 1:
        return c[1].split()[0]
    return 'None'

def main(dataset, transf, out_file):

    if transf['name_to_isnamed']:
        dataset['IsNamed'] = dataset.Name.notnull()

    if transf['datetime_to_sec']:
        dataset['DateTime'] = dataset.DateTime.apply(get_datetime_seconds)

    if transf['sex_to_gender_isintact']:
        dataset['Gender'] = dataset.SexuponOutcome.apply(get_gender)
        dataset['IsIntact'] = dataset.SexuponOutcome.apply(get_isintact)
        dataset.drop(['SexuponOutcome'], inplace=True, axis=1)

    if transf['age_to_days']:
        dataset['AgeuponOutcome'] = dataset.AgeuponOutcome.apply(get_age_days)
        #dataset.loc[dataset['AgeuponOutcome'].isnull(), 'AgeuponOutcome'] = dataset['AgeuponOutcome'].median()

    if transf['breed_to_mix_pure']:
        dataset['IsMix'] = dataset.Breed.apply(get_mix_breed)

    if transf['color_to_AB']:
        dataset['ColorA'] = dataset.Color.apply(get_colorA)
        dataset['ColorB'] = dataset.Color.apply(get_colorB)
        dataset.drop(['Color'], inplace=True, axis=1)

    if transf['outcomes']:
        #dataset['OutcomeType'] = dataset['OutcomeType'].map({'Adoption':1, 'Return_to_owner':4, 'Euthanasia':3, 'Adoption':0, 'Transfer':5, 'Died':2})
        dataset['Adoption'] = (dataset['OutcomeType'] == 'Adoption')
        dataset['Died'] = (dataset['OutcomeType'] == 'Died')
        dataset['Euthanasia'] = (dataset['OutcomeType'] == 'Euthanasia')
        dataset['Return_to_owner'] = (dataset['OutcomeType'] == 'Return_to_owner')
        dataset['Transfer'] = (dataset['OutcomeType'] == 'Transfer')

    # write to CSV file
    dataset.to_csv(out_file, index=False)

if __name__ == '__main__':
    # load CSV file to transform
    path = '../../data/'
    train_data = pd.read_csv(path + 'train.csv')
    test_data = pd.read_csv(path + 'test.csv')

    # transform train data
    transf = {
            'name_to_sex' : False, # not implemented
            'name_to_isnamed' : True,
            'datetime_to_sec' : True,
            'sex_to_gender_isintact' : True,
            'age_to_days' : True,
            'breed_to_mix_pure' : True,
            'breed_to_size' : False, # not implemented
            'color_to_AB' : True,
            'outcomes' : True
            }
    main(train_data, transf, path + 'transformed_train.csv')

    # transform test data
    transf['outcomes'] = False
    main(test_data, transf, path + 'transformed_test.csv')

#
#
#
#
#
#
#
#
#
#
