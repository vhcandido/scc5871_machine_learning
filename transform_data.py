from datetime import datetime as dt
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
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


def get_breedA(breed):
    b = breed.split('/')
    return b[0].split()[0]

def get_breedB(breed):
    b = breed.split('/')
    if len(b) > 1:
        return b[1].split()[0]
    return 'None'

def get_age_cat(age):
    if age <= 365:
        return 'Baby'
    else:
        return 'Adult'

def datetime_to_date(d):
    epoch = dt.utcfromtimestamp(0)
    d = d.split(' ')[0]
    date = dt.strptime(d, '%Y-%m-%d')
    return (date-epoch).days

def datetime_to_time(d):
    epoch = dt.utcfromtimestamp(0)
    d = d.split(' ')[1]
    time = dt.strptime('1970-01-01 '+d, '%Y-%m-%d %H:%M:%S')
    return (time-epoch).total_seconds()

def main(dataset, transf, out_file):


    if transf['name_to_isnamed']:
        dataset['IsNamed'] = dataset.Name.notnull()

    if transf['datetime_split']:
        dataset['Date'] = dataset.DateTime.apply(datetime_to_date)
        dataset['Date'] -= min(dataset['Date'])
        dataset['Time'] = dataset.DateTime.apply(datetime_to_time)

    if transf['datetime_to_sec']:
        dataset['DateTime'] = dataset.DateTime.apply(get_datetime_seconds)
        #dataset['DateTime'] -= min(dataset['DateTime'])

    if transf['sex_to_gender_isintact']:
        dataset['Gender'] = dataset.SexuponOutcome.apply(get_gender)
        dataset['IsIntact'] = dataset.SexuponOutcome.apply(get_isintact)
        #dataset.drop(['SexuponOutcome'], inplace=True, axis=1)

    if transf['name_to_sex']:
        if transf['sex_to_gender_isintact']:
            d = {'Hickory':'Male','Ice':'Male','Poodle':'Female','Fluffy':'Male','Pepy':'Female','Ursula':'Female','Grace':'Female','Chris':'Male','Daisy':'Female','Husky':'Male','Zara':'Female','Jan':'Male','Tinkerbell':'Female','Pinto':'Male','Karma':'Female','Taco':'Male','Ford':'Male','Diego':'Male','Teri':'Female','Beans':'Female','Precious':'Female','Maple':'Female','Bruce':'Male','Brown Dog':'Male','Packard':'Male','Aj':'Male','Nick':'Male','Spirit':'Male','Idris':'Male','Ulrika':'Female','Cyprus':'Unknown','Doodle':'Female','Boo Boo':'Female','Jd':'Male','Ash':'Male','K.C.':'Male','Lando':'Male','Uliana':'Female','Sissy':'Female','Moon':'Female','Dexter':'Male','Oliver':'Male','Chula':'Female','Serrano':'Unknown','Ulysses':'Male','Birch':'Unknown','Pumkin':'Unknown','Lucky':'Male','Oak':'Male','Jordan':'Male','Jamie':'Male','Monkey':'Unknown','Cedar':'Female'}
            r = dataset['Name'].loc[dataset['Gender'] == 'Unknown'].loc[dataset['Name'].isnull()==False].map(d)
            for i in r.index:
                dataset.iloc[i, dataset.columns.get_loc('Gender')] = r[i]

    if transf['age_to_days']:
        dataset['AgeInDays'] = dataset.AgeuponOutcome.apply(get_age_days)
        #dataset.loc[dataset['AgeuponOutcome'].isnull(), 'AgeuponOutcome'] = dataset['AgeuponOutcome'].median()

    if transf['age_to_categorical']:
        dataset['AgeInCategory'] = dataset.AgeuponOutcome.apply(get_age_days)
        dataset['AgeInCategory'] = dataset.AgeInCategory.apply(get_age_cat)


    if transf['breed_to_AB']:
        dataset['BreedA'] = dataset.Breed.apply(get_breedA)
        dataset['BreedB'] = dataset.Breed.apply(get_breedB)
        #dataset.drop(['Breed'], inplace=True, axis=1)

    if transf['breed_to_mix_pure']:
        dataset['IsMix'] = dataset.Breed.apply(get_mix_breed)

        if transf['breed_to_AB']:
            # remove mix from breedA and B
            dataset['BreedA'] = dataset.BreedA.apply(lambda(s): s.replace(' Mix',''))
            dataset['BreedB'] = dataset.BreedB.apply(lambda(s): s.replace(' Mix',''))

    if transf['color_to_AB']:
        dataset['ColorA'] = dataset.Color.apply(get_colorA)
        dataset['ColorB'] = dataset.Color.apply(get_colorB)
        #dataset.drop(['Color'], inplace=True, axis=1)

    if transf['outcome_encode']:
        le = LabelEncoder()
        target = ['Adoption', 'Died', 'Euthanasia', 'Return_to_owner',
                'Transfer']
        le.fit(target)
        dataset['OutcomeTypeEncoded'] = le.transform(dataset['OutcomeType'])
        #dataset['OutcomeTypeEncoded'] = dataset['OutcomeType'].map({'Adoption':1, 'Return_to_owner':4, 'Euthanasia':3, 'Transfer':5, 'Died':2})

    if transf['outcome_binarize']:
        lb = LabelBinarizer()
        target = ['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']
        lb.fit(target)
        out_bin = lb.transform(dataset['OutcomeType'])
        dataset = pd.concat([ dataset,
            pd.DataFrame(out_bin, columns=target) ],
            axis=1)

    # write to CSV file
    dataset.to_csv(out_file, index=False)

if __name__ == '__main__':
    # load CSV file to transform
    path = '../../data/'
    train_data = pd.read_csv(path + 'train.csv')
    test_data = pd.read_csv(path + 'test.csv')

    # transform train data
    transf = {
            'name_to_sex' : True,
            'name_to_isnamed' : True,
            'datetime_split': True,
            'datetime_to_sec' : True,
            'sex_to_gender_isintact' : True,
            'age_to_days' : False,
            'age_to_categorical' : True,
            'breed_to_mix_pure' : False,
            'breed_to_AB' : True,
            'breed_to_size' : False, # not implemented
            'color_to_AB' : True,
            'outcome_encode' : True,
            'outcome_binarize' : False
            }
    transf = dict.fromkeys(transf, True)
    main(train_data, transf, path + 'transformed_train.csv')

    # transform test data
    transf['outcome_binarize'] = False
    transf['outcome_encode'] = False
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

