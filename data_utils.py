import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np

DATA_DIR = 'data/'

def get_data_file(data_name):
    return os.path.join(DATA_DIR, '%s.csv' % data_name)

def process_adult_data():
    """
    Processes normalized adult dataset in DATA_DIR

    :returns: tuple (adult_X, adult_y, adult_actionable_indices, adult_categorical_features, adult_categorical_names)
        adult_categorical_features: indices of categorical features in the processed dataset
    """

    data_file = get_data_file("adult")

    # load and process data
    adult_df = pd.read_csv(data_file).reset_index(drop=True)
    adult_df.columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex',\
                                              'capital-gain','capital-loss','hours-per-week','native-country','label']

    adult_df = adult_df.dropna()
    adult_df['native-country-United-States'] = adult_df.apply(lambda row: 1 if 'United-States' in row['native-country'] else 0, axis=1)
    adult_df['marital-status-Married'] = adult_df.apply(lambda row: 1 if 'Married' in row['marital-status'] else 0, axis=1)
    adult_df['isMale'] = adult_df.apply(lambda row: 1 if 'Male' in row['sex'] else 0, axis=1)
    adult_df = adult_df.drop(['native-country', 'marital-status', 'relationship'], axis=1)

    adult_df.columns = adult_df.columns.str.replace(' ', '')
    adult_df = adult_df.drop(['fnlwgt', 'education', 'occupation'], axis=1)
    adult_X = adult_df.drop('label', axis=1)
    adult_X = adult_X.drop(['workclass', 'race', 'sex'], axis=1)
    adult_y = adult_df['label'].replace(' <=50K', 0.0)
    adult_y = adult_y.replace(' >50K', 1.0)

    # define the categorical features
    adult_categorical_features = [5, 6, 7]

    adult_X.columns = adult_X.columns.str.replace("_", "-")

    columns = adult_X.columns
    adult_categorical_names = [columns[i] for i in adult_categorical_features] 

    # normalize continuous features
    for col in adult_X.columns:
        if col not in adult_categorical_names:
            adult_X[col] = (adult_X[col] - adult_X[col].mean(axis=0)) / adult_X[col].std(axis=0)


    adult_actionable_indices = [1, 2, 4]
    return adult_X, adult_y, adult_actionable_indices, adult_categorical_features, adult_categorical_names


def process_bail_data():
    """
    Processes normalized bail dataset in DATA_DIR (only from bail_train)

    :returns: tuple (bail_X, bail_y, bail_actionable_indices, bail_categorical_features, bail_categorical_names)
        bail_categorical_features: indices of categorical features in the processed dataset

    """

    data_file = get_data_file("bail_train")

    # load and process data
    bail_df = pd.read_csv(data_file)
        
    '''
    From the data documentation: 

    If (FILE = 3), the value of ALCHY is recorded as zero, but is meaningless.
    If (FILE = 3), the value of JUNKY is recorded as zero, but is meaningless.
    PRIORS: the value -9 indicates that this information is missing.
    SCHOOL: the value zero indicates that this information is missing.
    For individuals for whom RECID equals zero, the value of TIME is meaningless.

    We set these values to nan so they do not affect binning

    https://www.ncjrs.gov/pdffiles1/Digitization/115306NCJRS.pdf
    '''

    bail_df.loc[bail_df["FILE"] == 3, "ALCHY"] = np.nan
    bail_df.loc[bail_df["FILE"] == 3, "JUNKY"] = np.nan
    bail_df.loc[bail_df["PRIORS"] == -9, "PRIORS"] = np.nan
    bail_df.loc[bail_df["SCHOOL"] == 0, "SCHOOL"] = np.nan

    bail_df = bail_df.dropna()
    bail_X = bail_df.copy()

    bail_y = bail_X['RECID']
    bail_X = bail_X.drop(['RECID', 'TIME', 'FILE'], axis=1)

    # define the categorical features
    bail_categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    columns = bail_X.columns
    bail_categorical_names = [columns[i] for i in bail_categorical_features] 


    # normalize continuous features
    for col in bail_X.columns:
        if col not in bail_categorical_names:
            bail_X[col] = (bail_X[col] - bail_X[col].mean(axis=0)) / bail_X[col].std(axis=0)


    bail_actionable_indices = [11, 12, 15]

    return bail_X, bail_y, bail_actionable_indices, bail_categorical_features, bail_categorical_names

def get_data(X, y, val_size = 0.1): 
    """
    Splits processed data into train/val/test datasets

    :param X: features
    :param y: labels
    :returns: dictionary with data

    """

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)
    
    data = {
        'X_train': X_train,
        'y_train': y_train,

        'X_val': X_val,
        'y_val': y_val,

        'X_test': X_test,
        'y_test': y_test
    }
    
    return data    

