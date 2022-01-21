import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np

DATA_DIR = 'data/'

def get_data_file(data_name):
    return os.path.join(DATA_DIR, '%s.csv' % data_name)

def process_data(data, do_print=True, all_continuous_mutable=False):
    if data == "compas":
        return process_compas_data(do_print=do_print)
    elif data == "bail":
        return process_bail_data(do_print=do_print)
    elif data == "adult":
        return process_adult_data(do_print=do_print, all_continuous_mutable=all_continuous_mutable)
    elif data == "german":
        return process_german_data(do_print=do_print)
    else:
        raise NotImplementedError

def process_compas_data(do_print=True):
    """
    processes normalized adult dataset in DATA_DIR

    :returns: tuple (adult_X, adult_y, adult_actionable_indices, adult_categorical_features, adult_categorical_names)
        adult_categorical_features: indices of categorical features in the processed dataset
    """

    data_file = get_data_file("compas-scores-two-years")

    # load and process data
    compas_df = pd.read_csv(data_file, index_col=0)
    compas_df = compas_df.loc[(compas_df['days_b_screening_arrest'] <= 30) &
                              (compas_df['days_b_screening_arrest'] >= -30) &
                              (compas_df['is_recid'] != -1) &
                              (compas_df['c_charge_degree'] != "O") &
                              (compas_df['score_text'] != "NA")]

    compas_df['length_of_stay'] = (pd.to_datetime(compas_df['c_jail_out']) - pd.to_datetime(compas_df['c_jail_in'])).dt.days
    compas_X = compas_df[['age', 'c_charge_degree', 'race', 'sex', 'priors_count', 'length_of_stay', 'days_b_screening_arrest']]


    compas_X['isMale'] = compas_X.apply(lambda row: 1 if 'Male' in row['sex'] else 0, axis=1)
    compas_X['isCaucasian'] = compas_X.apply(lambda row: 1 if 'Caucasian' in row['race'] else 0, axis=1)
    compas_X['c_charge_degree_F'] = compas_X.apply(lambda row: 1 if 'F' in row['c_charge_degree'] else 0, axis=1)
    compas_X = compas_X.drop(['sex', 'race', 'c_charge_degree'], axis=1)

    # if person has high score give them the _negative_ model outcome
    compas_df['label'] = compas_df.apply(lambda row: 1.0 if row['two_year_recid'] == 0 else 0.0, axis=1)
    compas_y = compas_df['label']

    compas_categorical_features = [4, 5, 6]

    columns = compas_X.columns
    compas_categorical_names = [columns[i] for i in compas_categorical_features] 
    means = [0 for i in range(len(compas_X.iloc[0]))]
    std = [1 for i in range(len(compas_X.iloc[0]))]

    # normalize continuous features
    for col_idx, col in enumerate(compas_X.columns):
        if col not in compas_categorical_names:
            means[col_idx] = compas_X[col].mean(axis=0)
            std[col_idx] = compas_X[col].std(axis=0)
            compas_X[col] = (compas_X[col] - compas_X[col].mean(axis=0)) / compas_X[col].std(axis=0)


    compas_actionable_features = ["priors_count"]
    compas_actionable_indices = [idx for idx, col in enumerate(compas_X.columns) if col in compas_actionable_features]
    assert len(compas_actionable_features) == len(compas_actionable_indices)

    compas_increasing_actionable_features = []
    compas_increasing_actionable_indices = [idx for idx, col in enumerate(compas_X.columns) if col in compas_increasing_actionable_features]

    compas_decreasing_actionable_features = []
    compas_decreasing_actionable_indices = [idx for idx, col in enumerate(compas_X.columns) if col in compas_decreasing_actionable_features]

    if do_print:
        print("processing compas data...")
        print("compas actionable features: ", compas_actionable_features)
        print("compas actionable indices: ", compas_actionable_indices)

        print("compas increasing actionable features: ", compas_increasing_actionable_features)
        print("compas increasing actionable indices: ", compas_increasing_actionable_indices)

        print("compas decreasing actionable features: ", compas_decreasing_actionable_features)
        print("compas decreasing actionable indices: ", compas_decreasing_actionable_indices)

    feature_names = compas_X.columns

    return compas_X, compas_y, compas_actionable_indices, compas_increasing_actionable_indices, compas_decreasing_actionable_indices, compas_categorical_features, compas_categorical_names, feature_names, means, std


def process_adult_data(do_print=True, all_continuous_mutable=False):
    """
    processes normalized adult dataset in DATA_DIR

    :returns: tuple (adult_X, adult_y, adult_actionable_indices, adult_categorical_features, adult_categorical_names)
        adult_categorical_features: indices of categorical features in the processed dataset
    """

    data_file = get_data_file("adult")

    # load and process data
    adult_df = pd.read_csv(data_file).reset_index(drop=True)
    adult_df.columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex',\
                                              'capital-gain','capital-loss','hours-per-week','native-country','label']

    adult_df = adult_df.dropna()
    adult_df['isWhite'] = adult_df.apply(lambda row: 1 if 'White' in row['race'] else 0, axis=1)
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
    adult_categorical_features = [5, 6, 7, 8]

    adult_X.columns = adult_X.columns.str.replace("_", "-")

    columns = adult_X.columns
    adult_categorical_names = [columns[i] for i in adult_categorical_features] 

    means = [0 for i in range(len(adult_X.iloc[0]))]
    std = [1 for i in range(len(adult_X.iloc[0]))]

    # normalize continuous features
    for col_idx, col in enumerate(adult_X.columns):
        if col not in adult_categorical_names:
            means[col_idx] = adult_X[col].mean(axis=0)
            std[col_idx] = adult_X[col].std(axis=0)            
            adult_X[col] = (adult_X[col] - adult_X[col].mean(axis=0)) / adult_X[col].std(axis=0)


    if all_continuous_mutable:
        print("making all continuous features actionable...")

        adult_actionable_features = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
        adult_actionable_indices = [idx for idx, col in enumerate(adult_X.columns) if col in adult_actionable_features]

        assert len(adult_actionable_indices) == len(adult_actionable_features)

        adult_increasing_actionable_features = []
        adult_increasing_actionable_indices = [idx for idx, col in enumerate(adult_X.columns) if col in adult_increasing_actionable_features]

        adult_decreasing_actionable_features = []
        adult_decreasing_actionable_indices = [idx for idx, col in enumerate(adult_X.columns) if col in adult_decreasing_actionable_features]

    else:
        adult_actionable_features = ["education-num", "hours-per-week"]
        adult_actionable_indices = [idx for idx, col in enumerate(adult_X.columns) if col in adult_actionable_features]

        assert len(adult_actionable_indices) == len(adult_actionable_features)

        adult_increasing_actionable_features = ["education-num"]
        adult_increasing_actionable_indices = [idx for idx, col in enumerate(adult_X.columns) if col in adult_increasing_actionable_features]

        adult_decreasing_actionable_features = []
        adult_decreasing_actionable_indices = [idx for idx, col in enumerate(adult_X.columns) if col in adult_decreasing_actionable_features]


    feature_names = adult_X.columns

    if do_print:
        print("processing adult data...")
        print("adult actionable features: ", adult_actionable_features)
        print("adult actionable indices: ", adult_actionable_indices)
        print("adult increasing actionable features: ", adult_increasing_actionable_features)
        print("adult increasing actionable indices: ", adult_increasing_actionable_indices)
        print("adult decreasing actionable features: ", adult_decreasing_actionable_features)
        print("adult decreasing actionable indices: ", adult_decreasing_actionable_indices)

    return adult_X, adult_y, adult_actionable_indices, adult_increasing_actionable_indices, adult_decreasing_actionable_indices, adult_categorical_features, adult_categorical_names, feature_names, means, std


def process_bail_data(subset="train", given_means=None, given_std=None, do_print=True):
    """
    processes normalized bail dataset in DATA_DIR (only from bail_train)

    :returns: tuple (bail_X, bail_y, bail_actionable_indices, bail_categorical_features, bail_categorical_names)
        bail_categorical_features: indices of categorical features in the processed dataset

    """

    data_file = get_data_file("bail_" + subset)

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
    bail_df['label'] = bail_df.apply(lambda row: 1.0 if row['RECID'] == 0 else 0.0, axis=1)

    bail_df = bail_df.dropna()
    bail_X = bail_df.copy()

    bail_y = bail_X['label']
    bail_X = bail_X.drop(['RECID', 'label', 'TIME', 'FILE'], axis=1)

    # define the categorical features
    bail_categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    columns = bail_X.columns
    bail_categorical_names = [columns[i] for i in bail_categorical_features] 

    means = [0 for i in range(len(bail_X.iloc[0]))]
    std = [1 for i in range(len(bail_X.iloc[0]))]


    # normalize continuous features
    for col_idx, col in enumerate(bail_X.columns):
        if col not in bail_categorical_names:
            means[col_idx] = bail_X[col].mean(axis=0)
            std[col_idx] = bail_X[col].std(axis=0)    
            feature_mean = means[col_idx] if given_means is None else given_means[col_idx]
            feature_std = std[col_idx] if given_std is None else given_std[col_idx]

            bail_X[col] = (bail_X[col] - feature_mean) / feature_std
    

    bail_actionable_features = ["SCHOOL", "RULE"]
    bail_actionable_indices = [idx for idx, col in enumerate(bail_X.columns) if col in bail_actionable_features]

    assert len(bail_actionable_indices) == len(bail_actionable_features)

    # bail_actionable_indices = [11, 12, 15]

    bail_increasing_actionable_features = ["SCHOOL"]
    bail_increasing_actionable_indices = [idx for idx, col in enumerate(bail_X.columns) if col in bail_increasing_actionable_features]

    bail_decreasing_actionable_features = []
    bail_decreasing_actionable_indices = [idx for idx, col in enumerate(bail_X.columns) if col in bail_decreasing_actionable_features]

    feature_names = bail_X.columns

    if given_means is not None:
        means = given_means
    if given_std is not None:
        std = given_std

    if do_print:
        print("processing bail data...")
        print("subset: ", subset)
        
        print("bail actionable features: ", bail_actionable_features)
        print("bail actionable indices: ", bail_actionable_indices)
        
        print("bail increasing actionable features: ", bail_increasing_actionable_features)
        print("bail increasing actionable indices: ", bail_increasing_actionable_indices)

        
        print("bail decreasing actionable features: ", bail_decreasing_actionable_features)
        print("bail decreasing actionable indices: ", bail_decreasing_actionable_indices)

    return bail_X, bail_y, bail_actionable_indices, bail_increasing_actionable_indices, bail_decreasing_actionable_indices, bail_categorical_features, bail_categorical_names, feature_names, means, std


def process_german_data(do_print=True):
    """
    processes normalized german dataset in DATA_DIR

    :returns: tuple (german_X, german_y, german_actionable_indices, german_categorical_features, german_categorical_names)
        german_categorical_features: indices of categorical features in the processed dataset
    """

    data_file = get_data_file("german")
    german_df = pd.read_csv(data_file)

    german_categorical_names = ["personal_status_sex"]
    features_to_include = ["duration", "amount", "age"] + german_categorical_names
    target = "credit_risk"

    german_df = german_df.dropna()
    german_df['label'] = german_df.apply(lambda row: 1.0 if row[target] == 0 else 0.0, axis=1)
    german_y = german_df['label']
    german_X = german_df.drop(columns=[c for c in list(german_df) if c not in features_to_include])

    german_actionable_features = ["amount"]
    german_categorical_features = [idx for idx, feat in enumerate(german_X.columns) if feat in german_categorical_names]


    means = [0 for i in range(len(german_X.iloc[0]))]
    std = [1 for i in range(len(german_X.iloc[0]))]

     # normalize continuous features
    for col_idx, col in enumerate(german_X.columns):
        if col not in german_categorical_names:
            means[col_idx] = german_X[col].mean(axis=0)
            std[col_idx] = german_X[col].std(axis=0)            
            german_X[col] = (german_X[col] - german_X[col].mean(axis=0)) / german_X[col].std(axis=0)

    #One-hot encode categorical features
    german_X = pd.get_dummies(german_X, columns=german_categorical_names)


    german_actionable_features = ["age", "amount"]
    german_actionable_indices = [idx for idx, col in enumerate(german_X.columns) if col in german_actionable_features]

    assert len(german_actionable_indices) == len(german_actionable_features)

    feature_names = german_X.columns

    german_increasing_actionable_features = ["age"]
    german_increasing_actionable_indices = [idx for idx, col in enumerate(german_X.columns) if col in german_increasing_actionable_features]

    german_decreasing_actionable_features = []
    german_decreasing_actionable_indices = [idx for idx, col in enumerate(german_X.columns) if col in german_decreasing_actionable_features]

    if do_print:
        print("processing german data...")
        print("german actionable features: ", german_actionable_features)
        print("german actionable indices: ", german_actionable_indices)
        print("german increasing actionable features: ", german_increasing_actionable_features)
        print("german increasing actionable indices: ", german_increasing_actionable_indices)
        print("german decreasing actionable features: ", german_decreasing_actionable_features)
        print("german decreasing actionable indices: ", german_decreasing_actionable_indices)
        print(german_X)

    return german_X, german_y, german_actionable_indices, german_increasing_actionable_indices, german_decreasing_actionable_indices, german_categorical_features, german_categorical_names, feature_names, means, std



def get_data(X, y, X_test = None, y_test = None, val_size = 0.2, test_size=500, random_state=0): 
    """
    splits processed data into train/val/test datasets

    :param X: features
    :param y: labels
    :returns: dictionary with data

    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state = random_state)
    if X_test is None and y_test is None:
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=test_size, random_state = random_state)
    else:
        print("test set provided")
        # randomly sample 500 instances
        _, X_test, _, y_test = train_test_split(X_test, y_test, test_size=test_size, random_state = random_state)

    data = {
        'X_train': X_train,
        'y_train': y_train,

        'X_val': X_val,
        'y_val': y_val,

        'X_test': X_test,
        'y_test': y_test
    }
    
    return data    

def write_data(data, output_dir):
    """
    Writes data dataframes to csv

    :param data: data dictionary created by get_data
    :param output_dir: output directory of experiment. a subdirectory 'data' will hold the data

    """

    data_dir = output_dir + "data/"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    (data['X_train']).to_csv(data_dir + 'X_train.csv', index_label='index')
    (data['X_val']).to_csv(data_dir + 'X_val.csv', index_label='index')
    (data['X_test']).to_csv(data_dir + 'X_test.csv', index_label='index')


    (data['y_train']).to_csv(data_dir + 'y_train.csv', index_label='index')
    (data['y_val']).to_csv(data_dir + 'y_val.csv', index_label='index')
    (data['y_test']).to_csv(data_dir + 'y_test.csv', index_label='index')

def read_data(output_dir):
    """
    reads data dataframes to csv

    :param output_dir: output directory of experiment. will look inside a subdirectory 'data'
    :returns: dictionary with data if files exist; None otherwise

    """

    data_dir = output_dir + "data/"

    if not os.path.exists(data_dir):
        return None

    data = {}

    (data['X_train']) = pd.read_csv(data_dir + 'X_train.csv', dtype=np.float64, index_col = 'index')
    (data['X_val']) = pd.read_csv(data_dir + 'X_val.csv', dtype=np.float64, index_col = 'index')
    (data['X_test']) = pd.read_csv(data_dir + 'X_test.csv', dtype=np.float64, index_col = 'index')

    (data['y_train']) = pd.read_csv(data_dir + 'y_train.csv', dtype=np.float64, index_col = 'index')['label']
    (data['y_val']) = pd.read_csv(data_dir + 'y_val.csv', dtype=np.float64, index_col = 'index')['label']
    (data['y_test']) = pd.read_csv(data_dir + 'y_test.csv', dtype=np.float64, index_col = 'index')['label']

    for x in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']:
        data[x].index = data[x].index.astype(int)

    return data

