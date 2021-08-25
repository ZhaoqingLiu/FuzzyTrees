"""
@author : Anjin Liu, Zhaoqing Liu
@email  : anjin.liu@uts.edu.au, Zhaoqing.Liu-1@student.uts.edu.au
"""
import os
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn import datasets

# ==================================================================================
# Functions for loading specific datasets.
# ==================================================================================

# Change it to your folder path.
from sklearn.model_selection import train_test_split

DATA_FOLDER_PATH = "./Datasets/"


# DATA_FOLDER_PATH = os.path.abspath(os.path.dirname(os.getcwd())) + '/Datasets/'


# logging.debug("+++++++++++++++++++++++++++++++++++++++++++++ %s", DATA_FOLDER_PATH)


def load_covertype():
    """
    Dua, D. & Graff, C., UCI machine learning repository [http://archive.ics.uci.edu/ml],
    Irvine, CA: University of California, School of Information and Computer Science.
    """
    df = pd.read_csv(DATA_FOLDER_PATH + 'Covertype/covtype.csv', sep=',', decimal='.')
    df.dropna(inplace=True)
    return df


def load_pokerhand():
    """
    Dua, D. & Graff, C., UCI machine learning repository [http://archive.ics.uci.edu/ml],
    Irvine, CA: University of California, School of Information and Computer Science.
    """
    df = pd.read_csv(DATA_FOLDER_PATH + 'Pokerhand/poker-hand-training-true.data', sep=',', decimal='.', header=None)
    # df = pd.read_csv(DATA_FOLDER_PATH + 'Pokerhand/poker-hand-testing.data', sep=',', decimal='.', header=None)
    return df


def load_mushroom():
    """
    Dua, D. & Graff, C., UCI machine learning repository [http://archive.ics.uci.edu/ml],
    Irvine, CA: University of California, School of Information and Computer Science.
    """
    column_names = ['class',
                    'cap-shape',
                    'cap-surface',
                    'cap-color',
                    'bruises',
                    'odor',
                    'gill-attachment',
                    'ill-spacing',
                    'gill-size',
                    'gill-color',
                    'stalk-shape',
                    'stalk-root',
                    'stalk-surface-above-ring',
                    'stalk-surface-below-ring',
                    'stalk-color-above-ring',
                    'stalk-color-below-ring',
                    'veil-type',
                    'veil-color',
                    'ring-number',
                    'vring-type',
                    'spore-print-color',
                    'population',
                    'habitat']

    df = pd.read_csv(DATA_FOLDER_PATH + 'Mushroom/agaricus-lepiota.data', sep=',', decimal='.', header=None)
    df.columns = column_names

    # (Optional for non-fuzzy FDT) Numerise all attributes.
    for column in column_names:
        y_flag = df[column].unique()
        df.loc[:, column] = df.loc[:, column].apply(lambda x: y_flag.tolist().index(x))
    class_ = df.pop('class')
    df.insert(df.shape[1], 'class', class_)

    return df


def load_vehicle():
    """
    Turing Institute Research Memorandum TIRM-87-018 "Vehicle
	Recognition Using Rule Based Methods" by Siebert,JP (March 1987)
    """
    file_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    file_list = ['xa{}.dat'.format(i) for i in file_list]
    column_names = ['COMPACTNESS',
                    'CIRCULARITY',
                    'DISTANCE CIRCULARITY',
                    'RADIUS RATIO',
                    'PR.AXIS ASPECT RATIO',
                    'MAX.LENGTH ASPECT RATIO',
                    'SCATTER RATIO',
                    'ELONGATEDNESS',
                    'PR.AXIS RECTANGULARITY',
                    'MAX.LENGTH RECTANGULARITY',
                    'SCALED VARIANCE ALONG MAJOR AXIS',
                    'SCALED VARIANCE ALONG MINOR AXIS',
                    'SCALED RADIUS OF GYRATION',
                    'SKEWNESS ABOUT MAJOR AXIS',
                    'SKEWNESS ABOUT MINOR AXIS',
                    'KURTOSIS ABOUT MINOR AXIS',
                    'KURTOSIS ABOUT MAJOR AXIS',
                    'HOLLOWS RATIO',
                    'CLASS']
    data_df_list = []
    for file in file_list:
        data = np.genfromtxt(DATA_FOLDER_PATH + 'Vehicle/{}'.format(file), dtype='unicode', delimiter=' ')
        data_df = pd.DataFrame(data)
        data_df_list.append(data_df)

    data_df_merge = pd.concat(data_df_list)
    data_df_merge.columns = column_names
    label_encoder_dict = {'opel': 0, 'saab': 1, 'bus': 2, 'van': 3}
    data_df_merge['CLASS'].replace(label_encoder_dict, inplace=True)

    for col_name in column_names:
        data_df_merge[col_name] = data_df_merge[col_name].astype(int)

    return data_df_merge


def load_waveform():
    """
    Breiman,L., Friedman,J.H., Olshen,R.A., & Stone,C.J. (1984).
    Classification and Regression Trees.  Wadsworth International
    """
    df = pd.read_csv(DATA_FOLDER_PATH + 'Waveform/waveform.data', header=None)
    return df


def load_German_credit():
    """
    Professor Dr. Hans Hofmann
    Institut f"ur Statistik und "Okonometrie
    Universit"at Hamburg
    FB Wirtschaftswissenschaften
    Von-Melle-Park 5
    2000 Hamburg 13

    """
    df = pd.read_csv(DATA_FOLDER_PATH + 'German Credit/german.data-numeric', skipinitialspace=True, sep=' ',
                     header=None)
    return df.iloc[:, :-1]


def load_chess():
    """
    (a) Database originally generated and described by Alen Shapiro.
    (b) Donor/Coder: Rob Holte (holte@uottawa.bitnet).  The database
        was supplied to Holte by Peter Clark of the Turing Institute
        in Glasgow (pete@turing.ac.uk).
    (c) Date: 1 August 1989
    """
    df = pd.read_csv(DATA_FOLDER_PATH + 'Chess/kr-vs-kp.data', header=None)

    # # Encode each column with Label-Encoding.
    # columns = df.columns
    # for column in columns:
    #     df[column] = pd.factorize(df[column])[0].astype(np.uint16)

    return df


def load_diabetes():
    data = arff.loadarff(DATA_FOLDER_PATH + 'Diabetes/dataset_37_diabetes.arff')
    df = pd.DataFrame(data[0])
    df['class'] = df['class'].str.decode('utf-8')
    class_replace_dict = {'tested_negative': 0, 'tested_positive': 1}
    df['class'].replace(class_replace_dict, inplace=True)
    return df


def load_iris():
    X, y = datasets.load_iris(return_X_y=True)
    y = np.expand_dims(y, axis=1)
    ds = np.concatenate((X, y), axis=1)
    return pd.DataFrame(ds)


def load_wine():
    X, y = datasets.load_wine(return_X_y=True)
    y = np.expand_dims(y, axis=1)
    ds = np.concatenate((X, y), axis=1)
    return pd.DataFrame(ds)


# ==================================================================================
# Constants and functions used to get datasets for experiments.
# Note: The following functions may select partial data to save experiment time.
# ==================================================================================
# Datasets (k: dataset name, v: function for getting data) on which the model is being trained.
DS_LOAD_FUNC_CLF = {"Vehicle": load_vehicle,
                    "German_Credit": load_German_credit,
                    "Diabetes": load_diabetes,
                    "Iris": load_iris,
                    "Wine": load_wine, }
# DS_LOAD_FUNC_CLF = {"Vehicle": load_vehicle}
# DS_LOAD_FUNC_CLF = {"German_Credit": load_German_credit}
# DS_LOAD_FUNC_CLF = {"Diabetes": load_diabetes}
# DS_LOAD_FUNC_CLF = {"Iris": load_iris}
# DS_LOAD_FUNC_CLF = {"Wine": load_wine}
# DS_LOAD_FUNC_CLF = {"Iris": load_iris, "Wine": load_wine}

COVERTYPE_LOAD_FUNC_CLF = {"Covertype": load_covertype}

ALL_DS_LOAD_FUNC_CLF = {"Covertype": load_covertype,
                        "Pokerhand": load_pokerhand,
                        "Mushroom": load_mushroom,
                        "Waveform": load_waveform,
                        }


def load_data_clf(ds_name):
    """
    Load the data for classification by the specified dataset name.

    Parameters
    ----------
    ds_name : str

    Returns
    -------
    data : DataFrame
    """
    ds_load_func = None

    if ds_name in DS_LOAD_FUNC_CLF.keys():
        ds_load_func = DS_LOAD_FUNC_CLF[ds_name]

    return None if ds_load_func is None else ds_load_func()
