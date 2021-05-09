# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:15:33 2020
@author: Anjin Liu
@email: anjin.liu@uts.edu.au

=============================================================================
@author: Zhaoqing Liu
@email: Zhaoqing.Liu-1@student.uts.edu.au
@date: 29/01/2021 10:32 pm
@desc: TODO: the followings
    1. Add more data set loading functions (in addition to the five data sets
        Vehicle, Waveform, German Credit, Chess, and Diabetes).
    2. Use encoder to encode the features if there are categorical variables,
        including:
        - binary features
        - low- and high-cardinality nominal features
        - low- and high-cardinality ordinal features
        - (potentially) cyclical features

"""
import io

import numpy as np
import pandas as pd
import requests
import pymysql
from scipy.io import arff

from sklearn import datasets


# ==================================================================================
# Methods to read local files:
# ds = data = pd.read_csv(r'../filename.csv')
# ds = pd.read_table(r'../filename.txt')
# ds = pd.read_excel(r'../filename.xlsx')


"""
General functions for reading online data, which are in the formats of:
    1. CSV
    2. TXT
    3. EXCEL
    4. MYSQL
"""
def read_online_csv(url):
    """
    Read a CSV-format online file.
    Parameters
    ----------
    url: raw URL
        URL should be a raw URL,
        e.g. "https://raw.githubusercontent.com/hunkim/DeepLearningZeroToAll/master/data-03-diabetes.csv",
        not "https://githubusercontent.com/hunkim/DeepLearningZeroToAll/master/data-03-diabetes.csv".

    Returns
    -------
    ds: pandas.DataFrame
    """
    # ds = pd.read_csv(url, header=None)  # Equivalent to the following two lines of code.
    s = requests.get(url).content
    ds = pd.read_csv(io.StringIO(s.decode("utf-8")), header=None)
    return ds


def read_online_txt(url):
    """
    Read a TXT-format online file.

    Parameters
    ----------
    url: raw URL
        URL should be a raw URL

    Returns
    -------
    ds: pandas.DataFrame
    """
    # ds = pd.read_csv(url, header=None)  # Equivalent to the following two lines of code.
    s = requests.get(url).content
    ds = pd.read_table(io.StringIO(s.decode("utf-8")), header=None)
    return ds


def read_online_excel(url):
    """
    Read a EXCEL-format online file.

    Parameters
    ----------
    url: raw URL
        URL should be a raw URL

    Returns
    -------
    ds: pandas.DataFrame
    """
    # ds = pd.read_csv(url, header=None)  # Equivalent to the following two lines of code.
    s = requests.get(url).content
    ds = pd.read_excel(io.StringIO(s.decode("utf-8")), header=None)
    return ds


def read_data_mysql(host, user, passwd, db, charset):
    # Connect to a database using your database information.
    conn = pymysql.connect(host=host, user=user, passwd=passwd, db=db, charset=charset)

    # Create a cursor.
    cur = conn.cursor()

    # Execute a SQL command.
    cur.execute("select * from train_data limit 100")

    # Fetch data.
    data = cur.fetchall()

    # Get columns.
    cols = cur.description

    # Commit the execution.
    conn.commit()

    # Close the cursor and the connection to the database.
    cur.close()
    conn.close()

    # Encapsulate the data.
    col = []
    for i in cols:
        col.append(i[0])
    data = list(map(list, data))
    ds = pd.DataFrame(data, columns=col)

    return ds


# ==================================================================================
# Functions for loading specific datasets.

# Change it to your folder path.
DATA_FOLDER_PATH = '/home/zhaoqliu/Datasets/'


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
    # TODO
    """
    (a) Database originally generated and described by Alen Shapiro.
    (b) Donor/Coder: Rob Holte (holte@uottawa.bitnet).  The database
        was supplied to Holte by Peter Clark of the Turing Institute
        in Glasgow (pete@turing.ac.uk).
    (c) Date: 1 August 1989
    """
    df = pd.read_csv(DATA_FOLDER_PATH + 'Chess/kr-vs-kp.data', header=None)

    return df


def load_diabetes():
    data = arff.loadarff(DATA_FOLDER_PATH + 'Diabetes/dataset_37_diabetes.arff')
    df = pd.DataFrame(data[0])
    df['class'] = df['class'].str.decode('utf-8')
    class_replace_dict = {'tested_negative': 0, 'tested_positive': 1}
    df['class'].replace(class_replace_dict, inplace=True)
    return df


def load_iris():
    # dataset = datasets.load_iris()
    # X = dataset.data
    # y = dataset.target
    #
    # y = np.expand_dims(y, axis=1)
    # ds = np.concatenate((X, y), axis=1)
    #
    # return pd.DataFrame(ds)

    return datasets.load_iris(as_frame=True).frame


def load_wine():
    # dataset = datasets.load_wine()
    # X = dataset.data
    # y = dataset.target
    #
    # y = np.expand_dims(y, axis=1)
    # ds = np.concatenate((X, y), axis=1)
    #
    # return pd.DataFrame(ds)

    return datasets.load_wine(as_frame=True).frame


if __name__ == "__main__":
    # print('Loading Vehicle')
    # data_vehicle = load_vehicle()
    # print('Loading Vehicle, shape', data_vehicle.shape)
    #
    # print('Loading Waveform')
    # data_waveform = load_waveform()
    # print('Loading Waveform, shape', data_waveform.shape)
    #
    # print('Loading German_credit')
    # data_gc = load_German_credit()
    # print('Loading German_credit, shape', data_gc.shape)
    #
    # print('Loading Chess')
    # data_chess = load_chess()
    # print('Loading Chess, shape', data_chess.shape)
    #
    # print('Loading Diabetes')
    # data_diabetes = load_diabetes()
    # print('Loading Diabetes, shape', data_diabetes.shape)
    #
    # print('Loading Iris')
    # data_iris = load_iris()
    # print('Loading Iris, shape', data_iris.shape)
    #
    # print('Loading Wine')
    # data_wine = load_wine()
    # print('Loading Wine, shape', data_wine.shape)

    url = "https://raw.githubusercontent.com/hunkim/DeepLearningZeroToAll/master/data-03-diabetes.csv"
    # print(load_online_csv(url=url))
