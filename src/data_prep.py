import os
import pathlib

import pandas as pd
from sklearn import preprocessing


def get_clean_data():
    file_path = os.path.join(pathlib.Path().resolve(), "data.csv")
    data = pd.read_csv(file_path)

    # remove shitty crypto
    data = data.drop(['Cardano Price', 'Bitcoin Cash Price'], axis=1)

    # remove where BTC price == 0.0
    data = data[data["BTC Price"] != 0.0]

    # and revert to follow linear time
    data = data.iloc[::-1]

    data = data[:-1]

    # fill forwards NAN and drop last row since that is NAN
    data = data.fillna(method='ffill')

    # drop Eth column where we have NAN
    data = data[data['Ethereum Price'].notna()]
    data.reset_index(drop=True, inplace=True)
    # print(data[data.isna().any(axis=1)])
    # print(len(data))

    return data


def normalise_data(X):
    df = pd.DataFrame(preprocessing.normalize(X, norm='l2'), columns=X.columns)
    return df
