# _*_coding:utf-8_*_
"""
@author: Zhaoqing Liu
@email: Zhaoqing.Liu-1@student.uts.edu.au
@date: 03/12/2020 4:42 pm
@desc:
TODO:
    1. Done - Do feature fuzzification in data_preprocessing_funcs.py
    2. Done - Add function implementing splitting criteria fuzzification in util_criterion_funcs.py
    3. Upgrade program from Python to Cython to speed up the algorithms.
"""
import os
import numpy as np
from fuzzytrees.settings import DirSave
from fuzzytrees.util_comm import get_today_str

if __name__ == '__main__':
    # df = pd.DataFrame({"id": [1, 2, 3, 4], "name": ["sam", "sam", "peter", "jack"], "number": [8, 6, 8, 2]})
    # df_after = df[["name", "number"]]
    # print(df_after.values)
    # unique_values = df["name"].unique()
    # for idx, val in enumerate(unique_values):
    #     print(idx, val)
    # print(np.size(unique_values))
    # print(np.shape(unique_values))
    # print(type(unique_values))
    # print(unique_values)
    # for unique_value in unique_values:
    #     print(unique_value)
    #     sub_df = df[df["name"] == unique_value]
    # print(type(sub_df))
    # print(type(sub_df.values))
    # print(type(sub_df["number"]))
    # print(sub_df["number"].values)
    # print(sub_df)
    # sdf = sub_df.sort_values(by="number")
    # print(sdf)

    # ds = df.groupby(["name"]).groups
    # print(ds)
    # for (name, data) in ds.items():
    #     print(name)
    #     print(type(data))
    #     print(type(data.values))
    #     print(type(df.iloc[data, :]))
    #     print(type(df.iloc[data.values, :]))

    # a = np.array([2, 8, 5])
    # print(a)
    # b = a - 1
    # print(b)

    dir = DirSave.EVAL_DATA.value + get_today_str()
    print(os.path.exists(dir))
    if not os.path.exists(dir) or not os.path.isdir(dir):
        os.makedirs(dir)

    pass

    # =======================================================================================
    # Test for mapping [0, infinity] to [0, 1]
    a = np.arange(0, 1, 0.01)
    print(a)
    b = 1 - 1 / (1 + 100 * a)
    print(b)

    b = np.log(a) - np.log(1 - a)
    print(b)
    # =======================================================================================
