from sklearn import preprocessing
import pandas as pd
import numpy as np

def square_feature(data,colname):
    colname_new = colname + "_squared"
    data_new = data.copy()
    data_new[colname_new] = data[colname]*data[colname]
    return data_new