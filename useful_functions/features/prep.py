from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def dropnans_newidx(data):
    """drops rows with NaNs and resets index"""
    nr_nans = data.isnull().sum().sum()
    data.dropna(axis=0,inplace=True)
    data.reset_index(inplace=True,drop=True)
    print(f"{nr_nans} rows with NaNs were removed from data. "
          f"{data.isnull().sum().sum()} NaNs still included in data")
    return data

def split_train_test(data,targetname,percentages):
    """percentages = list with percentage of train and test,e.g. [0.8,0.2]"""
    train_perc = percentages[0]
    test_perc = percentages[1]
    if train_perc + test_perc != 1:
        print("percentages don't sum up to 1! calculate again :) ")
        return None
    else:
        X_train,X_test,y_train,y_test = train_test_split(data.loc[:,data.columns != targetname],data[targetname], test_size=0.2, random_state=42)
        return [X_train,X_test,y_train,y_test]

def split_train_val_test(data,targetname,percentages,stratify=False,random_state = None):
    """percentages = list with percentage of train,val,test. e.g. [0.6,0.2,0.2]
    targetname = column name of target (string). data is a dataframe"""
    train_perc = percentages[0]
    test_perc = percentages[2]
    val_perc = test_perc/(train_perc + test_perc)
    random_state = random_state
    if sum(percentages) > 1:
        print("percentages sum up to more than 1! calculate again :) ")
        return None
    else:
        if stratify == True:
            X_train, X_test, y_train, y_test = train_test_split(data.loc[:,data.columns != targetname], data[targetname], test_size=test_perc, random_state=random_state, stratify = data[targetname])
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_perc, random_state=random_state, stratify = y_train)
            return [X_train, X_val, X_test, y_train, y_val, y_test]
        else:
            X_train, X_test, y_train, y_test = train_test_split(data.loc[:,data.columns != targetname], data[targetname], test_size=test_perc, random_state=random_state)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_perc, random_state=random_state)
            return [X_train, X_val, X_test, y_train, y_val, y_test]

def convert_cats(data,colname):
    # ONE-HOT-ENCODE category
    # transform to numerical
    le = preprocessing.LabelEncoder()
    coldata = le.fit_transform(data[colname])
    # transform to binary
    ohe = preprocessing.OneHotEncoder()
    coldata_arr = ohe.fit_transform(coldata.reshape(-1, 1)).toarray()
    coldata_df = pd.DataFrame(data = coldata_arr, columns = list(le.classes_))
    data.drop(colname,axis=1,inplace=True)
    return pd.concat([data,coldata_df],axis=1)



def scale_to_train(splitdata_list,cont_cols,scaler_type):
    splitdata = splitdata_list
    if scaler_type == "standard":
        scaler = preprocessing.StandardScaler()
    elif scaler_type == "minmax":
        scaler = preprocessing.MinMaxScaler()
    train_df_cont = splitdata[0].iloc[:, cont_cols]
    scaler.fit(train_df_cont)
    splitdata[0].iloc[:, cont_cols] = scaler.transform(train_df_cont)
    splitdata[1].iloc[:, cont_cols] = scaler.transform(splitdata[1].iloc[:, cont_cols])
    return splitdata



