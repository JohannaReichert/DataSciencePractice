import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
#http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate

def save_errors(test,predicted):
    rates = pd.DataFrame(np.zeros((1,5)),columns =["MSE","RMSE","R2","RMSE % of mean","Cali"])
    rates["MSE"] = mean_squared_error(test, predicted)
    rates["RMSE"] = np.sqrt(((test - predicted) ** 2).mean())
    rates["R2"] = r2_score(test, predicted)
    rates["RMSE % of mean"] = np.sqrt(((test - predicted) ** 2).mean()) / test.mean()
    rates["Cali"] = predicted.mean() / test.mean()
    return rates


def print_errors(test, predicted):
    print("MSE: ", mean_squared_error(test, predicted))
    print("RMSE: ", np.sqrt(((test - predicted) ** 2).mean()))
    print("R2: ", r2_score(test, predicted))
    print("RMSE % of mean:", np.sqrt(((test - predicted) ** 2).mean()) / test.mean())
    print("Calibration:", predicted.mean() / test.mean())
    return None


def cross_vali_rmse(estimator,x,y,cv=10):
    scores = cross_val_score(estimator, x, y, scoring="neg_mean_squared_error", cv=cv)
    rmse_scores = np.sqrt(-scores)
    return rmse_scores.mean(),rmse_scores.std()

def cross_vali_errors_classification(estimator,x,y,cv=10):
    scores = cross_validate(estimator, x, y, scoring=('accuracy','roc_auc','r2', 'neg_mean_squared_error'),cv=cv,return_train_score=False)
    return scores