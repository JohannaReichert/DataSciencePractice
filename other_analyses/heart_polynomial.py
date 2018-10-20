import pandas as pd
import numpy as np
import sklearn.linear_model as lm
import evaluation
import features.heart
from matplotlib import pyplot as plt
import classifier
import seaborn as sns
sns.set()

#pd.set_option('display.width', 520)
pd.set_option('display.max_columns',30)
np.set_printoptions(linewidth=520)

if __name__ == '__main__':
    data = pd.read_csv("../data/Heart.csv", sep =",")
    data['AHD'] = 1.0*(data['AHD']=='Yes')
    print(data.columns)
    print(data.iloc[0:10,0:15])
    print(data.describe())
    print(data.isnull().sum().sum()) # nr of NaNs
    cleaned_data = data.dropna(axis=0)
    cleaned_data.reset_index(inplace = True)
    print(cleaned_data.isnull().sum().sum())
    print(cleaned_data.head())
    print(cleaned_data.shape)
    cleaned_data = features.heart.convert_cats(cleaned_data,'ChestPain')
    cleaned_data = features.heart.convert_cats(cleaned_data, 'Thal')
    cleaned_data = features.heart.square_feature(cleaned_data,'RestBP')
    cleaned_data.drop(["index", "Unnamed: 0","ChestPain","Thal"], axis=1, inplace=True)
    print(cleaned_data.head())
    print(cleaned_data.shape)


    sns.set(style="ticks")
    sns.pairplot(cleaned_data.iloc[:,[0,1,2,3,4,5,11]])
    plt.tight_layout()
    plt.show()


    train_X, test_X, train_y, test_y = features.heart.split_train_test(cleaned_data,"AHD")
    train_X,test_X,train_y,test_y = features.heart.scale_to_train([train_X,test_X,train_y,test_y],[0,2,3,6,8,9,10,18],"minmax")
    print(train_X)

    # log reg with simple feature set
    print("Evaluating simple feature set")
    #log_reg = lm.SGDClassifier(n_jobs=10, loss="log", max_iter = 50)
    log_reg = lm.LogisticRegression()
    log_reg.fit(train_X, train_y)
    pred = log_reg.predict(test_X)
    pred_proba = log_reg.predict_proba(test_X)

    evaluation.print_errors(test_y, pred)
    print("")

    """
    # log reg with advanced feature set
    print("Evaluating modified feature set")
    log_reg2 = lm.SGDClassifier(n_jobs=1, loss="log", max_iter=50)

    classifier.fit(log_reg2, input_data2, targets)
    pred, pred_proba = classifier.predict(log_reg2, input_data2)

    evaluation.print_errors(targets, pred)
    """

