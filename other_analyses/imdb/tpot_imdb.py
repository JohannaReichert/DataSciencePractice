import pandas as pd
from time import sleep
from tpot import TPOTClassifier


nrows = 1000
pd.set_option('display.max_columns',30)
x_train = pd.read_csv("data/imdb/x_train_vect.csv", nrows = nrows)

y_train = pd.read_csv("data/imdb/y_train.csv", nrows = nrows)
sleep(7)
x_train_occ = pd.read_csv("data/imdb/x_train_occ.csv", nrows = nrows)
x_train_occ.columns = list(range(10000,20000))
x_train_comb = pd.concat([x_train,x_train_occ],axis = 1)
sleep(7)


my_tpot_vect = TPOTClassifier(generations=5,config_dict='TPOT light',verbosity=2)
my_tpot_vect.fit(x_train.iloc[:,0:100], y_train.values)
my_tpot_vect.export('vect_exported_pipeline_light_5gen.py')

"""
exported_pipeline = LogisticRegression(C=10.0, dual=False, penalty="l2")
RandomForestClassifier(BernoulliNB(input_matrix, BernoulliNB__alpha=10.0, BernoulliNB__fit_prior=False), RandomForestClassifier__bootstrap=False, RandomForestClassifier__criterion=gini, RandomForestClassifier__max_features=0.15000000000000002, RandomForestClassifier__min_samples_leaf=7, RandomForestClassifier__min_samples_split=2, RandomForestClassifier__n_estimators=100)

my_tpot_occ = TPOTClassifier(generations=10)
my_tpot_occ.fit(x_train, y_train.ravel())

my_tpot_occ.export('occ_exported_pipeline.py')
"""




