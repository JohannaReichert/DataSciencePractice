import pandas as pd
import trees
from sklearn.tree import DecisionTreeClassifier
from time import sleep
import evaluation


nrows = 3000
pd.set_option('display.max_columns',30)
x_train = pd.read_csv("../data/imdb/x_train_vect.csv", nrows = nrows)
y_train = pd.read_csv("../data/imdb/y_train.csv", nrows = nrows)
sleep(7)
x_train_occ = pd.read_csv("../data/imdb/x_train_occ.csv", nrows = nrows)
x_train_occ.columns = list(range(10000,20000))
x_train_comb = pd.concat([x_train,x_train_occ],axis = 1)
sleep(7)
## NaNs? ##
print("x_train NaNs:")
print(x_train.isnull().sum().sum())
print("x_train_occ:")
print(x_train_occ.isnull().sum().sum())
print("x_train_comb:")
print(x_train_comb.isnull().sum().sum())

### Decision Trees ###

dTree_vec = DecisionTreeClassifier(max_depth = 10)
dTree_vec.fit(x_train,y_train)
dTree_occ = DecisionTreeClassifier(max_depth = 10)
dTree_occ.fit(x_train_occ,y_train)
dTree_comb = DecisionTreeClassifier(max_depth = 10)
dTree_comb.fit(x_train_comb,y_train)
#Vectorized mean,std: (0.5838739865889755, 0.03191489501738701)
#Occurences mean,std: (0.574969657304286, 0.03816409097382029)
#Combined mean,std: (0.5760808776552618, 0.04740816460389906)
# CrossValiScores

vec_mean_score,vec_std_score = evaluation.cross_vali_rmse(dTree_vec,x_train,y_train)
occ_mean_score,occ_std_score = evaluation.cross_vali_rmse(dTree_occ,x_train_occ,y_train)
comb_mean_score,comb_std_score = evaluation.cross_vali_rmse(dTree_comb,x_train_comb,y_train)

print(f"Vectorized mean,std: {vec_mean_score,vec_std_score}")
print(f"Occurences mean,std: {occ_mean_score,occ_std_score}")
print(f"Combined mean,std: {comb_mean_score,comb_std_score}")

vec_mean_scores = evaluation.cross_vali_errors_classification(dTree_vec,x_train,y_train)
occ_mean_scores = evaluation.cross_vali_errors_classification(dTree_occ,x_train_occ,y_train)
comb_mean_scores = evaluation.cross_vali_errors_classification(dTree_comb,x_train_comb,y_train)

print(f"Vectorized acc mean: {vec_mean_scores['test_accuracy'].mean()}")
print(f"Occurences acc mean: {occ_mean_scores['test_accuracy'].mean()}")
print(f"Combined acc mean: {comb_mean_scores['test_accuracy'].mean()}")




#tree_pruner = trees.TreePruner(dTree)
#tree_pruner.run()





