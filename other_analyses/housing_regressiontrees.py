import pandas as pd
import viz
import evaluation
import features.prep
import trees
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_columns',30)

if __name__ == '__main__':
    data = pd.read_csv("../data/housing.csv",sep=";")
    #print(data.head())
    data = features.prep.dropnans_newidx(data)
    #print(data.shape)
    data = features.prep.convert_cats(data,"ocean_proximity")

    #####################
    data = data.sample(1000)
    data["median_house_value"] = data["median_house_value"]/100000
    #data = data[["latitude","longitude","median_income","median_house_value"]]
    ###################

    train_X, val_X, test_X, train_y, val_y, test_y = features.prep.split_train_val_test(data, "median_house_value",[0.6,0.2,0.2])

    # scaling is not necessary for decision trees

    # finding alpha --> best subtree from pruning

    dTree = DecisionTreeRegressor()
    dTree.fit(train_X, train_y)

    tree_pruner = trees.TreePruner(dTree)
    tree_pruner.run()

    error_rates = pd.DataFrame(columns=["tree", "MSE", "RMSE", "R2", "RMSE % of mean", "Cali"])
    error_rates_tr = pd.DataFrame(columns=["tree", "MSE", "RMSE", "R2", "RMSE % of mean", "Cali"])

    for t in tree_pruner.trees:
        pred = t.predict(val_X)
        pred_training = t.predict(train_X)
        idx = tree_pruner.trees.index(t)
        new_row = pd.concat([pd.Series(idx,name="tree"),evaluation.save_errors(val_y, pred)],axis=1)
        new_row_tr = pd.concat([pd.Series(idx,name="tree"),evaluation.save_errors(train_y, pred_training)],axis=1)
        error_rates = error_rates.append(new_row,sort=False)
        error_rates_tr = error_rates_tr.append(new_row_tr,sort=False)

    print("sorted error rates for val:\n")
    err_sorted = error_rates.sort_values(["R2","RMSE"],ascending=[False,True])
    print(err_sorted)
    best_tree_nr = err_sorted.iloc[0,0]
    best_tree = tree_pruner.trees[best_tree_nr]

    fig = viz.create_error_figures(error_rates, error_rates_tr, tree_pruner.alpha_array,"alpha",lim=1)
    a =viz.ScrollableWindow(fig)
    fig = viz.create_error_figures(error_rates,error_rates_tr,error_rates["tree"],"subtree",lim=0)
    b = viz.ScrollableWindow(fig)

    random_dTree = RandomForestRegressor()
    random_dTree.fit(train_X, train_y)
    rand_pred_val = random_dTree.predict(val_X)
    rand_pred_tr = random_dTree.predict(train_X)
    rand_error_rates_val = pd.DataFrame(columns=["type", "MSE", "RMSE", "R2", "RMSE % of mean", "Cali"])
    rand_error_rates_tr = pd.DataFrame(columns=["type", "MSE", "RMSE", "R2", "RMSE % of mean", "Cali"])

    rand_new_row = pd.concat([pd.Series("val", name="type"), evaluation.save_errors(val_y, rand_pred_val)], axis=1)
    rand_new_row_tr = pd.concat([pd.Series("train", name="type"), evaluation.save_errors(train_y, rand_pred_tr)],
                                axis=1)
    rand_error_rates = rand_error_rates_val.append(rand_new_row, sort=False)
    rand_error_rates_tr = rand_error_rates_tr.append(rand_new_row_tr, sort=False)
    print("pruned error rates val:")
    print(error_rates.loc[error_rates["tree"] == best_tree_nr, :])
    print("pruned error rates train:")
    print(error_rates_tr.loc[error_rates["tree"] == best_tree_nr, :])
    print("rand error rates val:")
    print(rand_error_rates)
    print("rand error rates train:")
    print(rand_error_rates_tr)

    fig5, axes5 = plt.subplots(3, 2, figsize=(10, 12))
    ax5 = axes5.flatten()
    for col in error_rates.iloc[:, 1:6].columns.values:
        axnr = error_rates.columns.get_loc(col) - 1
        ax5[axnr].set_title(col)
        ax5[axnr].set_xlabel("subtree")
        ax5[axnr].scatter("pruned", error_rates.loc[error_rates["tree"] == best_tree_nr, col], label="best pruned val")
        ax5[axnr].scatter("pruned", error_rates_tr.loc[error_rates["tree"] == best_tree_nr, col], label="best pruned train")
        ax5[axnr].scatter("rand", rand_error_rates[col], label="randomforest val")
        ax5[axnr].scatter("rand", rand_error_rates_tr[col], label="randomforest train")
        ax5[axnr].legend()

    fig5.tight_layout()
    b = viz.ScrollableWindow(fig5)

