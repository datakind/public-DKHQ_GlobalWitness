import argparse
import numpy as np
import pandas as pd
import seaborn as sns;

sns.set_style("whitegrid")
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import json
from sklearn.model_selection import GroupKFold

def main(args):

    param_grid = {
        'n_estimators': [5, 6],
        'max_depth': [10, 15, 20],
        'n_jobs': [-1]
    }
    X,y,groups = group_split(args.splits_path)

    print(np.shape(X), np.shape(y), np.shape(groups))
    model = gridsearch(X, y, groups, RandomForestClassifier, param_grid, scoring='precision_macro')

def group_split(splits):
    X=[]
    y=[]
    groups=[]

    i=0
    for i, split in enumerate(splits):
        a,b = load_data(split)

        X.append(a)
        y.append(b)
        groups.append((np.shape(a)[0])*[i])
        i+=1


    return np.concatenate(X), np.concatenate(y), np.concatenate(groups)

def gridsearch(X, y, groups, estimator, param_grid, scoring='f1'):
    gkf = list(GroupKFold(n_splits=3).split(X, y, groups))

    clf = GridSearchCV(estimator(), scoring=scoring, cv=gkf, param_grid=param_grid, verbose=2)
    A = clf.fit(X,y)

    print clf.best_params_
    print clf.best_score_
    return A.best_estimator_


def load_data(data_path):
    with np.load(data_path) as f:
        X = f['X']
        y = f['y']

    return X, y


def export_model(model, model_path):
    joblib.dump(model, model_path)


def predict(X, model):
    cluster_ids = model.predict(X).astype(int)

    return cluster_ids


def predict_pr(X, y, model):
    plt.figure(figsize=(6, 6))

    y_hat = model.predict_proba(X)[:, 1]
    auc_pr = average_precision_score(y, y_hat)

    precision, recall, _ = precision_recall_curve(y, y_hat)
    plt.plot(precision, recall, color='red', lw=3)

    y_hat = np.random.rand(len(X))
    precision_rand, recall_rand, _ = precision_recall_curve(y, y_hat)

    plt.plot(precision_rand, recall_rand, color='blue', lw=3)

    plt.axis('equal')
    plt.ylabel('Precision')
    plt.xlabel('Recall')

    plt.title("AUC under PR-Curve: %f%%" % (100 * auc_pr))
    plt.legend(["RF", "random"])
    plt.show()


def predict_confusion(X, y, model):
    cluster_ids = predict(X, model)

    cluster_ids = pd.DataFrame({
        "prediction": cluster_ids,
        "has_mine": y.astype(int),
        "example_id": range(len(cluster_ids)),
    })

    confusion_matrix = pd.pivot_table(
        cluster_ids,
        index="has_mine",
        columns="prediction",
        values="example_id",
        fill_value=0,
        aggfunc=lambda series: series.count())

    print confusion_matrix


def train(X, y, estimator=RandomForestClassifier, n_estimators=10, max_depth=10):
    random_forest = estimator(n_estimators=n_estimators, n_jobs=-1,
                              max_depth=max_depth, class_weight="balanced",
                              oob_score=True)

    return random_forest.fit(X, y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--splits_path',
        type=str,
        help='path to split file', action='append')

    parser.add_argument(
        '--export_model_path',
        type=str,
        help='Model Export Path')
    main(parser.parse_args())
