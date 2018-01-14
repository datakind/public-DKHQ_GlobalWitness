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
        'n_estimators': range(400, 551, 50),
        'max_depth': range(40, 60, 5) + [None],
        'min_samples_leaf': [1, 2, 10, 50, 75, 100],
        'n_jobs': [-1]
    }
    X,y,groups = group_split(args.splits_path)

    model = gridsearch(X, y, groups, RandomForestClassifier, param_grid, scoring='precision_macro', n_jobs=args.n_jobs)

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


def jsonify(data):
    json_data = dict()
    for key, value in data.iteritems():
        if isinstance(value, list):  # for lists
            value = [jsonify(item) if isinstance(item, dict) else item for item in value]
        if isinstance(value, dict):  # for nested lists
            value = jsonify(value)
        if isinstance(key, int):  # if key is integer: > to string
            key = str(key)
        if type(value).__module__ == 'numpy':  # if value is numpy.*: > to python list
            value = value.tolist()
        if type(value).__module__ == 'numpy.ma.core':
            value = value.tolist()
        json_data[key] = value
    return json_data

def gridsearch(X, y, groups, estimator, param_grid, scoring='f1', n_jobs=1):
    gkf = list(GroupKFold(n_splits=3).split(X, y, groups))

    # testing (kt)
    scoring = ['average_precision', 'precision', 'recall', 'accuracy', 'f1', 'roc_auc']


    clf = GridSearchCV(estimator(), scoring=scoring, cv=gkf, param_grid=param_grid, verbose=2, refit=False, n_jobs=n_jobs)
    A = clf.fit(X,y)

    data_to_json = jsonify(clf.cv_results_) #best_params_

    with open('params_eval.json', 'w') as outfile:
        json.dump(data_to_json, outfile)

    return #A.best_estimator_


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
        '--n_jobs',
        type=int,
        default=1,
        help='Number of jobs to run the gridsearch '
    )
    main(parser.parse_args())
