import argparse
import numpy as np
import pandas as pd
import seaborn as sns; sns.set_style("whitegrid")
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import json

def main(args):
    X, y = load_data(args.train_data_path)
    X_test, y_test = load_data(args.test_data_path)

    model = train(X, y)

    # param_grid = {
    #     'n_estimators':[50, 100, 200, 300],
    #     'max_depth':[5, 10, 20, 30, 50, None ]
    # }
    # model = gridsearch(X, y, RandomForestClassifier, param_grid)
    export_model(model, args.export_model_path)
    
    predict_confusion(X_test, y_test, model, num_estimators = 300, max_depth=40)
    #predict_pr(X_test, y_test, model)


def gridsearch(X, y, estimator, param_grid, scoring='f1'):
    clf = GridSearchCV(estimator(), scoring=scoring, param_grid=param_grid, n_jobs=16, verbose=2)
    A=clf.fit(X, y)

    print clf.best_params_
    print clf.best_score_
    return A.best_estimator_
def load_data(data_path):
    with np.load(data_path) as f:
        X=f['X']
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

def train(X, y, estimator=RandomForestClassifier, num_estimators = 10, max_depth=10):
    random_forest = estimator(n_estimators=num_estimators, n_jobs=-1,
                                           max_depth=max_depth, class_weight="balanced",
                                           oob_score=True)

    return random_forest.fit(X, y)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_data_path',
        type=str,
        help='train preprocessed data file')

    parser.add_argument(
        '--test_data_path',
        type=str,
        help='test preprocessed data file')


    parser.add_argument(
        '--export_model_path',
        type=str,
        help='Model Export Path')
    main(parser.parse_args())
