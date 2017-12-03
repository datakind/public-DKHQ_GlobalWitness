import argparse
import time
import numpy as np
import pandas as pd
import sonnet as snt
import seaborn as sns; sns.set_style("whitegrid")
from matplotlib import pyplot as plt
import mining; reload(mining)
import os
import h5py
from sklearn.cluster import KMeans

from pymasker import LandsatMasker
from pymasker import LandsatConfidence

import sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier

def main(args):
    X_train, X_test, y_train, y_test = load_data(args.data_path)

    train(X_train, y_train)

def load_data(data_path):
    f = np.load(data_path)

    X_train=f['X_train']
    X_test = f['X_test']
    y_train = f['y_train']
    y_test = f['y_test']

    return X_train, X_test, y_train, y_test

def train(x,y):

    num_estimators = [10]  # , 300, 600]
    max_depths = [10]  # , 40, 50]

    for n_estimators in num_estimators:
        for max_depth in max_depths:
            random_forest = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1,
                                                   max_depth=max_depth, class_weight="balanced",
                                                   oob_score=True)
            random_forest.fit(x, y)
            print 'num_estimators', n_estimators
            print 'max_depth', max_depth
            print 'Balanced Out-of-Bag score: %f' % random_forest.oob_score_

            # Plot confusion matrix for Random Forest classifier.
            # Do any clusters have more mines than the others?

            cluster_ids = random_forest.predict(x).astype(int)

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

            # Plot Precision-Recall Curve of Random Forest

            from sklearn.metrics import precision_recall_curve, average_precision_score, auc

            plt.figure(figsize=(6, 6))
            plt.axis('equal')
            plt.ylabel('Precision')
            plt.xlabel('Recall')

            # Plot Random Forest's PR-curve
            y_hat = random_forest.predict_proba(x)
            print(y_hat.shape)
            y_hat = random_forest.predict_proba(x)[:, 1]
            auc_pr = average_precision_score(y, y_hat)
            precision, recall, decision_boundaries = precision_recall_curve(y, y_hat)
            plt.plot(precision, recall, color='red', lw=3)

            # Plot random classifier's PR-curve
            y_hat = np.random.rand(len(x))
            precision_rand, recall_rand, _ = precision_recall_curve(y, y_hat)
            plt.plot(precision_rand, recall_rand, color='blue', lw=3)

            plt.title("AUC under PR-Curve: %f%%" % (100 * auc_pr))
            plt.legend(["RF", "random"])
            plt.show()
            # 0.838454 -- first try

    return random_forest

    print 'Constructed %d examples, %d features, %0.3f%% positive' % (x.shape[0], x.shape[1], np.mean(y))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        help='Data path')
    parser.add_argument(
        '--export_model_path',
        type=str,
        help='Model Export Path')
    main(parser.parse_args())