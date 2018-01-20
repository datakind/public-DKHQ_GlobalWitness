import argparse
import numpy as np
import pandas as pd
import seaborn as sns; sns.set_style("whitegrid")
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, auc, roc_curve, roc_auc_score
from sklearn.externals import joblib
import time


def main(args):
    model = load_model(args.model_path)
    X_test, y_test = load_data(args.test_data_path)


    start = time.time()
    predict_confusion(X_test,y_test,model)
    end = time.time()
    print(end - start)
    # predict_pr(X_test,y_test, model)
    # predict_roc(X_test, y_test, model)

def load_data(data_path):
    with np.load(data_path) as f:
        X=f['X']
        y = f['y']

    return X, y

def load_model(model_path):
    return joblib.load(model_path)

def predict_roc(X, y, model):

    plt.figure(figsize=(6, 6))

    y_hat = model.predict_proba(X)[:, 1]
    roc_area = roc_auc_score(y, y_hat)

    precision, recall, _ = roc_curve(y, y_hat)
    plt.plot(precision, recall, color='red', lw=3)

    y_hat = np.random.rand(len(X))
    precision_rand, recall_rand, _ = precision_recall_curve(y, y_hat)

    plt.plot(precision_rand, recall_rand, color='blue', lw=3)

    plt.axis('equal')
    plt.ylabel('TPR')
    plt.xlabel('FPR')

    plt.title("AUC under ROC-Curve: %f%%" % (100 * roc_area))
    plt.legend(["RF", "random"])
    plt.show()


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


def predict(X, model):
    cluster_ids = model.predict(X).astype(int)

    return cluster_ids


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


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--test_data_path',
        type=str,
        help='test preprocessed data file')


    parser.add_argument(
        '--model_path',
        type=str,
        help='Model Export Path')
    main(parser.parse_args())
