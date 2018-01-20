import argparse
import numpy as np
import pandas as pd
import seaborn as sns; sns.set_style("whitegrid")
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, auc, roc_curve, roc_auc_score
from sklearn.externals import joblib
import time
import storage

def main(args):
    source_id='landsat8'
    dataset = storage.DiskDataset(args.storage_path)

    predictions=[]

    for image, image_metadata in dataset.load_images(source_id):
        mask = dataset.load_image(image_metadata['location_id'], 'mask')
        prediction = dataset.load_image(image_metadata['location_id'], source_id+'_inference_timeagg')

        assert mask.shape==prediction.shape, "Error, prediciton and mask not the same size."

        predictions.append(pd.DataFrame({
                        "prediction": np.reshape(mask, (mask.size,)),
                        "has_mine": np.reshape(prediction, (prediction.size, ))
                    }))

    predictions=pd.concat(predictions)
    predictions['example_id'] = range(len(predictions))

    confusion_matrix = pd.pivot_table(
        predictions,
        index="has_mine",
        columns="prediction",
        values="example_id",
        fill_value=0,
        aggfunc=lambda series: series.count())

    print(confusion_matrix)
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
        '--storage_path',
        type=str,
        help='test preprocessed data file')

    main(parser.parse_args())
