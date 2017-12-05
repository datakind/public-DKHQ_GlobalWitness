import argparse
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import sonnet as snt
import seaborn as sns; sns.set_style("whitegrid")
from matplotlib import pyplot as plt
import mining; reload(mining)
import math

import storage

from pymasker import LandsatMasker
from pymasker import LandsatConfidence
from sklearn.model_selection import train_test_split

CLOUD_THRESHOLD = 3

def main(args):
    # We are just exporting our data

    image_features, bqas, masks = load_images_storage(args.data_input_path)

    bad_idxs = get_bad_idxs(image_features, bqas)
    X = image_features[~bad_idxs,:]
    y = masks[~bad_idxs]

    np.savez(args.data_export_path, X=X, y=y)

def get_bad_idxs(image_features, bqa):
    cloud_masks = get_cloud_mask(bqa)

    bad_idxs = cloud_masks >= CLOUD_THRESHOLD

    bad_idxs = np.logical_or(bad_idxs, np.logical_or.reduce(np.isinf(image_features), axis=1))
    bad_idxs = np.logical_or(bad_idxs, np.logical_or.reduce(np.isnan(image_features), axis=1))

    return bad_idxs

def load_images_storage(image_input_dir, source_id='landsat8', bqa_index=11):
    dataset = storage.DiskDataset(image_input_dir)

    images_features = []
    masks = []
    bqas = []

    for image, image_metadata in dataset.load_images(source_id):
        image = image[:,:,:,0:-1]

        mask = dataset.load_image(image_metadata['location_id'], 'mask')

        image_metadata['metadata']['dates'] = image_metadata['metadata']['dates'][0:-1]

        # Shift the indices so the order is  [dates,x,y, bands] (need this to reshape later)
        image = np.transpose(image, [3, 0, 1, 2])
        image_databands = image[:, :, :, :bqa_index]

        bqa = image[:, :, :, bqa_index]

        n_dates, n_x, n_y, n_bands = np.shape(image_databands)

        # We take the cosine of the month becase the altenative is making it a categorical variable
        # and one hot encoding it. The cosine trick means we can make it a continuous variable
        # and puts january and december in similar places.
        months = [math.cos(float(date[4:6])/12.0) for date in image_metadata['metadata']['dates']]

        dates = np.expand_dims(np.repeat(months, n_x*n_y), 1)

        #Append our mask (nparray) to a list of masks
        mask = np.repeat(np.expand_dims(mask, axis=0), n_dates, axis=0)
        masks.append(np.reshape(mask, (n_x*n_y*n_dates)))

        image_feature = np.reshape(image_databands, (n_x*n_y*n_dates, n_bands))
        image_features = np.hstack([image_feature, dates])

        # Append the numpy array to a list of numpy arrays
        images_features.append(image_features)

        bqa_features = np.reshape(bqa, (n_x*n_y*n_dates))
        bqas.append(bqa_features)

    bqas = np.concatenate(bqas, axis=0)
    images_features = np.concatenate(images_features, axis=0)
    masks = np.concatenate(masks, axis=0)

    return images_features, bqas, masks


def get_cloud_mask(bqa):
    '''
    Takes in a bqa array and returns a mask of clouds
    '''
    landsat_confidences = {
        LandsatConfidence.low: 1,
        LandsatConfidence.medium: 2,
        LandsatConfidence.high: 3
    }

    bqa = bqa.astype(np.int16)
    conf_is_cumulative = False
    masker = LandsatMasker(bqa, collection=0)
    cloud_conf = np.zeros_like(bqa).astype(np.int16)

    for conf in landsat_confidences:
        print "Getting confidence", conf
        mask = masker.get_cloud_mask(conf, cumulative=conf_is_cumulative)
        cloud_conf += mask * landsat_confidences[conf]  # multiply each conf by a value

    print "Done conf"
    return cloud_conf

def get_bqa_water_mask(bqa):
    bqa = bqa.astype(np.int16)
    conf_is_cumulative = False
    masker = LandsatMasker(bqa, collection=0)
    one_img_mask = np.zeros_like(bqa).astype(np.int16)

    num_images, height, width, num_times = bqa.shape
    for i in xrange(num_images):
        for j in xrange(num_times):
            conf = 2  # water mask only accepts values of 0 or 2
            mask = masker.get_water_mask(conf, cumulative=conf_is_cumulative)
            one_img_mask[i, :, :, j] = mask[i, :, :, j]

    return one_img_mask

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_export_path',
        type=str,
        help='Data path')
    parser.add_argument(
        '--data_input_path',
        type=str,
        help='Storage path')

    main(parser.parse_args())