import argparse
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import sonnet as snt
import seaborn as sns; sns.set_style("whitegrid")
from matplotlib import pyplot as plt
import mining; reload(mining)
import os
import h5py
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import storage

from pymasker import LandsatMasker
from pymasker import LandsatConfidence
from sklearn.model_selection import train_test_split
import time


# Where to read images from.
IMAGE_INPUT_DIR = "images_h5" #images_h5_sample images_h5

CLOUD_THRESHOLD = 3
def main(args):
    # We are just exporting our data

    image_features, bqas, masks = load_images_storage(args.image_input_dir)

    bad_idxs = get_bad_idxs(image_features, bqas)
    X = image_features[~bad_idxs,:]
    y = masks[~bad_idxs]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    np.savez(args.export_data_path+'/test_train.npz', X_train, X_test, y_train, y_test)

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


def load_images(image_input_dir):
    index_list = range(14, 21) + [30, 31, 37, 47, 56, 61, 76, 77, 80, 84, 92, 103,
                                  104]  # ,105,106,107,108,123,124,125,127,131,138,140,143]

    img_metadata = mining.load_metadata(image_input_dir)

    imgs = mining.load_images_with_hdf5(image_input_dir)

    print 'Loaded %d images of shape %s' % (imgs.shape[0], imgs.shape[1:])

    imgs = imgs[index_list, :, :, :, :]

    img_metadata = img_metadata.reset_index(drop=True)
    img_metadata = img_metadata.ix[index_list]

    return imgs, img_metadata

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


# Split into per-pixel problems. Ignore (band, month) pairs with inf entries.

def per_pixel_images(images, img_metadata, cloud_threshold):
    """Create per-pixel rows."""
    assert len(images.shape) == 5, "images should be [image id, height, width, color bands, time]"

    # split out the bqa band
    bqa = images[:, :, :, 11, :]
    # keep the image data separate
    num_bands = 16  # new hdf5 files have 18 bands, incl bqa and mines mask
    images_only = images[:, :, :, range(11) + range(12, num_bands + 1), :]
    num_images, height, width, num_bands, num_times = images_only.shape

    # transpose axes to reshape with band in last dim
    images_only = np.transpose(images_only, axes=[0, 1, 2, 4, 3])
    images_only = np.reshape(images_only, [num_images * height * width * num_times, num_bands])

    # make a months matrix
    months = create_a_months_matrix(images, img_metadata)
    months = np.transpose(months, axes=[0, 1, 2, 4, 3])
    months = np.reshape(months, [num_images * height * width * num_times, num_bands])

    # explode out the bqa band dimensions
    bqa = get_bqa_mask(bqa)
    clouds = expand_bqa_dims(bqa, num_images, num_times, num_bands)
    clouds = np.transpose(clouds, axes=[0, 1, 2, 4, 3])
    clouds = clouds.reshape([num_images * height * width * num_times, num_bands])

    print bqa.shape
    print bqa[bqa == 3].sum()
    print bqa[bqa >= cloud_threshold].sum()

    '''# explode out the water band dimensions
    WATER_THRESHOLD = 2
    water = get_bqa_water_mask(bqa) 
    water = expand_bqa_dims(water, num_images, num_times, num_bands) 
    water = np.transpose(water, axes=[0,1,2,4,3])
    water = water.reshape([num_images * height * width * num_times, num_bands])
    print 'bqa water mask', (water >= 2).sum()
    '''
    # create a mask flagging pixels that are inf, nan, or clouded
    print 'bqa cloud mask', (clouds >= cloud_threshold).sum()
    print 'np.isinf', np.isinf(images_only).sum()
    print 'np.isnan', np.isnan(images_only).sum()

    # print bqa >= cloud_threshold
    bad_idxs = np.logical_or(np.logical_or(clouds >= cloud_threshold, np.isinf(images_only)), \
                             np.isnan(images_only))  # , water == WATER_THRESHOLD) np.logical_or(
    print bad_idxs.sum()
    # collapse down to one col -- if there is a bad pixel (inf/nan) in one band, we throw out all bands (clouds are flagged in all bands)
    bad_row_idxs = np.logical_or.reduce(bad_idxs, axis=1)
    print bad_row_idxs.sum()
    images_only = images_only[~bad_row_idxs, :]
    months = months[~bad_row_idxs, 0]
    months = np.expand_dims(months, axis=1)

    # concatenate pixel data + month data
    images_only = np.hstack((images_only, months))

    return images_only, bad_row_idxs


# Split into per-pixel labels. Ignore (band, month) pairs with inf entries.

def per_pixel_labels(images, bad_row_idxs):
    """Create per-pixel rows."""
    assert len(images.shape) == 5, "images should be [image id, height, width, color bands, time]"

    num_images, height, width, num_bands, num_times = images.shape
    images = np.transpose(images, axes=[0, 1, 2, 4, 3])
    result = np.reshape(images, [num_images * height * width * num_times, num_bands])
    result = result[~bad_row_idxs]

    return result

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--export_data_path',
        type=str,
        help='Data path')
    parser.add_argument(
        '--image_input_dir',
        type=str,
        help='Storage path')

    main(parser.parse_args())