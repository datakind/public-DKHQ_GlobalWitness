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
from sklearn.cluster import KMeans

from pymasker import LandsatMasker
from pymasker import LandsatConfidence

# Where to read images from.
IMAGE_INPUT_DIR = "images_h5" #images_h5_sample images_h5

def main(args):
    # We are just exporting our data

    imgs, img_metadata = load_images(IMAGE_INPUT_DIR)

    # Set cloud mask threshold
    cloud_threshold = 3 # high confidence

    # Turn each pixel into an example. Use all color bands except for 'mask' (the last one).
    num_bands = imgs.shape[3]
    x, bad_row_idxs = per_pixel_images(imgs[:,:,:,range(num_bands-1),:], img_metadata, cloud_threshold)

    # Turn each pixel int a label. Use the last color band ('mask').
    y = per_pixel_labels(imgs[:,:,:,[num_bands-1],:], bad_row_idxs)
    y = y[:, 0]

    np.savez(args.export_data,x=x,y=y)
    print 'x shape', x.shape
    print 'y shape', y.shape

    assert x.shape[0] == y.shape[0], 'x and y lengths should be the same.'

    print 'Constructed %d examples, %d features, %0.3f%% positive' % (x.shape[0], x.shape[1], np.mean(y))

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

def create_a_months_matrix(images, img_metadata):
    a = np.zeros((images.shape[0], images.shape[4]))
    for i, (j, row) in enumerate(img_metadata.iterrows()):
        a[int(i),:] = np.array([int(j[4:6]) for j in row['dates']])
    a = np.expand_dims(a, axis=1)
    a = np.expand_dims(a, axis=1)
    a = np.expand_dims(a, axis=1)
    a = np.repeat(a, 100, axis=1)
    a = np.repeat(a, 100, axis=2)
    a = np.repeat(a, 16, axis=3)
    return a

def expand_bqa_dims(cloud_array, nsites, ntimes, nbands):
    clouds = np.expand_dims(cloud_array, axis=3)
    allclouds = np.repeat(clouds, 16, axis=3)
    return allclouds

def get_bqa_mask(bqa):
    '''
    Takes in a numpy array of shape(n_images, height, width, num_times) and returns
    a mask of shape(n_images, width, height, num_times)
    '''
    landsat_confidences = {
        LandsatConfidence.low: 1,
        LandsatConfidence.medium: 2,
        LandsatConfidence.high: 3
    }
    bqa = bqa.astype(np.int16)
    conf_is_cumulative = False
    masker = LandsatMasker(bqa, collection=0)
    one_img_mask_all_conf = np.zeros_like(bqa).astype(np.int16)

    num_images, height, width, num_times = bqa.shape

    for i in xrange(num_images):
        for j in xrange(num_times):
            for conf in landsat_confidences:
                mask = masker.get_cloud_mask(conf, cumulative=conf_is_cumulative)
                mask = mask * landsat_confidences[conf]  # multiply each conf by a value
                one_img_mask_all_conf[i,:,:,j] += mask[i,:,:,j]  # sum all conf masks together

    return one_img_mask_all_conf


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
        '--export_data',
        type=str,
        help='Data path')

    main(parser.parse_args())