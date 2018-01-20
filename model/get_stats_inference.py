import argparse
import storage
import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt

def main(args):
    dataset = storage.DiskDataset(args.data_path)

    # get_stats_by_site_agg(dataset, args)
    get_stats_by_site_and_time(dataset, args)

def get_stats_by_site_and_time(dataset, args):

    mask_list = []
    image_list = []

    for image, image_metadata in dataset.load_images(args.source_id+'_inference'):

        mask = dataset.load_image(image_metadata['location_id'], 'mask')

        fig, ax = plt.subplots(nrows=2, ncols=2)
        plt.subplot(1, 2, 1)
        plt.imshow(np.median(image, axis=0).astype(np.bool))

        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.show()

        fig, ax = plt.subplots(nrows=4, ncols=4)
        for time in range(image.shape[0]):

            mask_list.append(mask.flatten())
            image_list.append(image[time,:,:].flatten())
            plt.subplot(3,3, time+1)
            plt.imshow(image[time,:,:])

        plt.show()


    y_true = np.concatenate(mask_list)
    y_pred = np.concatenate(image_list)

    print 'recall', metrics.recall_score(y_true, y_pred)
    print 'precision', metrics.precision_score(y_true, y_pred)



def get_stats_by_site_agg(dataset, args):

    mask_list = []
    image_list = []

    for image, image_metadata in dataset.load_images(args.source_id+'_inference_timeagg'):
        mask = dataset.load_image(image_metadata['location_id'], 'mask')

        mask_list.append(mask.flatten())
        image_list.append(image.flatten())

        fig, ax = plt.subplots(nrows=1, ncols=2)

        plt.subplot(1, 2, 1)
        plt.imshow(image)

        plt.subplot(1, 2, 2)
        plt.imshow(mask)

        plt.show()

    y_true = np.concatenate(mask_list)
    y_pred = np.concatenate(image_list)

    print 'recall', metrics.recall_score(y_true, y_pred)
    print 'precision', metrics.precision_score(y_true, y_pred)

    # image_metadata['metadata']['dates'] = image_metadata['metadata']['dates']




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        help='Data path')

    parser.add_argument(
        '--source_id',
        type=str,
        default='landsat8',
        help='The source id (should almost certainly be "landsat8"'
    )
    main(parser.parse_args())