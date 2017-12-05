import argparse
from sklearn.externals import joblib
import storage
import numpy as np
import math
from matplotlib import pyplot as plt

from pymasker import LandsatMasker
from pymasker import LandsatConfidence

CLOUD_THRESHOLD = 3

def main(args):
    model = load_model(args.model_path)
    dataset = storage.DiskDataset(args.data_path)

    inference_store(dataset, model)

    # images_features, bqas, masks, shapes = load_images_storage(args.data_path)


def load_model(model_path):
    model = joblib.load(model_path)

    return model

def inference_store(dataset, model, source_id='landsat8', bqa_index=11):
    for image, image_metadata in dataset.load_images(source_id):
        mask = dataset.load_image(image_metadata['location_id'], 'mask')

        # bqa has shape dates, x, y
        # image_features has shape [-1, n_bands+1]
        image_features = get_image_features(image, image_metadata, bqa_index)

        # This extracts the bqa index in order [dates, x, y]
        bqa = np.transpose(image, [3,0,1,2])[:,:,:,bqa_index]
        bad_idxs = get_bad_idxs(bqa)

        # Does the inference, then reshapes back to our [dates, x, y] coordinates, then fills the bad indices with -1.0
        predictions = inference(model, image_features)
        predictions = np.reshape(predictions, np.shape(bad_idxs))
        predictions[bad_idxs] = -1.0

        predictions[predictions[:,:,:]>=0.5]=1
        predictions[predictions[:,:,:]<0]=0
        predictions[np.logical_and((predictions[:, :,:] > 0), (predictions[: :,:] <0.7)) ] = 0.5

        n_dates,_,_=np.shape(predictions)

        plt.subplot(1,3,1)
        plt.imshow(predictions[1,:,:])

        plt.subplot(1,3,2)
        plt.imshow(bad_idxs[1,:,:])

        plt.subplot(1,3,3)
        plt.imshow(image[:,:,5,1])
        # for i in range(8):
        #     plt.subplot(3,3,i+1)
        #     plt.imshow(predictions[i,:,:])

        # plt.subplot(3,3,9)
        # plt.imshow(mask)
        #
        plt.clim(0,1)
        plt.show()
        print image_metadata
        dataset.add_image(image_metadata['location_id'], source_id+'_inference', predictions, image_metadata.to_dict())


def inference(model, image_features):

    predictions = model.predict_proba(image_features)[:, 1]
    return predictions


def get_image_features(image, image_metadata, bqa_index=11):

    image_metadata['metadata']['dates'] = image_metadata['metadata']['dates']

    # Shift the indices so the order is  [dates,x,y, bands] (need this to reshape later)
    image = np.transpose(image, [3, 0, 1, 2])
    image_databands = image[:, :, :, :bqa_index]

    n_dates, n_x, n_y, n_bands = np.shape(image_databands)

    # We take the cosine of the month becase the altenative is making it a categorical variable
    # and one hot encoding it. The cosine trick means we can make it a continuous variable
    # and puts january and december in similar places.
    months = [math.cos(float(date[4:6]) / 12.0) for date in image_metadata['metadata']['dates']]

    dates = np.expand_dims(np.repeat(months, n_x * n_y), 1)

    image_features = np.reshape(image_databands, (n_x * n_y * n_dates, n_bands))
    image_features = np.hstack([image_features, dates])

    return image_features


def get_bad_idxs(bqa):
    cloud_masks = get_cloud_mask(bqa)

    # TODO(ryan) Add the nan stuff to this calculation. Just ignoring that for now because
    # of time
    bad_idxs = cloud_masks >= CLOUD_THRESHOLD

    return bad_idxs


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

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        help='Data path')
    parser.add_argument(
        '--model_path',
        type=str,
        help='Model Path')
    main(parser.parse_args())