import argparse
from sklearn.externals import joblib
import storage
import numpy as np
import math

def main(args):
    model = load_model(args.model_path)
    images_features, bqas, masks = load_images_storage(args.data_path)


def load_model(model_path):
    model = joblib.load(model_path)

    return model

def inference(model, image_features, bqas):
    predictions = model.predict_proba(image_features)[:, 1]




def load_images_storage(image_input_dir, source_id='landsat8', bqa_index=11, load_masks=False):
    dataset = storage.DiskDataset(image_input_dir)

    images_features = []
    masks = []
    bqas = []

    for image, image_metadata in dataset.load_images(source_id):
        image = image[:,:,:,0:-1]

        if load_masks:
            mask = dataset.load_image(image_metadata['location_id'], 'mask')
            #Append our mask (nparray) to a list of masks
            mask = np.repeat(np.expand_dims(mask, axis=0), n_dates, axis=0)
            masks.append(np.reshape(mask, (n_x*n_y*n_dates)))

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


        image_feature = np.reshape(image_databands, (n_x*n_y*n_dates, n_bands))
        image_features = np.hstack([image_feature, dates])

        # Append the numpy array to a list of numpy arrays
        images_features.append(image_features)

        bqa_features = np.reshape(bqa, (n_x*n_y*n_dates))
        bqas.append(bqa_features)

    bqas = np.concatenate(bqas, axis=0)
    images_features = np.concatenate(images_features, axis=0)

    if load_masks:
        masks = np.concatenate(masks, axis=0)

    return images_features, bqas, masks

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