import argparse
import storage
import random

def main(args):

    assert args.test_frac+args.train_frac+args.val_frac==1.0, "Fractions must add up to 1.0"

    dataset = storage.DiskDataset(args.data_input_path)

    split(dataset, args.data_output_path, args.train_frac, args.val_frac, args.test_frac)

def split(dataset, output_path, train_frac, val_frac, test_frac, source_id='landsat8'):

    images = list(dataset.load_images(source_id))

    n_images = len(images)
    n_train = int(round(n_images*train_frac))
    n_test = int(round(n_images*test_frac))
    n_val = int(round(n_images*val_frac))

    indices = list(range(n_images))
    random.shuffle(indices)

    assert n_train+n_test+n_val == n_images, "Error doesn't add up"

    if n_train>0:
        train_dataset = storage.DiskDataset(output_path+"_train")

        for train_index in indices[0:n_train]:
            image, image_metadata = images[train_index]
            mask = dataset.load_image(image_metadata['location_id'], 'mask')
            mask_metadata = dataset.image_metadata(image_metadata['location_id'], 'mask')

            copy_entry(train_dataset, mask, mask_metadata.to_dict(), image, image_metadata.to_dict(), source_id)

    if n_test>0:
        test_dataset = storage.DiskDataset(output_path+"_test")

        for test_index in indices[n_train:n_train+n_test]:
            image, image_metadata = images[test_index]
            mask = dataset.load_image(image_metadata['location_id'], 'mask')
            mask_metadata = dataset.image_metadata(image_metadata['location_id'], 'mask')

            copy_entry(test_dataset, mask, mask_metadata.to_dict(), image, image_metadata.to_dict(), source_id)

    if n_val>0:
        val_dataset = storage.DiskDataset(output_path+"_val")

        for val_index in indices[n_train+n_test:]:
            image, image_metadata = images[val_index]
            mask = dataset.load_image(image_metadata['location_id'], 'mask')
            mask_metadata = dataset.image_metadata(image_metadata['location_id'], 'mask')

            copy_entry(val_dataset, mask, mask_metadata.to_dict(), image, image_metadata.to_dict(), source_id)

def copy_entry(dataset, mask, mask_metadata, image, image_metadata, source_id):

    dataset.add_image(image_metadata['location_id'], source_id, image, image_metadata['metadata'])
    dataset.add_image(image_metadata['location_id'], 'mask', mask, mask_metadata['metadata'])

    return dataset

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_output_path',
        type=str,
        help='Data path')

    parser.add_argument(
        '--data_input_path',
        type=str,
        help='Path to storage api directory')
    parser.add_argument(
        '--train_frac',
        type=float,
        help='Fraction for the training set',
        default=0.0)
    parser.add_argument(
        '--test_frac',
        type=float,
        help='Fraction for the test set',
        default=0.0)
    parser.add_argument(
        '--val_frac',
        type=float,
        help='Fraction for the validation set',
        default=0.0)

    main(parser.parse_args())