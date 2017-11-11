"""Merge multiple DiskDatasets.

$ python -m bin.merge_datasets \
    --input_dir=/tmp/dir1 --input_dir=/tmp/dir2 --output_dir=/tmp/dir3
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging

import storage


def main(args):
    input_datasets = [storage.DiskDataset(path) for path in args.input_dirs]
    output_dataset = storage.DiskDataset(args.output_dir)

    total_input_images = sum(len(dataset.metadata()) for dataset in input_datasets)
    logging.info(
            "Merging %d input datasets with a combined %d images.",
            len(input_datasets),
            total_input_images)

    storage.merge_datasets(
        input_datasets,
        output_dataset,
        remove_existing_images=args.remove_existing_images)

    logging.info(
            "%s now contains %d images",
            output_dataset.base_dir,
            len(output_dataset.metadata()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dirs',
        type=str, required=True, action='append',
        help='Base directory for inputs to merge.')
    parser.add_argument(
        '--output_dir',
        type=str, required=True,
        help='Base directory for output of merge.')
    parser.add_argument(
        '--remove_existing_images',
        type=bool, default=False,
        help='If true and 2 images conflict, the last image wins.')
    logging.basicConfig(level=logging.INFO)
    main(parser.parse_args())
