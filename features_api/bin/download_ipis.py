"""Download image patches for IPIS dataset

Only contains images for which masks exist. Uses duckworthd@'s masks and the
IPIS Last Visit dataset.

$ python -m bin.download_ipis --base_dir=/tmp/ipis
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging

from features.feature_handlers import ee_utils
from features.feature_handlers import mask
import features
import parallel
import storage


def merge_dicts(*args):
    result = {}
    for arg in args:
        result.update(arg)
    return result


def add_image_to_dataset(dataset, location, feature_spec):
    """Add a single image to 'dataset'."""
    if dataset.has_image(location.pcode, feature_spec["source"]):
        return

    logging.info("Loading {}/{}.".format(location.pcode,
                                         feature_spec["source"]))
    # longitude, latitude.
    center = (location.geometry.x, location.geometry.y)

    image, metadata = features.get_image_patch(
        feature_spec=feature_spec,
        center=center,
        patch_size=100,
        meters_per_pixel=30)

    dataset.add_image(location.pcode, feature_spec["source"], image,
                      merge_dicts(feature_spec, metadata))


def main(args):
    dataset = storage.DiskDataset(args.base_dir)

    # Fetch mining masks, locations.
    masks = ee_utils.load_feature_collection_from_fusion_table(
        mask.FUSION_TABLES["duckworthd"])
    locations = ee_utils.load_feature_collection_from_fusion_table(
        "ft:1AFPyNO4MpeeV9TAD-dJj7xsnhS4splJ4DHVZnobG")
    locations = locations[locations.pcode.isin(masks.pcode)]
    assert len(locations) == len(locations.pcode.unique())

    # Fetch features for each location with a valid mask.
    logging.info("Loading features for {} locations.".format(len(locations)))
    feature_specs = [
        {
            "source": "mask",
            "table": "duckworthd"
        },
        {
            "source": "landsat8",
            "start_date": "2010-01-01",
            "end_date": "2018-01-01"
        }
    ]

    add_image_args_tuples = [
        (dataset, location, feature_spec)
        for feature_spec in feature_specs
        for _, location in locations.iterrows()
    ]

    for args_tuple in add_image_args_tuples:
        add_image_to_dataset(*args_tuple)

    # TODO(duckworthd): Don't do this until DiskDataset is thread-safe.
    # parallel.parallel_map(
    #       add_image_to_dataset, add_image_args, max_threads=8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_dir',
        type=str, required=True,
        help='Base directory for files.')
    logging.basicConfig(level=logging.INFO)
    main(parser.parse_args())
