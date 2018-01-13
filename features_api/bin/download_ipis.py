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
import time

from features.feature_handlers import ee_utils
from features.feature_handlers import mask
from gevent import monkey
import features
import gevent
import storage

monkey.patch_all()


def merge_dicts(*args):
    result = {}
    for arg in args:
        result.update(arg)
    return result


def with_retries(name, thunk, num_retries, timeout):
    err = gevent.Timeout()
    for i in range(num_retries):
        try:
            return thunk()
        except Exception as err:
            logging.info("Failed to execute {}. Retrying in {} seconds. Exception: {}.".format(name, timeout, err))
            time.sleep(timeout)
            timeout *= 2.0
    raise err


def add_image_to_dataset(dataset, location, feature_spec, lock):
    # longitude, latitude.
    center = (location.geometry.x, location.geometry.y)

    # Download image.
    name = "{}/{}".format(location.pcode, feature_spec["source"])
    logging.info("Loading {}.".format(name))
    thunk = lambda: features.get_image_patch(
        feature_spec=feature_spec,
        center=center,
        patch_size=100,
        meters_per_pixel=30)
    image, metadata = with_retries(name, thunk, num_retries=5, timeout=1.0)

    logging.info("Saving {} to disk.".format(name))
    lock.acquire()
    # Delete old image if there is one.
    if dataset.has_image(location.pcode, feature_spec["source"]):
        dataset.remove_image(location.pcode, feature_spec["source"])

    # Add image to dataset.
    dataset.add_image(location.pcode, feature_spec["source"], image,
        merge_dicts(feature_spec, metadata))
    lock.release()

    logging.info("Finished {}.".format(name))


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
            "start_date": "2016-10-01",
            "end_date": "2017-10-01"
        }
    ]

    # Download with gevent.
    pool = gevent.pool.Pool(args.max_concurrent_requests)
    lock = gevent.lock.Semaphore()
    for _, location in locations.iterrows():
        for feature_spec in feature_specs:
            pool.spawn(add_image_to_dataset, dataset, location, feature_spec, lock)
    pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_dir',
        type=str, required=True,
        help='Base directory for files.')
    parser.add_argument(
        '--max_concurrent_requests',
        type=int, default=5,
        help='Maximum number of concurrent RPCs. Use <= 5 for Earth Engine.')
    logging.basicConfig(level=logging.INFO)
    main(parser.parse_args())
