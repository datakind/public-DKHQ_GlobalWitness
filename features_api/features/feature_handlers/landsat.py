"""Fetch image patches from Landsat 8."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ee
import numpy as np

from features.feature_handlers import ee_utils
from features.feature_handlers import registry

ee.Initialize()


def load_landsat8_images(start_date, end_date):
    """Construct ee.Image for each Landsat 8 image available."""
    image_collection = ee.ImageCollection(
        "LANDSAT/LC8_L1T_32DAY_TOA").filterDate(start_date, end_date)
    feature_collection = image_collection.getInfo()

    result = []
    for feature in feature_collection['features']:
        image_id = feature['id']

        # Warning: This image collection has bands [B1, ... B11, BQA]. The BQA
        # band has type uint16, while B1...B11 have type float. The BQA band
        # will be bit-interpreted as float32, but should be bit-casted to
        # uint16 before use.
        bands = [band['id'] for band in feature['bands']]

        date = feature['properties']['system:index']
        result.append({
            "image": ee.Image(image_id).select(bands),
            "bands": bands,
            "date": date
        })
    return result


@registry.register_feature_handler("landsat8")
def get_landsat8_image_patch(feature_spec,
                             center,
                             patch_size,
                             meters_per_pixel):
    """Downloads Landsat 8 image patches.

    Args:
        feature_names: dict. May contain "start_date" and/or "end_date",
            strings formatted as YYYY-MM-DD indicating date range to download
            pictures for. Defaults to [2010, 2018).
        center: (longitude, latitude). Center of image patch.
        patch_size: int. height and width of square patch to extract, in
            pixels.
        meters_per_pixel. Number of m^2 per pixel.

    Returns:
        image: np.array of shape [width, height, num_bands, num_dates].
        metadata: dict contains keys "bands" (list of strings, color band
            names) and "dates" (list of "YYYYMMDD" strings, dates photos were
            taken).
    """
    start_date = feature_spec.get("start_date", "2010-01-01")
    end_date = feature_spec.get("end_date", "2018-01-01")
    assert start_date < end_date
    images = load_landsat8_images(start_date, end_date)

    circle = ee_utils.create_circle(center=center, radius_meters=(
        patch_size * meters_per_pixel / 2))
    circle_coordinates = circle.coordinates().getInfo()

    result = []
    bands = None
    metadata = None
    dates = []
    for image in images:
        raster_image, raster_image_metadata = ee_utils.download_map_tile(
            image["image"], circle_coordinates, meters_per_pixel)
        result.append(raster_image)

        if bands is None:
            bands = image["bands"]
        assert bands == image["bands"]

        if metadata is None:
            metadata = raster_image_metadata
        assert metadata == raster_image_metadata

        dates.append(image["date"])

    result = np.stack(result, axis=-1)
    result[np.isneginf(result)] = np.nan
    metadata["bands"] = bands
    metadata["dates"] = dates

    return result, metadata
