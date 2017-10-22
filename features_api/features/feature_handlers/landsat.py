"""Fetch image patches from Landsat 8."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ee
import numpy as np

from features.feature_handlers import ee_utils

ee.Initialize()


def load_landsat8_images(start_date, end_date):
    """Construct ee.Image for each Landsat 8 image available."""
    image_collection = ee.ImageCollection(
        "LANDSAT/LC8_L1T_32DAY_TOA").filterDate(start_date, end_date)
    feature_collection = image_collection.getInfo()

    result = []
    for feature in feature_collection['features']:
        image_id = feature['id']
        bands = [band['id'] for band in feature['bands']]
        date = feature['properties']['system:index']
        result.append({
            "image": ee.Image(image_id),
            "bands": bands,
            "date": date
        })
    return result


def get_landsat8_image_patch(feature_spec,
                             center,
                             patch_size,
                             meters_per_pixel):
    """Downloads Landsat 8 image patches.

    Args:
        feature_names: dict. May contain "start_date" and/or "end_date",
            strings formatted as YYYY-MM-DD indicating date range to download
            pictures for. Defaults to [2010, 2018).
        center: (latitude, longitude). Center of image patch.
        patch_size: int. height and width of square patch to extract, in
            pixels.
        meters_per_pixel. Number of m^2 per pixel.

    Returns:
        image: np.array of shape [height, width, num_bands, num_dates].
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
    dates = []
    for image in images:
        result.append(ee_utils.download_map_tile(
            image["image"], circle_coordinates, meters_per_pixel))
        if bands is None:
            bands = image["bands"]
        assert bands == image["bands"]
        dates.append(image["date"])

    result = np.stack(result, axis=-1)
    result[np.isneginf(result)] = np.nan

    return result, {"bands": bands, "dates": dates}
