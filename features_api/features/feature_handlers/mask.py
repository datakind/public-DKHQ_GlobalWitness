"""Utility for downloading mining masks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ee
import numpy as np

from features.feature_handlers import ee_utils

ee.Initialize()


def load_mask_image():
    """Load ee.Image containing mining masks."""
    geodataframe = ee_utils.load_feature_collection_from_fusion_table(
        "ft:1C4cfhvOZjqM6NRXPZjN4cBkCjVi9Ckl-tkYWvaMq")
    geodataframe = geodataframe[["geometry"]]
    geodataframe["MASK_VALUE"] = 1.0
    feature_collection_ee = ee_utils.geodataframe_to_earthengine(geodataframe)
    image_ee = feature_collection_ee.reduceToImage(
        properties=["MASK_VALUE"], reducer=ee.Reducer.first())
    return image_ee


def get_mask_image_patch(feature_spec, center, patch_size, meters_per_pixel):
    """Downloads binary mining site mask.

    Args:
        feature_spec: dict. Ignored.
        center: (latitude, longitude). Center of image patch.
        patch_size: int. height and width of square patch to extract, in
            pixels.
        meters_per_pixel. Number of m^2 per pixel.

    Returns:
        image: np.array of shape [height, width].
        metadata: empty dict.
    """
    image = load_mask_image()
    circle = ee_utils.create_circle(center=center, radius_meters=(
        patch_size * meters_per_pixel / 2))
    circle_coordinates = circle.coordinates().getInfo()
    result = ee_utils.download_map_tile(
        image, circle_coordinates, meters_per_pixel)
    image = np.reshape(result, [result.shape[0], result.shape[1]])
    return image, {}