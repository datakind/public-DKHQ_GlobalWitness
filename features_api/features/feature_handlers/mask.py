"""Utility for downloading mining masks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ee
import numpy as np

from features.feature_handlers import ee_utils
from features.feature_handlers import registry

ee.Initialize()


FUSION_TABLES = {
    # Masks from mines_ipis250.
    "mines_ipis250": "ft:1C4cfhvOZjqM6NRXPZjN4cBkCjVi9Ckl-tkYWvaMq",

    # Masks by duckworthd@.
    "duckworthd": "ft:18cLkd0WGZEPWY8Mee2hRvKlyD0p6Ca_qgqIdBqdg",
}


def load_mask_image(table):
    """Load ee.Image containing mining masks."""
    table = FUSION_TABLES[table]
    geodataframe = ee_utils.load_feature_collection_from_fusion_table(table)
    geodataframe = geodataframe[["geometry"]]
    geodataframe["MASK_VALUE"] = 1.0
    feature_collection_ee = ee_utils.geodataframe_to_earthengine(geodataframe)
    image_ee = feature_collection_ee.reduceToImage(
        properties=["MASK_VALUE"], reducer=ee.Reducer.first())
    return image_ee


@registry.register_feature_handler("mask")
def get_mask_image_patch(feature_spec, center, patch_size, meters_per_pixel):
    """Downloads binary mining site mask.

    Args:
        feature_spec: dict. May contains "table", indicating which Fusion Table
            to draw masks from.
        center: (longitude, latitude). Center of image patch.
        patch_size: int. height and width of square patch to extract, in
            pixels.
        meters_per_pixel. Number of m^2 per pixel.

    Returns:
        image: np.array of shape [height, width].
        metadata: empty dict.
    """
    table = feature_spec.get("table", "duckworthd")
    image = load_mask_image(table)
    circle = ee_utils.create_circle(center=center, radius_meters=(
        patch_size * meters_per_pixel / 2))
    circle_coordinates = circle.coordinates().getInfo()
    result, metadata = ee_utils.download_map_tile(
        image, circle_coordinates, meters_per_pixel)
    image = np.reshape(result, [result.shape[0], result.shape[1]])
    return image, metadata
