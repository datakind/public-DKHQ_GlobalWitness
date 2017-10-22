"""Library for retrieving image patches."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from features import feature_handlers


def get_image_patch(feature_spec, center, patch_size, meters_per_pixel=None):
    """Get image patch centered at 'center'.

    Args:
        feature_spec: dict with optional parameters for feature handler. Must
            have "source" parameter (e.g. "mask" or "landsat8").
        center: (latitude, longitude). Center for image patch.
        patch_size: int. Height and width of image patch, in pixels.
        meters_per_pixel: int or None. Number of m^2 per pixel. Defaults to 30.

    Returns:
        image: np.array. Image patch. Leading 2 dimensions are width and
            height. May contain additional dimensions for color band, time,
            etc.
        metadata: dict. Contains additional information about 'image'.
    """
    meters_per_pixel = meters_per_pixel or 30
    feature_handler = get_feature_handler(feature_spec["source"])
    return feature_handler(
        feature_spec,
        center, patch_size,
        meters_per_pixel=meters_per_pixel)


def get_feature_handler(feature_handler_name):
    """Get function to retrieve features."""
    global FEATURE_HANDLER_MAP
    return FEATURE_HANDLER_MAP[feature_handler_name]


FEATURE_HANDLER_MAP = {
    "mask": feature_handlers.get_mask_image_patch,
    "landsat8": feature_handlers.get_landsat8_image_patch,
}
