
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np

from features.feature_handlers import ee_utils


class EEUtilsTests(unittest.TestCase):

    def setUp(self):
        super(EEUtilsTests, self).setUp()
        self.center = (27.307420, -1.026223)  # (lon, lat) of Mondini, DRC
        self.patch_size = 100  # 100x100 pixels
        self.meters_per_pixel = 30  # 30 m^2 per pixel

    def test_download_pixel_centers(self):
        coordinates, metadata = ee_utils.download_pixel_centers(
            self.center, self.patch_size, self.meters_per_pixel)
        self.assertEqual((102, 101, 2), coordinates.shape)
        self.assertTrue(
            np.all(coordinates[..., 0] < metadata["bottom_right"][0]))
        self.assertTrue(np.all(coordinates[..., 0] > metadata["top_left"][0]))
        self.assertTrue(
            np.all(coordinates[..., 1] > metadata["bottom_right"][1]))
        self.assertTrue(np.all(coordinates[..., 1] < metadata["top_left"][1]))


if __name__ == '__main__':
    unittest.main()
