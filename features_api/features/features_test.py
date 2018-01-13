
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np

from features import features
from features import feature_handlers


class FeaturesTests(unittest.TestCase):

    def assertAboveLeft(self, above_left, origin):
        """Assert a (lon, lat) pair is above-left."""
        self.assertLess(above_left[0], origin[0])     # left
        self.assertGreater(above_left[1], origin[1])  # above

    def assertBelowRight(self, below_right, origin):
        """Assert a (lon, lat) pair is below-right."""
        self.assertGreater(below_right[0], origin[0])  # right
        self.assertLess(below_right[1], origin[1])     # below

    def setUp(self):
        super(FeaturesTests, self).setUp()
        self.center = (27.307420, -1.026223)  # (lon, lat) of Mondini, DRC
        self.patch_size = 100  # 100x100 pixels
        self.meters_per_pixel = 30  # 30 m^2 per pixel

    def test_get_image_patch_mask(self):
        image, metadata = features.get_image_patch(
            {"source": "mask"},
            self.center, self.patch_size,
            self.meters_per_pixel)
        self.assertEqual((102, 101), image.shape)
        np.testing.assert_allclose(
                self.center, metadata["center"], atol=1e-3)
        self.assertAboveLeft(metadata["top_left"], metadata["center"])
        self.assertBelowRight(metadata["bottom_right"], metadata["center"])

    def test_get_image_patch_landsat8(self):
        image, metadata = features.get_image_patch(
            {
                "source": "landsat8",
                "start_date": "2013-04-01",
                "end_date": "2013-06-01"
            },
            self.center,
            self.patch_size,
            self.meters_per_pixel)
        self.assertEqual((102, 101, 12, 2), image.shape)
        self.assertEqual(len(metadata["bands"]), 12)
        self.assertEqual(len(metadata["dates"]), 2)
        np.testing.assert_allclose(
                self.center, metadata["center"], atol=1e-3)
        self.assertAboveLeft(metadata["top_left"], metadata["center"])
        self.assertBelowRight(metadata["bottom_right"], metadata["center"])


if __name__ == '__main__':
    unittest.main()
