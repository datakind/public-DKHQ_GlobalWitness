
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from features import features
from features import feature_handlers


class FeaturesTests(unittest.TestCase):

    def setUp(self):
        super(FeaturesTests, self).setUp()
        self.center = (-1.026223, 27.307420)  # Mondini, DRC
        self.patch_size = 100  # 100x100 pixels
        self.meters_per_pixel = 30  # 30 m^2 per pixel

    def test_get_image_patch_mask(self):
        image, metadata = features.get_image_patch(
            {"source": "mask"},
            self.center, self.patch_size,
            self.meters_per_pixel)
        self.assertEqual((102, 101), image.shape)
        self.assertEqual(metadata, {})

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
        self.assertEqual((102, 101, 11, 2), image.shape)
        self.assertEqual(len(metadata["bands"]), 11)
        self.assertEqual(len(metadata["dates"]), 2)


if __name__ == '__main__':
    unittest.main()
