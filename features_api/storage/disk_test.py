"""Tests for storage.disk"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from storage import disk


class DiskDatasetTests(unittest.TestCase):

    def setUp(self):
        super(DiskDatasetTests, self).setUp()
        self.base_dir = tempfile.mkdtemp()
        self.dataset = disk.DiskDataset(self.base_dir)

        self.metadata = pd.DataFrame.from_records([
            {"location_id": "loc0", "source_id": "source0", "metadata": {}},
            {"location_id": "loc1", "source_id": "source1",
                "metadata": {"meta": "data"}},
            {"location_id": "loc1", "source_id": "source2", "metadata": {}},
            {"location_id": "loc1", "source_id": "source3", "metadata": {}},
            {"location_id": "loc0", "source_id": "source1",
                "metadata": {"meta": "data"}},
        ])

        self.image = np.zeros([10, 10])

    def tearDown(self):
        shutil.rmtree(self.base_dir)

    def test_metadata_empty(self):
        self.assertEqual(0, len(self.dataset.metadata))

    def test_metadata_nonempty(self):
        disk.save_metadata(self.metadata, self.base_dir)
        self.assertEqual(5, len(self.dataset.metadata))

    def test_image_metadata(self):
        disk.save_metadata(self.metadata, self.base_dir)
        row = self.dataset.image_metadata("loc0", "source1")
        self.assertEqual({"meta": "data"}, row.metadata)

    def test_load_image(self):
        # Update metadata on disk.
        disk.save_metadata(self.metadata, self.base_dir)

        # Store a random image for one of the rows.
        image_metadata = self.metadata.iloc[-1]
        image_path = os.path.join(
            self.base_dir, disk.image_path(image_metadata))
        disk.save_image(self.image, image_path)

        # Load that image.
        image = self.dataset.load_image(
            image_metadata.location_id, image_metadata.source_id)

        np.testing.assert_allclose(self.image, image)

    def test_add_image_duplicate(self):
        disk.save_metadata(self.metadata, self.base_dir)
        with self.assertRaises(AssertionError):
            self.dataset.add_image("loc0", "source0", self.image, {})

    def test_add_image(self):
        self.dataset.add_image("loc0", "source0", self.image, {})

        self.assertEqual(1, len(self.dataset.metadata))
        np.testing.assert_allclose(
            self.image, self.dataset.load_image("loc0", "source0"))

        disk.save_metadata(self.metadata, self.base_dir)

    def test_remove_image(self):
        # Store metadata.
        disk.save_metadata(self.metadata, self.base_dir)

        # Store a random image for one of the rows.
        image_metadata = self.metadata.iloc[-1]
        image_path = os.path.join(
            self.base_dir, disk.image_path(image_metadata))
        disk.save_image(self.image, image_path)

        # Remove that image.
        self.dataset.remove_image(
            image_metadata.location_id, image_metadata.source_id)

        self.assertFalse(os.path.exists(image_path))
        self.assertEqual(len(self.metadata) - 1, len(self.dataset.metadata))

    def test_remove_image_missing(self):
        with self.assertRaises(AssertionError):
            self.dataset.remove_image("loc0", "source0")


if __name__ == '__main__':
    unittest.main()
