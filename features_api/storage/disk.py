"""An on-disk format backed by bcolz."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import shutil

import bcolz
import pandas as pd


def metadata_file_path(base_dir):
    """Path to metadata file."""
    return os.path.join(base_dir, "metadata.json")


def load_metadata(base_dir):
    """Load metadata DataFrame from disk."""
    fpath = metadata_file_path(base_dir)
    if not os.path.exists(fpath):
        return pd.DataFrame(columns=["location_id", "source_id", "metadata"])
    with open(fpath) as f:
        return pd.DataFrame(json.load(f))


def save_metadata(metadata, base_dir):
    """Save metadata DataFrame to disk."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    fpath = metadata_file_path(base_dir)
    metadata.to_json(fpath)


def image_path(metadata):
    """Constructs relative path to an image wrt base_dir."""
    return "{}/{}".format(
        metadata['location_id'],
        metadata['source_id'])


def load_image(path):
    """Load a bcolz image stored at a path."""
    return bcolz.open(path)[:]


def save_image(image, path):
    """Store a single image to disk."""
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    c = bcolz.carray(image, rootdir=path, mode='w')
    c.flush()


def remove_image(path):
    """Removes an image on disk."""
    shutil.rmtree(path)
    parent_dir = os.path.dirname(path)
    if not os.listdir(parent_dir):
        os.rmdir(parent_dir)


class DiskDataset(object):
    """An on-disk image dataset backed by bcolz.
    
    Attributes:
        base_dir: string. Directory under which all data is stored.
        metadata: pd.DataFrame. Metadata for images available.
    """

    def __init__(self, base_dir):
        self._base_dir = base_dir

    @property
    def base_dir(self):
        """Base directory for all images and metadata."""
        return self._base_dir

    @property
    def metadata(self):
        """pd.DataFrame containing metadata about images stored within.

        Columns:
            location_id: str. identifier for this site location.
            source_id: str. identifier for service that provided these
                features.
            metadata: dict. Additional source- or location-specific attributes.
        """
        return load_metadata(self.base_dir)

    def image_metadata(self, location_id, source_id):
        """Retrieves metadata for a specific image."""
        metadata = self.metadata
        condition = metadata.location_id == location_id
        condition &= metadata.source_id == source_id
        result = metadata[condition]
        if len(result) == 0:
            return None
        assert len(result) == 1
        return result.iloc[0]

    def load_image(self, location_id, source_id):
        """Load an image.

        Args:
            location_id: str. identifier for this site location.
            source_id: str. identifier for service that provided these
                features.

        Returns:
            np.array of shape [width, height, ...]
        """
        image_metadata = self.image_metadata(location_id, source_id)
        path = os.path.join(self.base_dir, image_path(image_metadata))
        return load_image(path)

    def add_image(self, location_id, source_id, image, metadata=None):
        """Store an image.

        Fails if image with same location, source id already in this dataset.
        """
        assert self.image_metadata(location_id, source_id) is None
        metadata = metadata or {}
        image_metadata = self._add_metadata(location_id, source_id, metadata)
        path = os.path.join(self.base_dir, image_path(image_metadata))
        save_image(image, path)

    def _add_metadata(self, location_id, source_id, metadata):
        """Stores a new metadata row."""
        all_metadata = self.metadata
        result = pd.Series({
            "location_id": location_id,
            "source_id": source_id,
            "metadata": metadata,
        })
        all_metadata = all_metadata.append(
            pd.DataFrame.from_records([result]),
            ignore_index=True)
        save_metadata(all_metadata, self.base_dir)
        return result

    def remove_image(self, location_id, source_id):
        """Removes an image from the dataset."""
        image_metadata = self.image_metadata(location_id, source_id)
        assert image_metadata is not None
        self._remove_metadata(location_id, source_id)
        path = os.path.join(self.base_dir, image_path(image_metadata))
        remove_image(path)

    def _remove_metadata(self, location_id, source_id):
        """Remove an image's metadata row."""
        metadata = self.metadata
        condition = metadata.location_id == location_id
        condition &= metadata.source_id == source_id
        metadata = metadata[~condition]
        save_metadata(metadata, self.base_dir)

    def has_image(self, location_id, source_id):
        """Checks if image is registered in metadata."""
        return self.image_metadata(location_id, source_id) is not None
