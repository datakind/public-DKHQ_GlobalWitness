"""An on-disk format backed by bcolz."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import json
import os
import shutil
import time

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


def merge_datasets(
        input_datasets,
        output_dataset,
        remove_existing_images=False):
    for input_dataset in input_datasets:
        for _, row in input_dataset.metadata().iterrows():
            location_id = row['location_id']
            source_id = row['source_id']
            metadata = row['metadata']
            image = input_dataset.load_image(location_id, source_id)
            if (remove_existing_images and
                    output_dataset.has_image(location_id, source_id)):
                output_dataset.remove_image(location_id, source_id)
            output_dataset.add_image(location_id, source_id, image, metadata)


class DiskDataset(object):
    """An on-disk image dataset backed by bcolz.

    Methods that modify the base directory are thread-safe.

    Attributes:
        base_dir: string. Directory under which all data is stored.
        metadata: pd.DataFrame. Metadata for images available.
    """

    def __init__(self, base_dir, lock_timeout_sec=0):
        """Initializes a DiskDataset.

        Args:
            base_dir: str. Path to directory containing dataset.
            lock_timeout_sec: int or float. Number of seconds to wait when
                attempting to a acquire a lock on 'base_dir'.
        """
        self._base_dir = base_dir
        self._lock_timeout_sec = lock_timeout_sec

    def _lock_base_dir(self):
        """Create a new lock."""
        return DiskDatasetLock(self, timeout_sec=self._lock_timeout_sec)

    @property
    def base_dir(self):
        """Base directory for all images and metadata."""
        return self._base_dir

    def metadata(self):
        """Loads pd.DataFrame containing metadata about images stored within.

        Columns:
            location_id: str. identifier for this site location.
            source_id: str. identifier for service that provided these
                features.
            metadata: dict. Additional source- or location-specific attributes.
        """
        return load_metadata(self.base_dir)

    def image_metadata(self, location_id, source_id):
        """Retrieves metadata for a specific image."""
        metadata = self.metadata()
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

    def load_images(self, source_id):
        """Load all images from a particular source.

        Yields:
            image: np.array
            row: pd.Series. Contains location_id, source_id, and metadata.
        """
        metadata = self.metadata()
        metadata = metadata[metadata["source_id"] == source_id]
        for _, row in (metadata
                       .sort_values(["source_id", "location_id"])
                       .iterrows()):
            yield self.load_image(row["location_id"], row["source_id"]), row

    def add_image(self, location_id, source_id, image, metadata=None):
        """Store an image.

        Fails if image with same location, source id already in this dataset.
        This operation is thread-safe.
        """
        with self._lock_base_dir():
            assert self.image_metadata(location_id, source_id) is None
            metadata = metadata or {}
            image_metadata = self._add_metadata(
                location_id, source_id, metadata)
            path = os.path.join(self.base_dir, image_path(image_metadata))
            save_image(image, path)

    def _add_metadata(self, location_id, source_id, metadata):
        """Stores a new metadata row.

        This operation is not thread-safe.
        """
        all_metadata = self.metadata()
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
        """Removes an image from the dataset.

        This operation is thread-safe.
        """
        with self._lock_base_dir():
            image_metadata = self.image_metadata(location_id, source_id)
            assert image_metadata is not None
            self._remove_metadata(location_id, source_id)
            path = os.path.join(self.base_dir, image_path(image_metadata))
            remove_image(path)

    def _remove_metadata(self, location_id, source_id):
        """Remove an image's metadata row.

        This operation is not thread-safe.
        """
        metadata = self.metadata()
        condition = metadata.location_id == location_id
        condition &= metadata.source_id == source_id
        metadata = metadata[~condition]
        save_metadata(metadata, self.base_dir)

    def has_image(self, location_id, source_id):
        """Checks if image is registered in metadata."""
        return self.image_metadata(location_id, source_id) is not None


class DiskDatasetLock(object):
    """A context manager for locking a DiskDataset."""

    def __init__(self, dataset, timeout_sec=0):
        self._dataset = dataset
        self._timeout_sec = timeout_sec

    @property
    def lockfile_path(self):
        return os.path.join(self._dataset.base_dir, "LOCK")

    def __enter__(self):
        start = time.time()
        while True:
            try:
                # Create directory for lockfile if it doesn't exists already.
                lock_dir = os.path.dirname(self.lockfile_path)
                if not os.path.exists(lock_dir):
                    os.makedirs(lock_dir)

                file_handle = os.open(self.lockfile_path,
                                      os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except OSError as err:
                if err.errno != errno.EEXIST:
                    # File already exists. Wait until it gets deleted.
                    raise
            else:
                # File does not exist. Write to it.
                with os.fdopen(file_handle, 'w') as file_obj:
                    file_obj.write("This directory is locked.")
                return

            # Check if we've waited too long.
            seconds_elapsed = time.time() - start
            if seconds_elapsed > self._timeout_sec:
                raise LockTimeoutException(
                    ("Waited > {} seconds to create lockfile {}. If you're "
                     "certain there shouldn't be a lock on this Dataset "
                     "directory, delete this file and try again.").format(
                        seconds_elapsed, self.lockfile_path))

    def __exit__(self, exc_type, exc_value, traceback):
        os.remove(self.lockfile_path)


class LockTimeoutException(Exception):
    """Exception raises when a lock has waited too long."""
    pass
