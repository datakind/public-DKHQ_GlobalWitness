"""A bounding box for north-up images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


_BoundingBox = collections.namedtuple(
    'BoundingBox',
    ['min_longitude', 'min_latitude', 'max_longitude', 'max_latitude'])


class BoundingBox(_BoundingBox):
    """Bounding box for a north-up image."""

    @classmethod
    def from_metadata(cls, metadata):
        for key in ["top_left", "bottom_right"]:
            assert key in metadata, "%s isn't in metadata. Unable to infer image bounding box." % (key,)

        min_lat = metadata["bottom_right"][1]
        max_lat = metadata["top_left"][1]
        min_lon = metadata["top_left"][0]
        max_lon = metadata["bottom_right"][0]

        assert min_lat < max_lat, 'min_lat = %s should be less than max_lat = %s' % (min_lat, max_lat)
        assert min_lon < max_lon, 'min_lon = %s should be less than max_lon = %s' % (min_lon, max_lon)

        return cls(min_longitude=min_lon, max_longitude=max_lon,
                   min_latitude=min_lat, max_latitude=max_lat)

    @property
    def top_left(self):
        return (self.min_longitude, self.max_latitude)

    @property
    def top_right(self):
        return (self.max_longitude, self.max_latitude)

    @property
    def bottom_right(self):
        return (self.max_longitude, self.min_latitude)

    @property
    def bottom_left(self):
        return (self.min_latitude, self.min_latitude)

    @property
    def center(self):
        center_longitude = self.min_longitude + (self.max_longitude - self.min_longitude) / 2
        center_latitude = self.min_latitude + (self.max_latitude - self.min_latitude) / 2
        return (center_longitude, center_latitude)
