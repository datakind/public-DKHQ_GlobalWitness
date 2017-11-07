"""Utilities for working with the Google Earth Engine API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import shutil
import string
import cStringIO as StringIO
import tempfile
import time
import urllib2
import zipfile

import ee
import gdal
import geopandas as gpd
import numpy as np

ee.Initialize()


def create_circle(center, radius_meters):
    """Create a circle geometry around a (latitude, longitude)."""
    longitude, latitude = center
    point = ee.Geometry.Point([longitude, latitude])
    return ee.Geometry.buffer(point, radius_meters)


def load_feature_collection_from_fusion_table(path):
    """Load a FeatureCollection from Google Fusion Table."""
    assert path.startswith("ft:"), "Path is not a Fusion Table location."
    result_json = ee.FeatureCollection(path).getInfo()
    return gpd.GeoDataFrame.from_features(result_json)


def download_map_tile(image, roi_coordinates, meters_per_pixel):
    """Get rasterized image containing ROI from Earth Engine.

    Constructs a rasterized image tile subsetting 'image'. The image is large
    enough to fully contain the polygon described by 'roi', and will contain
    one pixel per 'meters_per_pixel' m^2 area.

    Args:
      image: ee.Image instance. To be used as mask. Must have exactly 1 band.
      roi_coordinates: Triple-nested list of floats, where lowest level is
        [longitude, latitude] pairs from 'coordinates' of a GeoJSON polygon.
      meters_per_pixel: int. Number of squared meters per pixel.

    Returns:
      numpy array of shape [N x M x K], where N is width, M is height, and K is
      number of bands.
    """
    # Generate a random filename.
    filename = ''.join(np.random.choice(list(string.ascii_letters), size=10))

    # Fetch URL to download image from.
    url = ee.data.makeDownloadUrl(
        ee.data.getDownloadId({
            'image': image.serialize(),
            'scale': '%d' % meters_per_pixel,
            'filePerBand': 'false',
            'name': filename,
            'region': roi_coordinates,
        }))

    # Download image. Retry a few times if too many requests.
    timeout = 0.5
    request = urllib2.Request(url, headers={"User-Agent": "datakind-mining-detection"})
    contents = urllib2.urlopen(request).read()

    # Attempt to read the zipfile.
    local_zip = StringIO.StringIO(contents)
    with zipfile.ZipFile(local_zip) as local_zipfile:
        local_tif_dir = tempfile.mkdtemp()
        local_tif_filename = local_zipfile.extract(
            filename + '.tif', local_tif_dir)

    # Read image into memory. Result has shape [x, y, color bands].
    dataset = gdal.Open(local_tif_filename, gdal.GA_ReadOnly)
    bands = [dataset.GetRasterBand(i + 1).ReadAsArray()
             for i in range(dataset.RasterCount)]
    shutil.rmtree(local_tif_dir)

    return np.stack(bands, axis=2)


def geodataframe_to_earthengine(geodataframe):
    """Converts a GeoDataFrame to an ee.FeatureCollection."""
    geojson_str = geodataframe.to_json()
    geojson = json.loads(geojson_str)
    return geojson_to_earthengine(geojson)


def geojson_to_earthengine(geojson):
    """Converts a GeoJSON dict to an Earth Engine type.

    Args:
        geojson: GeoJSON-supported object as a nested dict/list/tuple.

    Returns:
        A matching type that Earth Engine understands (e.g.
        ee.FeatureCollection, ee.Geometry.Point).
    """
    if isinstance(geojson, dict):
        if 'type' not in geojson:
            raise ValueError(
                "Not 'type' attribute in geojson: %s" % (geojson,))
        if geojson['type'] == 'FeatureCollection':
            return ee.FeatureCollection(
                geojson_to_earthengine(geojson['features']))
        elif geojson['type'] == 'Feature':
            return ee.Feature(
                geojson_to_earthengine(geojson['geometry']),
                geojson['properties'])
        elif geojson['type'] == 'Point':
            return ee.Geometry.Point(coords=geojson['coordinates'])
        elif geojson['type'] == 'Polygon':
            return ee.Geometry.Polygon(
                coords=geojson['coordinates'],
                geodesic=geojson.get('geodesic', None))
        raise ValueError("Unsupported GeoJSON dict type: %s" % geojson['type'])
    elif isinstance(geojson, list):
        return [geojson_to_earthengine(element) for element in geojson]
    elif isinstance(geojson, tuple):
        return tuple(geojson_to_earthengine(element) for element in geojson)
    elif isinstance(geojson, (int, float, str, unicode)):
        return geojson
    else:
        raise ValueError("Unable to parse type: %s" % type(geojson))
