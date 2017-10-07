"""Python binary for downloading image masks.

Overview
========

This binary downloads rasterized image masks around mining site locations
defined in --mining_site_masks_shapefile. One mask is produced per mining site
location in --mining_site_locations_shapefile that has at least one entry in
the corresponding masks file (all others are dropped).

Each rasterized image is an N x N binary array centered at a mining site
location. Use --roi_radius_meters to control how many meters about the site to
capture and --resolution_square_meters to describe how many square meters are
assigned per pixel.

TODO(duckworthd): How are masks stored?

Usage
=====

```shell
$ python duckworthd/download_mining_masks_main.py \
    --mining_site_masks_shapefile="mining_masks.shp" \
    --mining_site_locations_shapefile="mining_locations.shp"
```
"""

# Set root directory to nearest parent folder for this file
import argparse
import logging
import os
import sys

def git_root(current_dir=None):
    """Find root directory for a github repo above 'current_dir'.

    Args:
        current_dir: str. Path to directory within a git repo.
            If None, defaults to the current working directory.

    Returns:
        Path to parent directory containing '.git'.

    Raises:
        ValueError: If no parent directory contains '.git'.
    """
    result = current_dir or os.getcwd()
    while True:
        if '.git' in os.listdir(result):
            return result
        if result == "/":
            raise ValueError("Could not find parent directory containing .git.")
        result = os.path.dirname(result)

def maybe_add_to_sys_path(path):
    """Add 'path' to 'sys.path' if it's not already there."""
    if path in sys.path:
        return
    sys.path.append(path)

maybe_add_to_sys_path(git_root())

import ee; ee.Initialize()
import geopandas as gpd
import shapely

from duckworthd import mining


def create_earth_engine_mask(geodataframe, key=None):
    """Construct ee.Image mask based on a GeoDataFrame's contents."""
    geodataframe = geodataframe.copy()
    if key is None:
        key = 'MASK_VALUE'
        geodataframe[key] = 1.0
    feature_collection_ee = mining.geodataframe_to_earthengine(geodataframe)
    image_ee = feature_collection_ee.reduceToImage(
        properties=[key],
        reducer=ee.Reducer.first())
    return image_ee


def create_earth_engine_roi(point_as_shapely, buffer_radius_in_meters):
    """Constructs a round ee.Geometry.Polygon around a shapely point."""
    point_as_geojson = shapely.geometry.mapping(point_as_shapely)
    point_as_earth_engine = ee.Geometry(point_as_geojson)
    buffer_as_earth_engine = ee.Geometry.buffer(
        point_as_earth_engine, buffer_radius_in_meters)
    return buffer_as_earth_engine


def main(args):
  # Load mining site locations, masks.
  mining_site_masks = gpd.read_file(args.mining_site_masks_shapefile)
  mining_site_locations = gpd.read_file(args.mining_site_locations_shapefile)
  logging.info("Loaded %d mining site locations, %d mining site masks"
      % (len(mining_site_locations), len(mining_site_masks)))

  # Filter mining site locations to places we have masks for.
  #
  # TODO(duckworthd): Should also grab locations that have been reviewed and
  # don't have visible mining sites. These locations won't have masks.
  mining_site_locations = mining_site_locations[
      mining_site_locations['pcode'].isin(mining_site_masks['pcode'])]

  # Construct a binary ee.Image that's 1.0 wherever there's a mine.
  mining_site_masks_ee = create_earth_engine_mask(mining_site_masks)

  for idx, row in mining_site_locations.iterrows():
    logging.debug("Loading mask for row %s." % idx)

    # Construct circle around mining site's location.
    roi = create_earth_engine_roi(row.geometry, args.roi_radius_meters)

    # Get coordinates for this polygon.
    roi = roi.coordinates().getInfo()

    # Download mask.
    tile = mining.load_map_tile_containing_roi(
        mining_site_masks_ee, roi, scale=args.resolution_square_meters)

    # TODO(duckworthd): Write 'tile' to disk somewhere...
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--mining_site_masks_shapefile',
      type=str, required=True,
      help='Path to shapefile containing mining site masks. Must contain "pcode" column that identifies which mine each mask belongs to.')
  parser.add_argument(
      '--mining_site_locations_shapefile',
      type=str, required=True,
      help='Path to shapefile containing mining site locations. Must contain "pcode" column that uniquely identifies each mine.')
  parser.add_argument(
      '--roi_radius_meters',
      type=int,
      default=1500,
      help='Radius in meters in resulting mask around mining site location.')
  parser.add_argument(
      '--resolution_square_meters',
      type=int,
      default=30,
      help='Number of square meters per pixel in resulting mask.')

  args = parser.parse_args()
  main(args)
