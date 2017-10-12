"""Python binary for downloading image masks.

Overview
========

This binary downloads rasterized image masks around mining site locations
defined in --mining_site_masks_shapefile. One mask is produced per mining site
location in --mining_site_locations_shapefile that has at least one entry in
the corresponding masks file (all others are dropped).

Each rasterized image is an N x N binary array centered at a mining site
location. Use --roi_bounding_box_width to control how many meters about the
site to capture and --resolution_square_meters to describe how many square
meters are assigned per pixel.

Masks are stored in a folder with format matching the existing raw image data.
If there are existing images in the same directory, they will be overwritten
and the metadata will be updated.

```
image_root/
  <Integer IPIS Visit ID>/
    mining_masks/              # 4-D bcolz array
  ...
  <Integer IPIS Visit ID>/
    mining_masks/              # 4-D bcolz array
  metadata4.json               # Directory info.
```

Usage
=====

```shell
$ python duckworthd/download_mining_masks_main.py \
    --mining_site_masks_shapefile="mining_masks.shp" \
    --mining_site_locations_shapefile="mining_locations.shp" \
    --output_image_root="/tmp/masks"
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

import ee
ee.Initialize()
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from duckworthd import mining

# Name for this dataset.
MASK_COLLECTION_NAME = "mining_masks"

# pcodes associated with the mines_ipis250 dataset.
MINES_IPIS250_PCODES = [
    "codmine00862",
    "codmine00901",
    "codmine00906",
    "codmine01158",
    "codmine01671",
    "codmine01696",
    "codmine01740",
    "codmine01756",
    "codmine01771",
    "codmine01784",
    "codmine01782",
    "codmine01786",
    "codmine01788",
    "codmine01792",
    "codmine00037",
    "codmine00038",
    "codmine00039",
    "codmine00040",
    "codmine00041",
    "codmine00043",
    "codmine00044",
    "codmine00051",
    "codmine00085",
    "codmine00090",
    "codmine00098",
    "codmine00099",
    "codmine00623",
    "codmine01021",
    "codmine01097",
    "codmine01098",
    "codmine01106",
    "codmine01206",
    "codmine01394",
    "codmine01395",
    "codmine01402",
    "codmine01406",
    "codmine01408",
    "codmine01426",
    "codmine01433",
    "codmine01435",
    "codmine01461",
    "codmine01466",
    "codmine01467",
    "codmine01468",
    "codmine01484",
    "codmine01514",
    "codmine01525",
    "codmine01527",
    "codmine01531",
    "codmine01533",
    "codmine01545",
    "codmine01558",
    "codmine01564",
    "codmine01570",
    "codmine01574",
    "codmine01585",
    "codmine01586",
    "codmine01588",
    "codmine01589",
    "codmine01596",
    "codmine01599",
    "codmine01600",
    "codmine01601",
    "codmine01603",
    "codmine01612",
    "codmine01613",
    "codmine01617",
    "codmine01618",
    "codmine01619",
    "codmine01627",
    "codmine01631",
    "codmine01968",
    "codmine02599",
    "codmine00227",
    "codmine00643",
    "codmine00180",
    "codmine00646",
    "codmine00182",
    "codmine00507",
    "codmine00833",
    "codmine00651",
    "codmine00514",
    "codmine00658",
    "codmine00532",
    "codmine00534",
    "codmine00535",
    "codmine00004",
    "codmine00198",
    "codmine00195",
    "codmine00202",
    "codmine00203",
    "codmine00204",
    "codmine00205",
    "codmine00206",
    "codmine00207",
    "codmine00338",
    "codmine00333",
    "codmine00332",
    "codmine00209",
    "codmine00211",
    "codmine00214",
    "codmine00215",
    "codmine00016",
    "codmine00235",
    "codmine00236",
    "codmine01028",
    "codmine01029",
    "codmine01032",
    "codmine01031",
    "codmine01019",
    "codmine01021",
    "codmine01048",
    "codmine00367",
    "codmine00370",
    "codmine00373",
    "codmine00372",
    "codmine00537",
    "codmine00542",
    "codmine00241",
    "codmine00244",
    "codmine00245",
    "codmine00248",
    "codmine00251",
    "codmine00041",
    "codmine00037",
    "codmine00038",
    "codmine00046",
    "codmine00045",
    "codmine00044",
    "codmine00050",
    "codmine00051",
    "codmine00047",
    "codmine00048",
    "codmine00056",
    "codmine00055",
    "codmine00054",
    "codmine00053",
    "codmine00057",
    "codmine00068",
    "codmine00067",
    "codmine00066",
    "codmine00065",
    "codmine00064",
    "codmine00073",
    "codmine00072",
    "codmine00075",
    "codmine00074",
    "codmine00076",
    "codmine00080",
    "codmine00079",
    "codmine00078",
    "codmine00077",
    "codmine00379",
    "codmine00685",
    "codmine00558",
    "codmine00556",
    "codmine00396",
    "codmine00557",
    "codmine00699",
    "codmine00698",
    "codmine00398",
    "codmine00399",
    "codmine00401",
    "codmine00083",
    "codmine00973",
    "codmine00092",
    "codmine00097",
    "codmine00098",
    "codmine00101",
    "codmine00405",
    "codmine00407",
    "codmine01054",
    "codmine00102",
    "codmine00406",
    "codmine00707",
    "codmine02526",
    "codmine00415",
    "codmine00107",
    "codmine00277",
    "codmine00996",
    "codmine00593",
    "codmine01013",
    "codmine00445",
    "codmine00151",
    "codmine00174",
    "codmine00297",
    "codmine00788",
    "codmine00468",
    "codmine00467",
    "codmine00466",
    "codmine00475",
    "codmine00474",
    "codmine00480",
    "codmine00821",
    "codmine00996",
    "codmine00985",
    "codmine01321",
    "codmine01299",
    "codmine00055",
    "codmine01143",
    "codmine00102",
    "codmine01208",
    "codmine00101",
    "codmine00099",
    "codmine01206",
    "codmine00061",
    "codmine01171",
    "codmine00062",
    "codmine00065",
    "codmine00067",
    "codmine00073",
    "codmine00075",
    "codmine00396",
    "codmine01155",
    "codmine01251",
    "codmine01200",
    "codmine01174",
    "codmine01138",
    "codmine01140",
    "codmine01137",
    "codmine01021",
    "codmine01019",
    "codmine01193",
    "codmine01805",
    "codmine01024",
    "codmine01835",
    "codmine01029",
    "codmine01175",
    "codmine01796",
    "codmine01855",
    "codmine01807",
    "codmine01831",
    "codmine01834",
    "codmine01800",
    "codmine01801",
    "codmine01851",
    "codmine00833",
    "codmine01964",
    "codmine00041",
    "codmine02138",
    "codmine01925",
    "codmine01894",
    "codmine01869",
    "codmine01893",
    "codmine01888",
    "codmine00696",
    "codmine02162",
    "codmine00906",
    "codmine00332",
    "codmine00338"]


def initialize_logging(loglevel):
    """Set global logging level to 'loglevel'."""
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(level=numeric_level)


def create_earth_engine_mask(geodataframe, key=None):
    """Construct ee.Image mask based on a GeoDataFrame's contents."""
    geodataframe = geodataframe.copy()

    # Some mining site locations may have an undefined geometry. Filter those
    # out.
    geodataframe = geodataframe[pd.notnull(geodataframe.geometry)]

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


def load_feature_collection_from_fusion_table(path):
    """Load a FeatureCollection from Google Fusion Table."""
    assert path.startswith("ft:"), "Path is not a Fusion Table location."
    result_json = ee.FeatureCollection(path).getInfo()
    return gpd.GeoDataFrame.from_features(result_json)


def load_mines_ipis250():
    """Load mining site masks for 'mines_ipis250' dataset.

    Returns:
      GeoDataFrame containing mining site mask polygons and pcodes.
    """
    sinas_mining_site_masks = load_feature_collection_from_fusion_table(
        "ft:1C4cfhvOZjqM6NRXPZjN4cBkCjVi9Ckl-tkYWvaMq")
    sinas_mining_site_masks = sinas_mining_site_masks[["geometry"]]
    sinas_mining_site_masks["pcode"] = np.nan

    # Load pcodes for all mining sites in 'mines_ipis250'. Though we don't have
    # polygons to match these, we'll add it to 'mining_site_masks' so we can
    # filter by pcode.
    sinas_mining_site_pcodes = gpd.GeoDataFrame.from_dict(
        {"pcode": MINES_IPIS250_PCODES})
    sinas_mining_site_pcodes["geometry"] = np.nan

    return pd.concat([sinas_mining_site_masks, sinas_mining_site_pcodes], ignore_index=True)


def img_already_exists(image_root, vid):
    """Checks if an image and its metadata are already on disk."""
    fpath = create_fpath(vid)
    img_exists = os.path.exists(os.path.join(image_root, fpath))

    metadata = mining.load_metadata(image_root)
    metadata_exists = np.sum(metadata['fpath'] == fpath) == 1

    return (img_exists and metadata_exists)


def create_fpath(vid):
    """Creates the 'fpath' field for a given image."""
    return "%d/%s" % (vid, MASK_COLLECTION_NAME)


def create_img_metadata(img_shape, vid):
    """Constructs img_metadata for a single image."""
    return pd.Series({
        "bands": ['mask'],
        "collection": MASK_COLLECTION_NAME,

        # TODO(duckworthd): This should be the date of the satellite image upon
        # which this mask is based.
        "dates": ['20171001'],

        "dim": img_shape,
        "fpath": create_fpath(vid),
        "id": vid,
    })


def main(args):
    initialize_logging(args.loglevel)

    if args.roi_bounding_box_width % 2 != 0:
        raise ValueError(
            "--roi_bounding_box_width needs to be divisible by 2 as it will be used "
            "as a radius.")

    if args.roi_bounding_box_width % args.resolution_square_meters != 0:
        raise ValueError(
            "--roi_bounding_box_width needs to be divisible by "
            "--resolution_square_meters.")
    roi_box_dims = (args.roi_bounding_box_width //
                    args.resolution_square_meters,) * 2

    # Load mining site locations, masks.
    mining_site_masks = gpd.read_file(args.mining_site_masks_shapefile)
    mining_site_locations = gpd.read_file(args.mining_site_locations_shapefile)
    logging.info("Loaded %d mining site locations, %d mining site masks"
                 % (len(mining_site_locations), len(mining_site_masks)))

    # Load Sina's mining site masks for 'mines_ipis250'. This collection lacks 'pcode'.
    if args.use_mines_ipis250:
        mining_site_masks = pd.concat(
            [mining_site_masks, load_mines_ipis250()], ignore_index=True)
        logging.info(
            "Loaded additional mining site masks, locations from 'mines_ipis250'.")

    # Filter mining site locations to places we have masks for.
    #
    # TODO(duckworthd): Should also grab locations that have been reviewed and
    # don't have visible mining sites. These locations won't have masks.
    mining_site_locations = mining_site_locations[
        mining_site_locations['pcode'].isin(mining_site_masks['pcode'])]

    # Depending on the dataset (All Visits vs. Last Visit), we may have multiple rows per location.
    if mining_site_locations["pcode"].value_counts().max() > 1:
        logging.info(
            "--mining_site_locations_shapefile contains multiple entries per location. Taking the last.")
        mining_site_locations = mining_site_locations.groupby("pcode").last()

    # Construct a binary ee.Image that's 1.0 wherever there's a mine.
    mining_site_masks_ee = create_earth_engine_mask(mining_site_masks)

    logging.info("Creating masks for %d mining sites." %
                 len(mining_site_locations))
    for idx, row in mining_site_locations.reset_index(drop=True).iterrows():
        logging.debug("Downloading mask for row %d/%d." %
                      (idx, len(mining_site_locations)))

        # If this image already exists, don't bother re-downloading it.
        if img_already_exists(args.output_image_root, row.vid) and not args.force_download:
            logging.debug("Mask already exists. Skipping.")
            continue

        # Construct circle around mining site's location.
        roi = create_earth_engine_roi(
            row.geometry, args.roi_bounding_box_width / 2)

        # Get coordinates for the circle.
        roi = roi.coordinates().getInfo()

        # Construct img. Ensure that its shape is as expected.
        img = mining.load_map_tile_containing_roi(
            mining_site_masks_ee, roi, scale=args.resolution_square_meters)
        img = img[0:roi_box_dims[0], 0:roi_box_dims[1]]
        img = np.reshape(img, [img.shape[0], img.shape[1], img.shape[2], 1])

        # Construct img metadata.
        img_metadata = create_img_metadata(img.shape, row.vid)

        # TODO(duckworthd): Should save then atomatically move to final destination
        # to ensure no race conditions.
        #
        # Save image.
        mining.save_image(args.output_image_root, img, img_metadata)

        # Load old metadata, add new row for this img, save.
        metadata = mining.load_metadata(args.output_image_root)
        metadata = mining.merge_metadata(
            metadata, pd.DataFrame.from_records([img_metadata]))
        mining.save_metadata(args.output_image_root, metadata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required arguments.
    parser.add_argument(
        '--mining_site_masks_shapefile',
        type=str, required=True,
        help='Path to shapefile containing mining site masks. Must contain "pcode" column that identifies which mine each mask belongs to.')
    parser.add_argument(
        '--mining_site_locations_shapefile',
        type=str, required=True,
        help='Path to shapefile containing mining site locations. Must contain "pcode" column that uniquely identifies each mine and "vid" that uniquely identifies each mine visit.')
    parser.add_argument(
        '--output_image_root',
        type=str, required=True,
        help='Path to directory to write masks under.')

    # Optional arguments.
    parser.add_argument(
        '--roi_bounding_box_width',
        type=int, default=3000,
        help='Width of square bounding box around mining site location to capture.')
    parser.add_argument(
        '--resolution_square_meters',
        type=int, default=30,
        help='Number of square meters per pixel in resulting mask.')
    parser.add_argument(
        '--loglevel',
        type=str, default='INFO',
        help='Default log level (DEBUG, INFO, WARNING, ERROR).')
    parser.add_argument(
        '--use_mines_ipis250',
        action="store_true",
        help='If True, Also include masks and ROIs from the "mines_ipis250" dataset.')
    parser.add_argument(
        '--force_download',
        action="store_true",
        help='If True, download mask again even if it already exists.')

    args = parser.parse_args()
    main(args)
