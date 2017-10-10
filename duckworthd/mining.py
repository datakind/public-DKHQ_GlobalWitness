"""Utilities for working with satellite imagery for mining."""
import json
import os
import re
import string
import tempfile
import urllib
import zipfile

from matplotlib import pyplot as plt
import bcolz
import ee as earth_engine; earth_engine.Initialize()
import gdal
import h5py
import numpy as np
import pandas as pd
import geopandas as gpd


# Default directory containing images.
DEFAULT_IMAGE_ROOT = '/workspace/lab/data_s3/mines_ipis'

# Fusion Table ID containing polygons around mining sites.
DEFAULT_IPIS_MINING_POLYGONS = 'ft:1HG3R3cebqMp2yK0cOimTL7wLnh41c1DH24GyWQg1'

# Images with 4 axes. The first two are typical for images -- x, y. The
# third is color band (traditionally RGB, but Landsat captures more). The
# fourth is time, representing when the image was captured.
X_AXIS = 0
Y_AXIS = 1
BAND_AXIS = 2
TIME_AXIS = 3


def load_ipis_mining_sites_dataset():
    """Load all mining sites annotated by IPIS from FusionTable as GeoJSON."""
    return earth_engine.FeatureCollection('ft:1P1f-A2Sl44YJEqtD1FvA1z7QtDFsRut1QziMD-nV').getInfo()


def _get_metadata_file_path(image_root):
    """Get absolute path to metadata.json file in a given directory.

    If there are more than one metadata.json files, pick the last one after
    sorting.
    """
    if not os.path.exists(image_root):
      raise ValueError(u'%s does not exist. No metadata files found.' % image_root)

    filenames = os.listdir(image_root)
    metadata_filenames = [name for name in filenames if 'metadata' in name]
    if not metadata_filenames:
      raise ValueError(u'No files with "metadata" in name found under %s' % image_root)
    metadata_filename = list(sorted(metadata_filenames))[-1]
    return os.path.join(image_root, metadata_filename)


def load_metadata(image_root=None):
    """Load JSON file storing image metadata from disk.

    If no JSON file can be found, an empty DataFrame is returned.
    """
    image_root = image_root or DEFAULT_IMAGE_ROOT

    try:
      fpath = _get_metadata_file_path(image_root)
    except ValueError:
      return pd.DataFrame(
          columns=["bands", "collection", "dates", "dim", "fpath", "id"])

    with open(fpath) as f:
        return pd.DataFrame(json.load(f))


def save_metadata(image_root, metadata):
    """Store DataFrame containing image metadata to disk."""
    if not os.path.exists(image_root):
        os.makedirs(image_root)
    with open(os.path.join(image_root, "metadata4.json"), "w") as f:
        return metadata.to_json(f)


def merge_metadata(old_metadata, new_metadata):
    """Merge two metadata DataFrames."""
    # Remove all rows from 'old_metadata' that have the same path as in 'new_metadata'
    old_metadata = old_metadata[~old_metadata['fpath'].isin(new_metadata['fpath'])]

    # Concatenate new and old together.
    return pd.concat([old_metadata, new_metadata], ignore_index=True)


def load_image(img_metadata, image_root=None):
    """Load a single image from disk."""
    image_root = image_root or DEFAULT_IMAGE_ROOT
    fname = os.path.join(image_root, img_metadata['fpath'])
    return bcolz.open(fname)[:]


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
        A matching type that Earth Engine understands (e.g. ee.FeatureCollection, ee.Geometry.Point).
    """
    if isinstance(geojson, dict):
        if 'type' not in geojson:
            raise ValueError("Not 'type' attribute in geojson: %s" % (geojson,))
        if geojson['type'] == 'FeatureCollection':
            return earth_engine.FeatureCollection(
                geojson_to_earthengine(geojson['features']))
        elif geojson['type'] == 'Feature':
            return earth_engine.Feature(
                geojson_to_earthengine(geojson['geometry']),
                geojson['properties'])
        elif geojson['type'] == 'Point':
            return earth_engine.Geometry.Point(coords=geojson['coordinates'])
        elif geojson['type'] == 'Polygon':
            return earth_engine.Geometry.Polygon(
                coords=geojson['coordinates'],
                geodesic=geojson.get('geodesic', None))
        raise ValueError("Unsupported GeoJSON dict type: %s" % geojson['type'])
    elif isinstance(geojson, list):
        return [geojson_to_earthengine(element) for element in geojson]
    elif isinstance(geojson, tuple):
        return tuple(geojson_to_earthengine(element) for element in geojson)
    elif type(geojson) in [int, float, str, unicode]:
        return geojson
    else:
        raise ValueError("Unable to parse type: %s" % type(geojson))


def to_earthengine_featurecollection(obj):
  """Converts an object to an ee.FeatureCollection.

  'obj' can be one of:
  - str: a Fusion Table ID ("ft:xxx")
  - GeoDataFrame
  - GeoJSON dict of type 'FeatureCollection'
  """
  # If string, load FeatureCollection using Earth Engine.
  if isinstance(obj, basestring):
    return earth_engine.FeatureCollection(obj)

  # If GeoDataFrame, convert to ee.FeatureCollection.
  if isinstance(obj, gpd.GeoDataFrame):
    return geodataframe_to_earthengine(obj)

  # If GeoJSON, convert to ee.FeatureCollection.
  if isinstance(obj, dict):
    assert 'type' in obj
    assert obj['type'] == 'FeatureCollection'
    return geojson_to_earthengine(obj)


def load_image_mask(img_metadata, ipis_mining_sites=None, ipis_mining_polygons=None, image_root=None):
    """Load binary mask labeling pixels as "mining" or "not mining".

    Args:
      img_metadata: pd.Series from a metadata.json file.
      ipis_mining_sites: FeatureCollection GeoJSON dict containing all IPIS
        mining site locations as Points.
      ipis_mining_polygons: Object that can be converted to an
        ee.FeatureCollection. See to_earthengine_featurecollection() for
        available options. Default's to Sina's Fusion Table.
      image_root: string. unused?

    Returns:
      numpy array of shape [100, 100] with values {0, 1}, where 0.0 == no mine
      and 1.0 == mine, centered at the location described by img_metadata.
    """
    # If None, use the Fusion Table containing mining sites that Sina created.
    if ipis_mining_sites is None:
      ipis_mining_polygons = DEFAULT_IPIS_MINING_POLYGONS

    ipis_mining_polygons = to_earthengine_featurecollection(ipis_mining_sites)

    ipis_mining_image = ipis_mining_polygons.reduceToImage(
        properties=['mine'],
        reducer=earth_engine.Reducer.first()) # earth_engine.Image() type

    # Get Point corresponding to this image from IPIS dataset.
    roi_id = img_metadata['id']
    if ipis_mining_sites is None:
        ipis_mining_sites = load_ipis_mining_sites_dataset()
    roi = ipis_mining_sites['features'][roi_id]['geometry']
    assert roi['type'] == 'Point'


    # Create a circle around the point with a given buffer size (in meters).
    buff = 1500  # radius of 1500 meters about the point.
    roi_point = earth_engine.Geometry.Point(roi['coordinates'])
    roi_buff = earth_engine.Geometry.buffer(roi_point, buff) # ee.Geometry()
    roi_buff = roi_buff.getInfo() # GeoJSON dict

    # Download image containing circle from Earth Engine.
    scale = 30   # 30 meters/pixel --> circle with 100 pixel diameter.
    mask = load_map_tile_containing_roi(ipis_mining_image, roi_buff['coordinates'], scale=scale)

    # Some images are 101 x 101, some are 100 x 100. Let's ensure they're all
    # 100 x 100.
    mask = mask[:100, :100]
    assert mask.shape[2] == 1, 'Mask has > 1 band.'

    return mask.reshape(mask.shape[0], mask.shape[1])


def load_map_tile_containing_roi(image, roi, scale=30):
    """Get rasterized image containing ROI from Earth Engine.

    Constructs a rasterized image tile subsetting 'image'. The image is large
    enough to fully contain the polygon described by 'roi', and will contain
    one pixel per 'scale' m^2 area.

    Args:
      image: ee.Image instance. To be used as mask. Must have exactly 1 band.
      roi: Triple-nested list of floats, where lowest level is [longitude,
        latitude] pairs from 'coordinates' of a GeoJSON polygon.
      scale: int. Number of squared meters per pixel.

    Returns:
      numpy array of shape [N x M x K], where N is width, M is height, and K is
      number of bands.
    """
    # Generate a random filename.
    filename = ''.join(np.random.choice(list(string.ascii_letters), size=10))

    # Download image containing ROI.
    url = earth_engine.data.makeDownloadUrl(
        earth_engine.data.getDownloadId({
          'image': image.serialize(),
          'scale': '%d' % scale,
          'filePerBand': 'false',
          'name': filename,
          'region': roi
        }))
    local_zip, headers = urllib.urlretrieve(url)
    with zipfile.ZipFile(local_zip) as local_zipfile:
        local_tif_filename = local_zipfile.extract(filename + '.tif', tempfile.mkdtemp())

    # Read image into memory. Result has shape [x, y, color bands].
    dataset = gdal.Open(local_tif_filename, gdal.GA_ReadOnly)
    bands = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
    return np.stack(bands, axis=2)


def save_images(image_root, images, metadata):
    """Store a list of images to disk."""
    assert len(images) == len(metadata)
    if not os.path.exists(image_root):
        os.makedirs(image_root)
    for (img, (_, img_metadata)) in zip(images, metadata.iterrows()):
        save_image(image_root, img, img_metadata)


def save_image(image_root, img, img_metadata):
    """Store a single image to disk."""
    if not os.path.exists(image_root):
        os.makedirs(image_root)
    fname = os.path.join(image_root, img_metadata['fpath'])
    dname = os.path.dirname(fname)
    if not os.path.exists(dname):
        os.makedirs(dname)
    c = bcolz.carray(img, rootdir=fname, mode ='w')
    c.flush()


def save_images_with_hdf5(image_root, images, metadata):
    assert len(images) > 0, "Must have 1+ images to write."

    # Make directory if necessary.
    if not os.path.exists(image_root):
        os.makedirs(image_root)

    # Construct an empty HDF5 dataset on disk.
    image_shape = images[0].shape
    initial_images_shape = (len(images),) + image_shape
    max_images_shape = (None,) + image_shape
    with h5py.File(os.path.join(image_root, "images.h5"), "w") as h5f:
        dataset = h5f.create_dataset("images" , initial_images_shape, maxshape=max_images_shape)

        # Write images into space.
        for i, image in enumerate(images):
            dataset[i] = image


def save_images_with_bcolz(image_root, imgs, metadata):
    assert len(imgs) == len(metadata)

    # Make directory if necessary.
    if not os.path.exists(image_root):
        os.makedirs(image_root)

    # Construct a bcolz array with the first image only.
    assert len(imgs) > 0, "Must have 1+ images to write."
    output_shape = (1, ) + imgs[0].shape
    with bcolz.carray(imgs[0].reshape(output_shape), rootdir=os.path.join(image_root, "images"), mode="w") as array:

        # Add all other images.
        for i, img in enumerate(imgs):
            if i == 0:
                continue
            array.append(img.reshape(output_shape))


def load_images_with_hdf5(image_root):
    """Load all images from HDF5 array."""
    with h5py.File(os.path.join(image_root, "images.h5")) as h5f:
        return h5f['images'][:]


def load_image_with_hdf5(image_root, img_metadata):
    """Load all images from HDF5 array."""
    with h5py.File(os.path.join(image_root, "images.h5")) as h5f:
        return h5f['images'][int(img_metadata.name)]


def load_images_with_bcolz(image_root):
    """Load all images from bcolz array."""
    with bcolz.open(os.path.join(image_root, "images")) as array:
        return array[:]


def load_image_with_bcolz(image_root, img_metadata):
    """Load a single image from bcolz array."""
    with bcolz.open(os.path.join(image_root, "images")) as array:
        return array[int(img_metadata.name)]


def plot_image(image, metadata=None, band=None, ax=None, cmap='gray'):
    ax = ax or plt.gca()

    # Aggregate over time.
    if len(image.shape) == 4:
    	image = np.nanmedian(image, axis=TIME_AXIS)

    # Select only the bands requested.
    if len(image.shape) == 3:
        assert band is not None, "You must choose a band to plot."
        assert metadata is not None, "metadata required to select color band."
        band_index = metadata['bands'].index(band)
        image = image[:, :, band_index]

    ax.imshow(image, cmap=cmap)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    return ax


def canonicalize_image(img, img_metadata):
    """Canonicalize image for machine learning models.

    - Aggregates across 2016/06 to 2017/06
    - Drops all bands but B1...B11.
    """
    img_metadata = img_metadata.copy()

    # Get all dates in a 12 month span.
    dates = [date for date in img_metadata['dates']
             if date >= '20160601' and date < '20170601']

    if len(dates) < 12:
        raise ValueError(
	     "Found %d dates for the following image when 12 were expected. %s"
             % (len(dates), img_metadata))

    img_metadata['dates'] = dates

    # Aggregate across 12 month span (hides cloud cover). Only keep the start
    # date in the metadata, as there's exactly one date dimension.
    img = np.nanmedian(img, axis=TIME_AXIS, keepdims=True)
    img_metadata['dates'] = [dates[0]]

    # Only keep raw bands. All others bands are simple functions of these.
    bands = [band for band in img_metadata['bands']
             if re.search('^B\d+$', band) is not None]
    band_indices = [img_metadata['bands'].index(band) for band in bands]

    img = img[:,:, band_indices]
    img_metadata['bands'] = bands

    img_metadata["dim"] = img.shape

    return img, img_metadata


def canonicalize_image_by_month(img, img_metadata, band=None):
    """Canonicalize an image by taking its median pixel value per month.

    Args:
        img: numpy array, shape [height, width, num color bands, num dates].
        img_metadata: pandas Series. Contains 'bands' and 'dates' entries.
	band: None, string, or list of strings. If None, output all color
	    bands. If string, output a single color band, if list of strings,
            output one color band per string.
    """
    assert len(img.shape) == 4, "img must be [width, height color band, time]."

    # Select bands to process.
    if band is None:
        bands = img_metadata["bands"]
    if isinstance(band, basestring):
        bands = [band]
    elif isinstance(band, list):
        bands = band
    else:
        raise ValueError("Unrecognized type for argument 'band': %s" % band)

    band_idxs = [img_metadata["bands"].index(b) for b in bands]
    img_band = img[:, :, band_idxs, :]


    # Extract month out of each date (YYYYMMDD string)
    dates = pd.DataFrame({"dates": img_metadata['dates']})
    dates["month"] = dates["dates"].str.slice(4, 6)

    # Construct result image. There will be 12 months.
    width, height, _, _ = img.shape
    result_img = np.full((width, height, len(bands), 12), np.nan)
    for month, group in dates.groupby("month"):
        # Select the appropriate time, color bands.
        time_idxs = list(group.index)
        img_month = img_band[:, :, :, time_idxs]

	# Take median pixel intensity over time.
        result_img[:, :, :, int(month)-1] = np.nanmedian(
            img_month, axis=[TIME_AXIS])

    # Construct new metadata. We'll use the first date for each month in the
    # grouping.
    result_metadata = img_metadata.copy()
    result_metadata["dim"] = result_img.shape
    result_metadata["bands"] = bands
    result_metadata["dates"] = list(dates.groupby("month").first()["dates"])

    return result_img, result_metadata


def merge_canonical_image_and_mask(canonical_img, mask, img_metadata):
    """Combine canonical_image and mask into a single array."""
    # Ensure canonical_img and mask have the same shape.
    assert len(canonical_img.shape) == 4
    mask = np.reshape(mask, [mask.shape[0], mask.shape[1], 1, 1])

    # Copy time dim as many times as necessary to match 'canonical_img'.
    mask = np.tile(mask, [1, 1, 1, canonical_img.shape[3]])

    # Concatenate mask as the final band.
    canonical_img = np.concatenate([canonical_img, mask], axis=BAND_AXIS)

    # Add 'mask' as the final band to the metadata.
    img_metadata = img_metadata.copy()
    img_metadata['bands'] = img_metadata['bands'] + ['mask']

    return canonical_img, img_metadata


def plot_monthly_image(img, img_metadata):
    assert len(img.shape) == 4, "img shape must be [height, width, color band, month]"
    assert img.shape[3] == 12, "img must have 1 entry per month for every color band."

    months = ["Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"]
    num_cols = len(img_metadata["bands"])
    num_rows = len(months)
    plt.figure(figsize=(2 * num_cols, 2 * num_rows))
    for i in range(num_rows):
        for j in range(num_cols):
            ax = plt.subplot(num_rows, num_cols, i * num_cols + j + 1)
            ax.set_title("%s/%s" % (months[i], img_metadata["bands"][j]))
            plot_image(img[:,:,j,i])
