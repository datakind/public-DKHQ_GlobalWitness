# Features API

**Author**: duckworthd@

**Last Edit**: 2017 Oct 22

Prototype implementation of feature extraction API.

## Installation

1. Clone this repository and branch.

```shell
$ git clone git@github.com:datakind/ON-MiningDetection.git github
$ git branch -f features_api origin/features_api
$ git checkout features_api
```

1. Activate an Anaconda environment

```shell
$ conda create --name mining_detection
$ source activate mining_detection
```

1. Install dependencies with `conda`, `pip`

```shell
$ conda install gdal numpy pandas
$ pip install earthengine-api geopandas
$ earthengine authenticate
```

1. Install Storage API (see instructions in ../storage_api).

## Usage

```python
import features
import storage

# Get 100x100 (approximately) binary image mining mask image centered at
# (lat, lon) at a resolution of 30 meters^2 per pixel.
image, metadata = features.get_image_patch(
    {"source": "mask"},
    center=(lon, lat),
    patch_size=100,
    meters_per_pixel=30)
image     # np.array of shape [102, 101]
metadata  # {}

# Get Landsat 8 top-of-atmosphere images for all color bands, all dates
# available in 2015.
image, metadata = features.get_image_patch(
    {"source": "landsat8", "start_date": "2015-01-01", "end_date": "2016-01-01"},
    center=(lon, lat),
    patch_size=100,
    meters_per_pixel=30)
image     # np.array of shape [102, 101, 12, 2]
metadata  # {"bands": ["B1", "B2", ...], "dates": ["201501XX", ...]}

# Open an on-disk image dataset (may not exist yet).
dataset = storage.DiskDataset("/path/to/data_directory")

# Add a new image to the dataset. Images are indexed by location and image
# source (e.g. "landsat8"). Both are arbitrary strings.
dataset.add_image(location_id, image_source_id, image, metadata)

# pd.DataFrame listing all (location_id, image_source_id, metadata).
dataset.metadata

# np.array image.
dataset.load_image(location_id, image_source_id)
```
