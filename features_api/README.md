# Features API

Author: duckworthd@
Last Edit: 2017 Oct 22

Prototype implementation of feature extraction API.

## Usage

This library provides `get_image_patch()`, a function that takes in a "feature
spec", a location, an image size, and a resolution. The service calls out to
whatever APIs are necessary and returns an `image` and `metadata` matching the
feature spec.

A "feature spec" is a dict with options for the underlying API, such as
`start_date` for Landsat 8 imagery. Depending on the API, different options are
available. The only required option is `source`, which specifies which
underlying API to use.

The return value includes a numpy array and a dict of metadata.
- The array's shape is [lon, lat, ...]. The first 2 dims are spatial.  The
  array may contains additional dimensions, such as color bands or time. See
  individual API implementations' docs for details.
- The dict contains additional information about the returned image, such as
  the names of the color bands or the dates photos were taken. See individual
  API implementations' docs for details.

```python
from features import get_image_patch

# Get 100x100 (approximately) binary image mining mask image centered at (lat,
# lon) at a resolution of 30 meters^2 per pixel.
image, metadata = get_image_patch(
    {"source": "mask"}, center=(lat, lon), patch_size=100, meters_per_pixel=30)
image     # np.array of shape [102, 101]
metadata  # {}

# Get Landsat 8 top-of-atmosphere images for all color bands, all dates available in 2015.
image, metadata = get_image_patch(
    {"source": "landsat8", "start_date": "2015-01-01", "end_date": "2016-01-01"},
    center=(lat, lon), patch_size=100, meters_per_pixel=30)
image     # np.array of shape [102, 101, 12, 2]
metadata  # {"bands": ["B1", "B2", ...], "dates": ["201501XX", ...]}
```
