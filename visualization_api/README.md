# Visualization API

**Author**: duckworthd@

**Last Edit**: 2017 Nov 18

API for visualizing satellite imagery.

## Installation

1. Clone this repository and branch.

```shell
$ git clone git@github.com:datakind/ON-MiningDetection.git github
$ cd github/visualization_api
```

1. Activate an Anaconda environment

```shell
$ conda create --name mining_detection
$ source activate mining_detection
```

1. Install dependencies with `conda`,

```shell
$ conda install bcolz folium jupyter matplotlib pandas -c conda-forge
```

1. Install `storage`

```shell
$ cd ../storage_api
$ python setup.py install
```

1. Install library in environment.

```shell
$ python setup.py install
```

## Usage

See the [example notebook][example_notebook] for a live example.



```python
import storage
import visualization

# Open an on-disk image dataset (may not exist yet).
dataset = storage.DiskDataset("/path/to/data_directory")

# Load a single image into memory.
location_id = ...
image = dataset.load_image(location_id, "landsat8")
meta = dataset.image_metadata(location_id, "landsat8")["metadata"]

# Construct bounding box for this image.
bounding_box = visualization.BoundingBox.from_metadata(meta)

# Show a single RGB image.
plt.figure()
visualization.show_image(image[:, :, [6, 5, 4], 0], bounding_box, title="...")

# Show each of an image's color bands in a 4-column array.
bands = ["B1", "B2", ...]
visualization.show_color_bands(image[..., 0], ncols=4, bands=bands)

# Show Landsat's special color combinations.
visualization.show_landsat8_color_combinations(image[..., date], ncols=4)

# Show image on top of a map.
folium_map = visualization.create_folium_map(bounding_box, tiles='ArcGIS')
visualization.overlay_image_on_map(
    folium_map,
    image[:, :, [5, 4, 3], date],
    bounding_box,
    opacity=0.75)
folium_map
```

[example_notebook]: notebooks/Visualization API Demo.ipynb
