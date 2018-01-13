# Visualization API

**Author**: duckworthd@

**Last Edit**: 2018 Jan 13

API for visualizing satellite imagery.

## Installation

1. Clone this repository and branch.

```shell
$ git clone git@github.com:datakind/ON-MiningDetection.git github
$ cd github/visualization_api
```

2. Activate an Anaconda environment

If you do not yet have Anaconda installed, follow the instructions for your
platform [here](https://conda.io/docs/user-guide/install/linux.html).

```shell
$ conda create --name mining_detection python=2.7
$ conda activate mining_detection
```

3. Install dependencies with `conda`,

```shell
$ conda install bcolz folium jupyter matplotlib pandas -c conda-forge
```

4. Install Storage API (see instructions in `../storage_api`).

5. Install library in environment.

```shell
$ python setup.py test
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
