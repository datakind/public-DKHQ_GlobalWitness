# Dataset API

**Author**: duckworthd@

**Last Edit**: 2017 Oct 29

API for storing and loading images from disk.

## Installation

1. Clone this repository and branch.

```shell
$ git clone git@github.com:datakind/ON-MiningDetection.git github
$ cd github/storage_api
```

1. Activate an Anaconda environment

```shell
$ conda create --name mining_detection
$ source activate mining_detection
```

1. Install dependencies with `conda`,

```shell
$ conda install bcolz pandas
```

## Usage

```python
import storage

# Open an on-disk image dataset (may not exist yet).
dataset = storage.DiskDataset("/path/to/data_directory")

# Add a new image to the dataset. Images are indexed by location and image
# source (e.g. "landsat8"). Both are arbitrary strings.
for i in range(5):
    location_id = "mine_site_{}".format(i)
    source_id = "landsat8_32day"
    image = np.random.rand(100, 100, 3, 2)
    metadata = {"bands": ["B1", "B2", "B3"], "dates": ["20150101", "20150202"]}
    dataset.add_image(location_id, source_id, image, metadata)

# pd.DataFrame listing all (location_id, image_source_id, metadata).
dataset.metadata()

# Loads 'image' from disk.
dataset.load_image("mine_site_2", "landsat8_32day")

# Load all images with the same source.
for image, image_metadata in dataset.load_images("landsat8_32day"):
    image                          # np.array
    image_metadata["location_id"]  # "mine_site_0"
    image_metadata["source_id"]    # "landsat8_32day"
    image_metadata["metadata"]     # {"bands": ..., "dates": ...}
```
