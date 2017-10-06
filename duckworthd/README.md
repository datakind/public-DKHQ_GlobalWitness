# Daniel Duckworth's repository

Contact: duckworthd@gmail.com

## Usage

1. Download satellite images (> 20GB).

```shell
$ wget https://s3-us-west-2.amazonaws.com/rvilim.mining/mines_ipis250.tar.gz
$ mkdir /tmp/mines_ipis250
$ tar -vzxf mines_ipis250.tar.gz -C /tmp/mines_ipis250
```

1. Install Anaconda. See [website](https://conda.io/docs/user-guide/install/linux.html#install-linux-silent) for instructions.

1. Create a virtual environment

```shell
$ conda create --name datakind
$ source activate datakind
```

1. Install libraries

```shell
$ pip install -r requirements.txt
```

1. Use library

```python
from duckworthd import mining

IMAGE_ROOT = "/tmp/mines_ipis250"

# Load dataset's metadata into memory.
all_metadata = mining.load_metadata(IMAGE_ROOT)

# Select and load a single image.
img_metadata = all_metadata.iloc[0]
image = mining.load_image(img_metadata, IMAGE_ROOT)

# Plot image.
mining.plot_image(image, img_metadata, band="NDVI")
```
