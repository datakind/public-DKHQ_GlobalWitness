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

[example_notebook]: notebooks/Visualization API Demo.ipynb
