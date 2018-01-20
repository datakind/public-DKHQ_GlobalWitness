# Team Repository for DataKind's Mining Detection in the DRC

Shared code for Mining Detection. Ask duckworthd@gmail.com for details.

## Structure

This repo contains one folder per user and subproject. See README.md files in
each directory for details and points of contact.

## Workflow

1. Install packages.

Follow instructions in `features_api`, `storage_api`, and `visualization_api`.
These libraries provide the core tools for downloading satellite imagery,
serializing it to disk, and visualizing it.

2. Download training data.

The following command downloads satellite imagery and rasterized "masks"
indicating where mines are located. Results are stored in `/tmp/ipis`.

```shell
$ python -m bin.download_ipis --base_dir="/tmp/ipis"
```

3. Process data.

This script ingests the training data, generates features for the model,
removes clouded datapoints, and saves the resultant vectors in a .npz (gzipped numpy) array.

```shell
$ python ./model/export_data.py --data_export_path /path/to/processed/data --data_input_path /path/to/downloaded/training/data
```

3. Train model.

The following command trains a random forest model and serializes it to disk. The model is trained to predict if a mine lies under a pixel or not.

```shell
$ python ./model/export_model.py --train_data_path /path/to/processed/data/ --test_data_path /path/to/processed/test/data --export_model_path /path/to/dir
```

4. Make predictions. Store to disk.

The following command uses a stored model to make predictions. Predictions are
stored to disk alongside satellite imagery.

```shell
python ./model/inference.py --data_path /path/to/data --model_path /path/to/model
```

5. Visualize predictions.

Visualizations are presented in Jupyter notebook. Open the notebook with the
following command,

```shell
$ jupyter notebook "visualization_api/notebooks/Visualization API Demo.ipynb"
```

Follow the `TODO` statements in the notebook to apply visualize predictions
made in the previous step.
