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

This script ingests a folder of satellite images and masks which are stored in dataset api
then features for the model,
removes clouded datapoints, and saves the resultant vectors in a .npz (gzipped numpy) array.

```shell
$ python ./model/export_data.py --data_export_path /path/to/processed/data --data_input_path /path/to/downloaded/training/data
```

4. Train model.

Using the .npz files generated in the previous step, export_model.py trains a random forest model and serializes it to disk. 
It takes data arguments, one for the actual training data, and a second which points to a test dataset (.npz format) to 
report model performance. The model is trained to predict if a mine lies under a pixel or not.

```shell
$ python ./model/export_model.py --train_data_path /path/to/processed/data.npz --test_data_path /path/to/processed/test/data.npz --export_model_path /path/to/dir
```

5. Make predictions. Store to disk.

The following command uses a stored model to make predictions. 

```shell
python ./model/inference.py --data_path /path/to/data --model_path /path/to/model
```

The --data_path option is a dataset in the Dataset API format. This command makes predictions on each image and saves 
the results to the same dataset. This adds two new source ids to the dataset
'landsat_inference_timeagg' and 'landsat_inference'. 'landsat_inference_timeagg' aggregates the predictions for the same 
pixel location in time by taking the majority prediction across time. 'landsat_inference' stores each prediction in time as an extra
dimension in the output array. This means that the image in the landsat_inference_timeagg source has shape \[n_x, n_y\]
while the image in the landsat_inference has shape \[n_time, n_x, n_y\].

6. Visualize predictions.

Visualizations are presented in Jupyter notebook. Open the notebook with the
following command,

```shell
$ jupyter notebook "visualization_api/notebooks/Visualization API Demo.ipynb"
```

Follow the `TODO` statements in the notebook to apply visualize predictions
made in the previous step. By default, predictions generated via 
 model/inference.py are stored in the 'landsat8_inference_timeagg' source id
 of the dataset.