# Ryan Vilim's repository

## Description of Files

- __export_data.py__: Takes a set of data in the Storage API format, generates features for the model, removes clouds
  (according to the BQA cloud mask) and saves the resultant vectors in a .npz (gzipped numpy) array for input into the
  machine learning model. The specific command line options as well as descriptions are given in the --help output.

- __export_model.py__: Given a set of training data (.npz format) this will train and export a model. The specific command
 line options as well as descriptions are given in the --help output.

- __export_model_gridsearch.py__: Given a set of preprocessed CV splits, this script will run a gridsearch of parameters
  the specific parameters are hardcoded into the python file.

- __get_stats.py__: Given a test set and a model file, this script will ouptut a confusion matrix, a Precision-Recall curve
as well as a ROC curve

- __inference.py__: Given a dataset and a model file, this script will run predictions on the dataset, then add these
predictions to the dataset under the 'location_id'+_inference_timeagg and 'location_id'_inference. The _timeagg suffix
indicates that for a site with several observations, the most popular prediction has been taken. For example, if we
observed the same site 6 times, and the model predicted "mine" for 4 of those times, 'location_id'_inference_timeagg
would indicate that there was a mine here.

- __split_data.py__: Given a dataset this will split the dataset into a train, test and validation dataset with user defined
fractions. The split occurs along site boundaries, so different pixels from the same site will be put into the same split.
This is because pixels from the same site are likely correlated. If they are in different splits, the model can cheat
and use the fact that it has seen similar data before to make a prediction.

- __get_stats_inference.py__: Given a dataset contianing output from a trained model, this function contains some 
scripts to evaluate the precision and recall of predictions.

## Running
First we should make train/test/validation splits using split_data.py. After that, preprocess the data into 
.npz format with export_data.py, then train and export a model with export_model.py

