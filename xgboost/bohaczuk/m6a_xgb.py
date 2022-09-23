# Run XGBoost from CSV file. Each training/validation example should be
# a single line containing the label as the first value. The model is
# saved as .json


# Modified from https://xgboost.readthedocs.io/en/stable/get_started.html

import xgboost as xgb
import numpy as np


# Read in training and validation data
dtrain = xgb.DMatrix('m6a_train_otherhalf_hifi2.csv?format=csv&label_column=0')
dval = xgb.DMatrix('m6a_val_otherhalf_hifi2.csv?format=csv&label_column=0')

# Specify parameters via map
param = {'max_depth':6, 'objective':'binary:logistic', 'eval_metric':'auc'}
num_round = 2
eval_s=[(dtrain, "train"), (dval, "validation")]

# Train model
bst = xgb.train(param, dtrain, num_round, evals=eval_s)

# Save model
bst.save_model('xgboost.tstrun.json')

# Make prediction
# preds = bst.predict(dtest)
