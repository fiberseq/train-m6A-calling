# Reformats data from CNN matrix to single vector for XGBoost

import numpy as np

data_path = "../data/m6A_train_more_large.npz"
train_val_data = np.load(data_path, allow_pickle=True)

# Get the dictionary from the containing relevant data
train_val_data = train_val_data['save_data_dict'][()]

# Load training and validation features and labels
X_train = train_val_data['X_train']
y_train = train_val_data['y_train']
X_val = train_val_data['X_val']
y_val = train_val_data['y_val']

# Reshape features to one dimension
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1] * X_val.shape[2])

# Add labels before feature list
y_train = y_train[:, 0]
y_val = y_val[:, 0]
final_train_data = np.concatenate([y_train[:, np.newaxis], X_train], axis=1)
final_val_data= np.concatenate([y_val[:, np.newaxis], X_val], axis=1)

# Save data as CSV
np.savetxt("m6a_train_more_large.csv", final_train_data, delimiter=",")
np.savetxt("m6a_val_more_large.csv", final_val_data, delimiter=",")
