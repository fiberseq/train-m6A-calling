#!/usr/bin/env python3
"""
m6a_supervised_cnn_ont.py
Author: Anupama Jha <anupamaj@uw.edu>
This module predicts whether an
adenine is methylated or not. The
model is trained with ONT read sequence, 
read quality score and methylation score
from oxford nanopore. The model is a
convolutional neural network.
"""

import torch
import argparse
import numpy as np
import configparser
import _pickle as pickle
from m6a_cnn_ont import M6ANet
from torchsummary import summary


def count_pos_neg(labels, set_name="", verbose=True):
    """
    Count the number of positive
    (m6A) and negative (everything
    else) examples in our set
    :param labels: np.array, one-hot-encoded
                             label,
    :param set_name: str, training,
                          validation or
                          test set.
    :return: #positives, #negatives
    """
    # First column are positive labels
    m6as = np.where(labels == 1)[0]
    # Second column are negative labels
    nulls = np.where(labels == 0)[0]
    num_pos = len(m6as)
    num_neg = len(nulls)
    if verbose:
        print(f"{set_name} has {num_pos}" f" positives and {num_neg} negatives")
    return num_pos, num_neg


def make_one_hot_encoded(y_array):
    """
    Convert int labels to one
    hot encoded labels
    :param y_array: np.array, int labels
    :return: one-hot-encoded labels
    """
    # Convert y_array to
    # a one-hot-encoded
    # vector
    y_array_ohe = np.zeros((len(y_array), 2))
    one_idx = np.where(y_array == 1)[0]
    y_array_ohe[one_idx, 0] = 1

    zero_idx = np.where(y_array == 0)[0]
    y_array_ohe[zero_idx, 1] = 1
    return y_array_ohe


class M6ADataGenerator(torch.utils.data.Dataset):
    """
    Data generator for the m6A
    model. It randomly selects
    batches from the data to
    train the m6A model.
    """

    def __init__(self, features, labels):
        """
        Constructor for the data
        generator class, expects
        features and labels matrices.
        :param features: numpy.array,
                            Nx15, N=number of
                            sequences, each
                            sequence is of
                            length 15.
        :param labels: numpy array,
                            one-hot-encoded
                            labels for whether
                            a sequence in
                            features variable
                            contains methylated
                            A or not.
        """
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        Get random indices from
        the training data
        to form batches.
        :param idx: numpy.array,
                    indices to retrieve.
        :return: x, y: features and
                       labels from
                       selected indices.
        """
        x = self.features[idx]
        y = self.labels[idx]

        x = torch.tensor(x)
        y = torch.tensor(y)

        return x, y


def m6AGenerator(
    train_path,
    val_path,
    input_size,
    pin_memory=True,
    num_workers=0,
    batch_size=32
):
    """
    This generator returns a training
    data generator as well as validation
    features and labels.
    :param train_path: str, path where
                            the training
                            data is stored.
    :param input_size: int, number of input
                            channels
    :param val_path: str, path where the
                          val data is stored.
    :param pin_memory: bool, memory efficiency.
    :param num_workers: int, number of threads
                             for data generator.
    :param batch_size: int, number of examples
                            in each batch
    :return: X_gen: training data generator,
             X_val: validation features,
             y_val: validation labels
    """
    # Load training data
    train_data = np.load(train_path, allow_pickle=True)
    print(list(train_data.keys()))

    # Load training and validation
    # features and labels. Sometimes
    # we want to train on input subsets,
    # this will achieve that.
    
    X_train = np.array(train_data["features"], dtype=float)
    X_train = X_train[:, 0:input_size, :]
    y_train = np.array(train_data["labels"], dtype=int)
    print(f"y_train: {y_train.shape}, {y_train}, {np.unique(y_train)}")
    
    count_pos_neg(y_train, set_name="train")

    # One-hot-encode train labels
    y_train_ohe = make_one_hot_encoded(y_train)
    
    

    # Load validation data
    val_data = np.load(val_path, allow_pickle=True)

    X_val = np.array(val_data["features"], dtype=float)
    X_val = X_val[:, 0:input_size, :]
    y_val = np.array(val_data["labels"], dtype=int)
    
    count_pos_neg(y_val, set_name="validation")

    # One-hot-encode val labels
    y_val_ohe = make_one_hot_encoded(y_val)
    
    

    print(
        f"Training features shape {X_train.shape},"
        f" training labels shape: {y_train.shape}"
    )
    print(
        f"Validation features shape {X_val.shape}, "
        f" validation labels shape: {y_val.shape}"
    )

    # Get the training data generator
    X_gen = M6ADataGenerator(X_train, y_train_ohe)

    # Wrap it in a data loader
    X_gen = torch.utils.data.DataLoader(
        X_gen,
        pin_memory=pin_memory,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
    )

    return X_gen, (X_val, y_val_ohe)


def run(config_file, train_chem):
    """
    Run data preprocess and model training.
    :param config_file: str, path to config
                            file.
    :param train_chem: str, which chemistry
                            to train.
    :return: None
    """
    # read parameters from config file
    config = configparser.ConfigParser()
    config.read(config_file)
    # get parameters for the
    # relevant chemistry
    rel_config = config[train_chem]
    # Number of input channels
    input_size = int(rel_config["input_size"])
    # length of input sequence
    input_length = int(rel_config["input_length"])
    # path to training data set
    train_data = rel_config["sup_train_data"]
    # path to validation data set
    val_data = rel_config["sup_val_data"]
    # cpu or cuda for training
    device = rel_config["device"]
    # path to save best model
    best_save_model = rel_config["best_supervised_model_name"]
    # path to save final model
    final_save_model = rel_config["final_supervised_model_name"]
    # maximum number of epochs for training
    max_epochs = int(rel_config["supervised_train_epochs"])
    # number of threads to process training data fetch
    num_workers = int(rel_config["sup_num_workers"])
    # batch size of training data
    batch_size = int(rel_config["sup_batch_size"])
    # learning rate
    sup_lr = float(rel_config["sup_lr"])

    # Move the model to appropriate
    # device
    model = M6ANet(input_size=input_size).to(device)

    # Adam optimizer with learning
    # rate 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=sup_lr)

    # Print model architecture summary
    summary_str = summary(model, input_size=(input_size, input_length))

    # Get training data generator
    # and validation data.
    X_train, (X_val, y_val) = m6AGenerator(
        train_data,
        val_data,
        input_size=input_size,
        pin_memory=True,
        num_workers=num_workers,
        batch_size=batch_size
    )

    validation_iter = int(np.floor(len(X_train) / (batch_size * 10.0)))

    # Train the model
    model.fit_supervised(
        X_train,
        optimizer,
        X_valid=X_val,
        y_valid=y_val,
        max_epochs=max_epochs,
        validation_iter=validation_iter,
        device=device,
        best_save_model=best_save_model,
        final_save_model=final_save_model,
        input_example=(1, input_size, input_length)
    )


def main():
    # if torch.cuda.is_available():
    #     print("GPU is available.")
    # else:
    #     print("GPU is not available.")
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file", type=str, default="config.yml", help="path to the config file."
    )

    parser.add_argument(
        "--train_chem",
        type=str,
        default="train_ONT_chemistry",
        help="Which chemistry to validate. The name should match the section header in the config file.",
    )

    args = parser.parse_args()

    print(f"Training a {args.train_chem} " f"supervised CNN model.")

    run(args.config_file, args.train_chem)


if __name__ == "__main__":
    main()
