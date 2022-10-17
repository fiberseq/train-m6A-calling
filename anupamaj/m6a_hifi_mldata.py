"""
m6A_hifi_mldata.py
Author: Anupama Jha <anupamaj@uw.edu>
This module converts m6A HIFI data format to training features and labels for ML modelling tasks.
"""

import argparse
import numpy as np


def count_pos_neg(labels, set_name=""):
    """
    Count the number of positive (m6A) and negative (everything else) examples in our set
    :param labels: np.array, one-hot-encoded label
    :param set_name: str, training, validation or test set.
    :return:
    """
    # First column are positive labels
    m6as = np.where(labels[:, 0] == 1)[0]
    # Second column are negative labels
    nulls = np.where(labels[:, 1] == 1)[0]
    print(f"{set_name} has {len(m6as)} positives and {len(nulls)} negatives")


def get_hifi_data(hifi_data_file):
    """
    Get HiFi data formatted as two arrays (features and labels)
    in a dictionary.
    :param hifi_data_file: str, path to the hifi_data_file
    :return: (hifi_feats:np.array, m6a_labels:np.array)
    """
    load_file = np.load(hifi_data_file, allow_pickle=True)
    hifi_feats = np.array(load_file['features'], dtype=float)
    m6a_labels = load_file['labels']

    print(f"hifi_feats shape: {hifi_feats.shape}")
    print(f"m6a_labels shape: {m6a_labels.shape}")

    # Normalize the IP and PW data by dividing by 255.0
    # IP is index 4 and PW is index 5.
    #hifi_feats[:, 4:6, :] = hifi_feats[:, 4:6, :] / 255.0

    return hifi_feats, m6a_labels


def save_train_val_test_data(hifi_data_file, train_save_path, test_save_path):
    """
    Given the path to the hifi_data_file, save two files, one with train and
    validation split and another with the test split.
    :param hifi_data_file: str, Path to the input HiFi data file, this file should exist
    :param train_save_path: str, Path to the output object where the train and validation
                                data will be saved.
    :param test_save_path: str, Path to the output object where the test data will be saved.
    :return: None.
    """
    # Get Hi-Fi features and labels
    hifi_feats, m6a_labels = get_hifi_data(hifi_data_file)

    # One-hot-encode labels
    labels_ohe = np.zeros((len(m6a_labels), 2))
    labels_ohe[np.where(m6a_labels == 1)[0], 0] = 1
    labels_ohe[np.where(m6a_labels == 0)[0], 1] = 1

    # Since the train, validation and test data are roughly
    # ordered by CCS Read ID, this approach should give
    # us blocked design where we have features from
    #  mostly unique CCS IDs.
    train_idx = np.arange(0, 0.6 * len(hifi_feats), 1, dtype=int)
    val_idx = np.arange(0.6 * len(hifi_feats), 0.8 * len(hifi_feats), 1, dtype=int)
    test_idx = np.arange(0.8 * len(hifi_feats), len(hifi_feats), 1, dtype=int)

    # Get the train, validation and test data
    X_train = hifi_feats[train_idx, :, :]
    y_train = labels_ohe[train_idx, :]

    X_val = hifi_feats[val_idx, :, :]
    y_val = labels_ohe[val_idx, :]

    X_test = hifi_feats[test_idx, :, :]
    y_test = labels_ohe[test_idx, :]

    # See how many m6A versus not are present in
    # each set, each data should have roughly the
    # same ratio.
    count_pos_neg(np.array(y_train), set_name="Train")
    count_pos_neg(np.array(y_val), set_name="Validation")
    count_pos_neg(np.array(y_test), set_name="Test")

    # Save the training and validation data
    save_data_dict = dict()
    save_data_dict['X_train'] = X_train
    save_data_dict['y_train'] = y_train
    save_data_dict['X_val'] = X_val
    save_data_dict['y_val'] = y_val

    np.savez_compressed(train_save_path, save_data_dict=save_data_dict, compress=True)

    print(f"Saved training and validation data at: {train_save_path}")

    # save test data
    save_data_dict = dict()
    save_data_dict['X_test'] = X_test
    save_data_dict['y_test'] = y_test

    np.savez_compressed(test_save_path, save_data_dict=save_data_dict, compress=True)

    print(f"Saved test data at: {test_save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--hifi_data_file',
        type=str,
        default="../data/hifi.half.other.test.npz",
        help="path to the hifi data npz file."
    )

    parser.add_argument(
        '--train_save_path',
        type=str,
        default='../data/m6A_train_other_half_hifi',
        help="Path to where we want to store training and validation data."
    )

    parser.add_argument(
        '--test_save_path',
        type=str,
        default="../data/m6A_test_other_half_hifi",
        help="Path to where we want to store test data. "
    )

    args = parser.parse_args()

    # Main function to partition and save train, validation and test data.
    save_train_val_test_data(args.hifi_data_file, args.train_save_path, args.test_save_path)
