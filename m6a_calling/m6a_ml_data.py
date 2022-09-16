"""
m6A_ml_data.py
Author: Anupama Jha <anupamaj@uw.edu>
one-hot-encoding code adapted from bpnet-lite (https://github.com/jmschrei/bpnet-lite)
This module converts m6A SMRTMatrix format to training features and labels for ML modelling tasks.
"""

import pickle
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


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


def one_hot_encode(sequence, ignore='N', alphabet=None, dtype='int8',
                   desc=None, verbose=False, **kwargs):
    """Converts a string or list of characters into a one-hot encoding.

    This function will take in either a string or a list and convert it into a
    one-hot encoding. If the input is a string, each character is assumed to be
    a different symbol, e.g. 'ACGT' is assumed to be a sequence of four
    characters. If the input is a list, the elements can be any size.

    Although this function will be used here primarily to convert nucleotide
    sequences into one-hot encoding with an alphabet of size 4, in principle
    this function can be used for any types of sequences.

    Parameters
    ----------
    sequence : str or list
        The sequence to convert to a one-hot encoding.

    ignore : str, optional
        A character to indicate setting nothing to 1 for that row, keeping the
        encoding entirely 0's for that row. In the context of genomics, this is
        the N character. Default is 'N'.

    alphabet : set or tuple or list, optional
        A pre-defined alphabet. If None is passed in, the alphabet will be
        determined from the sequence, but this may be time consuming for
        large sequences. Default is None.

    dtype : str or np.dtype, optional
        The data type of the returned encoding. Default is int8.

    desc : str or None, optional
        The title to display in the progress bar.

    verbose : bool or str, optional
        Whether to display a progress bar. If a string is passed in, use as the
        name of the progressbar. Default is False.

    kwargs : arguments
        Arguments to be passed into tqdm. Default is None.

    Returns
    -------
    ohe : np.ndarray
        A binary matrix of shape (alphabet_size, sequence_length) where
        alphabet_size is the number of unique elements in the sequence and
        sequence_length is the length of the input sequence.
    """

    d = verbose is False

    # If sequence is string, convert it to a list
    if isinstance(sequence, str):
        sequence = list(sequence)

    # get the alphabet
    alphabet = alphabet or np.unique(sequence)
    # every character in the alphabet except N
    alphabet = [char for char in alphabet if char != ignore]
    alphabet_lookup = {char: i for i, char in enumerate(alphabet)}
    # Get one hot encoding for our sequence
    ohe = np.zeros((len(sequence), len(alphabet)), dtype=dtype)
    for i, char in tqdm(enumerate(sequence), disable=d, desc=desc, **kwargs):
        if char != ignore and char in alphabet:
            idx = alphabet_lookup[char]
            ohe[i, idx] = 1
    return ohe


def get_pwm(seq_array, alphabet=['A', 'C', 'G', 'T']):
    """
    For all reads in one example, convert each to one-hot-encoded matrix and then get
    a position weight matrix by normalizing frequencies observed at each sequence position.
    :param seq_array: numpy array, array of sequences
    :param alphabet: list, sequence alphabet to one-hot-encode over.
    :return:
    """
    # Get a matrix of size of (sequence, alphabet)
    total_ohe = np.zeros((seq_array.shape[1], len(alphabet)))
    # For every read
    for i in range(seq_array.shape[0]):
        sequence = ""
        for j in range(seq_array.shape[1]):
            sequence += str(seq_array[i, j])
        # One-hot-encode the read
        ohe_seq = one_hot_encode(sequence, alphabet=alphabet)
        # Add it to the total count
        total_ohe += ohe_seq

    # Add small pseudo count to the data
    total_ohe += 1e-5
    # Normalize by dividing by total number of reads observed at each position
    total_ohe = total_ohe / (np.sum(total_ohe, axis=1)[:, np.newaxis])

    return total_ohe


def get_n_validate_smrtdata(positive_pickle, negative_pickle):
    """
    Load positive and negative SMRTmatrix pickle files
    :param positive_pickle: str, path to the pickle file with positive samples
    :param negative_pickle: str, path to the pickle file with negative samples
    :return: positive and negative pickle files.
    """
    # Load both pickle files
    positive = pickle.load(open(positive_pickle, "rb"))
    negative = pickle.load(open(negative_pickle, "rb"))

    # Check the sanity of the positive and negative datasets
    # There should be no None types in the data and no label
    # should have value 0 in the positive labels set.
    cnt_none = 0
    cnt_zero = 0
    for i in range(len(positive)):
        if positive[i] is not None:
            if positive[i].label == 0:
                cnt_zero += 1
        else:
            cnt_none += 1

    print(f"Number of positive samples: {len(positive)}")
    assert cnt_none == 0, f"{cnt_none} None types found in {positive_pickle}"
    assert cnt_zero == 0, f"{cnt_zero} labels with value 0 found in {positive_pickle}"
    # There should be no None types in the data and no label
    # should have value 1 in the negative labels set.
    cnt_none = 0
    cnt_one = 0
    for i in range(len(negative)):
        if negative[i] is not None:
            if negative[i].label == 1:
                cnt_one += 1
        else:
            cnt_none += 1

    print(f"Number of negative samples: {len(negative)}")
    assert cnt_none == 0, f"{cnt_none} None types found in {negative_pickle}"
    assert cnt_one == 0, f"{cnt_one} labels with value 1 found in {negative_pickle}"

    return positive, negative


def get_feat_labels_matrix(smrt_obj, req_label=1):
    """
    Get features and labels from the smrtmatrix object. We take
    the read sequences, convert them into PWM, then we take the
    pulse width (PW) and inter-pulse distance (IPD) data for each
    read and average them over each sequence position. This results
    in a feature matrix of shape (15, 6) for each example. 15 is the
    sequence length and 4/6 are nucleotide frequency channels and
    the remaining are the average PW and IPD channels.
    :param smrt_obj: list, list of SMRTmatrix custom objects
                            (see m6Adata.py for details).
    :param req_label: int, required label, 1 for positive set,
                            0 for the negative set.
    :return: features and labels matrix
    """
    # Features list
    total_feat_array = []
    # Labels list
    all_labels = []
    # for every element in the list of
    # smrt_obj
    for i in range(len(smrt_obj)):
        # print every 1000th iteration
        if i % 1000 == 0:
            print(i)
        # If the object is not none
        if smrt_obj[i] is not None:
            # Get the list of sequence reads
            base = smrt_obj[i].base
            # and the labels
            label = smrt_obj[i].label
            # If the label is correct according to the class
            # 1 for positive class and 0 for negative class
            if label == req_label:
                # Get the inter-pulse distance
                ip = smrt_obj[i].ip
                # and the pulse width
                pw = smrt_obj[i].pw
                # get the pwm for all sequences
                base_ohe = get_pwm(base)
                # normalize the ip and pw arrays
                ip = np.array(ip)
                pw = np.array(pw)
                ip = ip / 255
                pw = pw / 255
                # take mean along genomic position axis
                ip_avg = np.mean(ip, axis=0)
                pw_avg = np.mean(pw, axis=0)
                # concatenate the PWN, IPD and PW channels.
                feat_array = np.concatenate((base_ohe,
                                             ip_avg[:, np.newaxis],
                                             pw_avg[:, np.newaxis]), axis=1)

                # append it to the total features array
                total_feat_array.append(feat_array)
                # and the labels array
                all_labels.append(int(label))

    # convert the features and labels list to arrays
    total_feat_array = np.array(total_feat_array)
    all_labels = np.array(all_labels)
    print(f"total_feat_array shape: {total_feat_array.shape}")
    print(f"all_labels shape: {all_labels.shape}")
    return total_feat_array, all_labels


def save_train_test_data(positive_pickle, negative_pickle, save_path_prefix):
    """
    main function to run everything. Takes in the positive and negative class
    file paths and gets the features and labels for both. Then divides the data
    into train, validation and test sets. Train and validation sets are stored
    together and the test set is stored separately.
    :param positive_pickle: str, path to positive pickle file
    :param negative_pickle: str, path to negative pickle file
    :param save_path_prefix: str, path to store train, validation and test data
    :return:
    """
    # Get positive features and labels
    pos_feats, pos_labels = get_feat_labels_matrix(positive_pickle, req_label=1)
    # Get negative features and labels
    neg_feats, neg_labels = get_feat_labels_matrix(negative_pickle, req_label=0)

    # Assert that all positive and negative sets have correct labels
    assert np.sum(pos_labels) == pos_labels.shape[0], "all positives should have a label 1"
    assert np.sum(neg_labels) == 0, "all negatives should have a label 0"

    # Generate final set by concatenating positive and negative features
    final_feats = np.concatenate((pos_feats, neg_feats), axis=0)
    # and labels.
    final_labels = np.concatenate((pos_labels, neg_labels), axis=0)

    # Shuffle to mix in the positives and negatives
    final_shuffle = np.arange(0, len(final_feats), 1, dtype=int)
    np.random.shuffle(final_shuffle)
    final_feats = final_feats[final_shuffle]
    final_labels = final_labels[final_shuffle]

    # reshape from (N, 15, 6) -> (N, 6, 15) to put channels first.
    final_feats = final_feats.reshape((final_feats.shape[0], final_feats.shape[2], final_feats.shape[1]))

    # One-hot-encode labels
    labels_ohe = np.zeros((len(final_labels), 2))
    labels_ohe[np.where(final_labels == 1)[0], 0] = 1
    labels_ohe[np.where(final_labels == 0)[0], 1] = 1

    # Get Stratified split with equal size of validation and test set (hence test set=0.2 and 0.25 because the
    # train set becomes smaller after step 1).
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    # Get the test set
    for train_val_index, test_index in sss_test.split(final_feats, labels_ohe):
        X_train_val, X_test = final_feats[train_val_index], final_feats[test_index]
        y_train_val, y_test = labels_ohe[train_val_index], labels_ohe[test_index]
        # get the validation set.
        for train_index, val_index in sss_val.split(X_train_val, y_train_val):
            X_train, X_val = X_train_val[train_index], X_train_val[val_index]
            y_train, y_val = y_train_val[train_index], y_train_val[val_index]
            print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
            print(f"y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")

    # Save train and validation data
    save_data_dict = dict()
    save_data_dict['X_train'] = X_train
    save_data_dict['y_train'] = y_train
    save_data_dict['X_val'] = X_val
    save_data_dict['y_val'] = y_val

    save_path = save_path_prefix + "m6A_train"

    np.savez(save_path, save_data_dict=save_data_dict)

    # save test data
    save_data_dict = dict()
    save_data_dict['X_test'] = X_test
    save_data_dict['y_test'] = y_test
    save_path = save_path_prefix + "m6A_test"
    np.savez(save_path, save_data_dict=save_data_dict)

    # Print the number of positive and negative examples
    # in each set.
    count_pos_neg(y_train, set_name="Train")
    count_pos_neg(y_val, set_name="Validation")
    count_pos_neg(y_test, set_name="Test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--positive_pickle',
        type=str,
        default="data/PositiveSMRTmatrix.pkl",
        help="path to the positive sample pickle file."
    )

    parser.add_argument(
        '--negative_pickle',
        type=str,
        default='data/LargeNegativeSMRTmatrix.pkl',
        help="path to the negative sample pickle file."
    )

    parser.add_argument(
        '--save_path_prefix',
        type=str,
        default="data/",
        help="Where do you want to save the data. Default is current directory"
    )

    args = parser.parse_args()
    
    positive_pickle, negative_pickle = get_n_validate_smrtdata(args.positive_pickle, args.negative_pickle)
    
    save_train_test_data(positive_pickle, negative_pickle, args.save_path_prefix)
