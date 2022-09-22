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


def one_hot_encode(
    sequence,
    ignore="N",
    alphabet=None,
    dtype="int8",
    desc=None,
    verbose=False,
    **kwargs,
):
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


def get_pwm(seq_array, alphabet=["A", "C", "G", "T"]):
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


def get_n_validate_smrtdata_both(both_pickle):
    # Load pickle file
    all_samples = pickle.load(open(both_pickle, "rb"))
    print(f"The {both_pickle} dataset has {len(all_samples)} samples.")
    cnt_zero = 0
    cnt_one = 0
    cnt_none = 0
    positive = []
    negative = []
    for i in range(len(all_samples)):
        if all_samples[i] is not None:
            if all_samples[i].label == 0:
                cnt_zero += 1
                negative.append(all_samples[i])
            elif all_samples[i].label == 1:
                cnt_one += 1
                positive.append(all_samples[i])
        else:
            cnt_none += 1

    assert cnt_none == 0, f"{cnt_none} None types found in {both_pickle}"
    print(f"{both_pickle} has {cnt_zero} negatives and {cnt_one} positives")

    return positive, negative


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
    subread_counts = []
    ccss = []
    m6a_call_positions = []
    strands = []
    # for every element in the list of
    # smrt_obj
    for i in tqdm.tqdm(range(len(smrt_obj)), total=len(smrt_obj)):
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
                # print(f"ip: {ip.shape}")
                # print(f"pw: {pw.shape}")
                # take mean along genomic position axis
                ip_avg = np.mean(ip, axis=0)
                pw_avg = np.mean(pw, axis=0)

                # print(f"ip_avg: {ip_avg.shape}")
                # print(f"pw_avg: {pw_avg.shape}")

                offset = np.abs(smrt_obj[i].offset)

                # print(f"offset: {offset.shape}")

                offset_mean = np.mean(offset, axis=0)

                # print(f"offset_mean: {offset_mean.shape}")

                offset_mean = 1.0 / (offset_mean + 1.0)

                # concatenate the PWN, IPD and PW channels.
                feat_array = np.concatenate(
                    (
                        base_ohe,
                        ip_avg[:, np.newaxis],
                        pw_avg[:, np.newaxis],
                        offset_mean.T,
                    ),
                    axis=1,
                )

                # append it to the total features array
                total_feat_array.append(feat_array)
                # and the labels array
                all_labels.append(int(label))
                subread_counts.append(smrt_obj[i].subread_count)
                ccss.append(smrt_obj[i].ccs)
                m6a_call_positions.append(smrt_obj[i].m6a_call_position)
                strands.append(smrt_obj[i].strand)

    # convert the features and labels list to arrays
    total_feat_array = np.array(total_feat_array)
    all_labels = np.array(all_labels)
    subread_counts = np.array(subread_counts)
    ccss = np.array(ccss)
    m6a_call_positions = np.array(m6a_call_positions)
    strands = np.array(strands)

    others = [subread_counts, ccss, m6a_call_positions, strands]

    print(f"total_feat_array shape: {total_feat_array.shape}")
    print(f"all_labels shape: {all_labels.shape}")

    return total_feat_array, all_labels, others


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
    pos_feats, pos_labels, pos_others = get_feat_labels_matrix(
        positive_pickle, req_label=1
    )
    # Get negative features and labels
    neg_feats, neg_labels, neg_others = get_feat_labels_matrix(
        negative_pickle, req_label=0
    )

    pos_subread_counts, pos_ccss, pos_m6a_call_positions, pos_strands = pos_others

    neg_subread_counts, neg_ccss, neg_m6a_call_positions, neg_strands = neg_others

    # Assert that all positive and negative sets have correct labels
    assert (
        np.sum(pos_labels) == pos_labels.shape[0]
    ), "all positives should have a label 1"
    assert np.sum(neg_labels) == 0, "all negatives should have a label 0"

    # Generate final set by concatenating positive and negative features
    final_feats = np.concatenate((pos_feats, neg_feats), axis=0)
    # and labels.
    final_labels = np.concatenate((pos_labels, neg_labels), axis=0)

    # Other stats

    final_subread_counts = np.concatenate(
        (pos_subread_counts, neg_subread_counts), axis=0
    )
    final_ccss = np.concatenate((pos_ccss, neg_ccss), axis=0)
    final_m6a_call_pos = np.concatenate(
        (pos_m6a_call_positions, neg_m6a_call_positions), axis=0
    )
    final_strands = np.concatenate((pos_strands, neg_strands), axis=0)

    # Shuffle to mix in the positives and negatives
    final_shuffle = np.arange(0, len(final_feats), 1, dtype=int)
    np.random.shuffle(final_shuffle)
    final_feats = final_feats[final_shuffle]
    final_labels = final_labels[final_shuffle]

    final_subread_counts = final_subread_counts[final_shuffle]
    final_ccss = final_ccss[final_shuffle]
    final_m6a_call_pos = final_m6a_call_pos[final_shuffle]
    final_strands = final_strands[final_shuffle]

    # reshape from (N, 15, 6) -> (N, 6, 15) to put channels first.
    final_feats = np.moveaxis(final_feats, [1, 2], [2, 1])
    # final_feats = final_feats.reshape((final_feats.shape[0], final_feats.shape[2], final_feats.shape[1]))

    # One-hot-encode labels
    labels_ohe = np.zeros((len(final_labels), 2))
    labels_ohe[np.where(final_labels == 1)[0], 0] = 1
    labels_ohe[np.where(final_labels == 0)[0], 1] = 1

    # Stratify based on ccs read
    unique_ccss = np.unique(final_ccss)

    train_ccs = unique_ccss[0 : int(0.6 * (len(unique_ccss)))]
    val_ccs = unique_ccss[int(0.6 * (len(unique_ccss))) : int(0.8 * (len(unique_ccss)))]
    test_ccs = unique_ccss[int(0.8 * (len(unique_ccss))) :]

    X_train_ccs = []
    y_train_ccs = []
    train_subread_counts_ccs = []
    train_final_ccss_ccs = []
    train_final_strands_ccs = []
    train_final_m6a_call_positions_ccs = []

    X_val_ccs = []
    y_val_ccs = []
    val_subread_counts_ccs = []
    val_final_ccss_ccs = []
    val_final_strands_ccs = []
    val_final_m6a_call_positions_ccs = []

    X_test_ccs = []
    y_test_ccs = []
    test_subread_counts_ccs = []
    test_final_ccss_ccs = []
    test_final_strands_ccs = []
    test_final_m6a_call_positions_ccs = []

    for i in range(len(final_ccss)):
        if final_ccss[i] in train_ccs:
            X_train_ccs.append(final_feats[i])
            y_train_ccs.append(labels_ohe[i])
            train_subread_counts_ccs.append(final_subread_counts[i])
            train_final_ccss_ccs.append(final_ccss[i])
            train_final_strands_ccs.append(final_strands[i])
            train_final_m6a_call_positions_ccs.append(final_m6a_call_pos[i])

    for i in range(len(final_ccss)):
        if final_ccss[i] in val_ccs:
            X_val_ccs.append(final_feats[i])
            y_val_ccs.append(labels_ohe[i])
            val_subread_counts_ccs.append(final_subread_counts[i])
            val_final_ccss_ccs.append(final_ccss[i])
            val_final_strands_ccs.append(final_strands[i])
            val_final_m6a_call_positions_ccs.append(final_m6a_call_pos[i])

    for i in range(len(final_ccss)):
        if final_ccss[i] in test_ccs:
            X_test_ccs.append(final_feats[i])
            y_test_ccs.append(labels_ohe[i])
            test_subread_counts_ccs.append(final_subread_counts[i])
            test_final_ccss_ccs.append(final_ccss[i])
            test_final_strands_ccs.append(final_strands[i])
            test_final_m6a_call_positions_ccs.append(final_m6a_call_pos[i])

    print(f"X_train_ccs: {len(X_train_ccs)}")
    print(f"X_val_ccs: {len(X_val_ccs)}")
    print(f"X_test_ccs: {len(X_test_ccs)}")

    # Save train and validation data
    save_data_dict = dict()
    save_data_dict["X_train"] = np.array(X_train_ccs)
    save_data_dict["y_train"] = np.array(y_train_ccs)
    save_data_dict["X_val"] = np.array(X_val_ccs)
    save_data_dict["y_val"] = np.array(y_val_ccs)

    save_data_dict["train_subread_counts"] = np.array(train_subread_counts_ccs)
    save_data_dict["val_subread_counts"] = np.array(val_subread_counts_ccs)
    save_data_dict["train_final_ccss"] = np.array(train_final_ccss_ccs)
    save_data_dict["val_final_ccss"] = np.array(val_final_ccss_ccs)

    save_data_dict["train_final_strands"] = np.array(train_final_strands_ccs)
    save_data_dict["val_final_strands"] = np.array(val_final_strands_ccs)
    save_data_dict["train_final_m6a_call_positions"] = np.array(
        train_final_m6a_call_positions_ccs
    )
    save_data_dict["val_final_m6a_call_positions"] = np.array(
        val_final_m6a_call_positions_ccs
    )

    save_path = save_path_prefix + "m6A_train"

    np.savez(save_path, save_data_dict=save_data_dict)

    # save test data
    save_data_dict = dict()
    save_data_dict["X_test"] = X_test_ccs
    save_data_dict["y_test"] = y_test_ccs

    save_data_dict["test_subread_counts"] = test_subread_counts_ccs
    save_data_dict["test_final_ccss"] = test_final_ccss_ccs
    save_data_dict["test_final_strands"] = test_final_strands_ccs
    save_data_dict["test_final_m6a_call_positions"] = test_final_m6a_call_positions_ccs

    save_path = save_path_prefix + "m6A_test"
    np.savez(save_path, save_data_dict=save_data_dict)

    # Print the number of positive and negative examples
    # in each set.
    count_pos_neg(np.array(y_train_ccs), set_name="Train")
    count_pos_neg(np.array(y_val_ccs), set_name="Validation")
    count_pos_neg(np.array(y_test_ccs), set_name="Test")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "both_pickle",
        type=str,
        help="path to the pickle file with both positive and negative samples.",
    )
    parser.add_argument(
        "--save-path-prefix",
        type=str,
        default="data/",
        help="Where do you want to save the data.",
    )
    args = parser.parse_args()
    positive_pickle, negative_pickle = get_n_validate_smrtdata_both(args.both_pickle)
    save_train_test_data(positive_pickle, negative_pickle, args.save_path_prefix)


if __name__ == "__main__":
    main()
