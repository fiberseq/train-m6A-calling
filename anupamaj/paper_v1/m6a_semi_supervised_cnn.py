"""
m6a_semi_supervised_cnn.py
Author: Anupama Jha <anupamaj@uw.edu>
This module predicts whether an
adenine is methylated or not. The
model is trained with Fiber-seq
HiFi read sequence, inter-pulse
distance and pulse width signal
from pacbio. The model is a
convolution neural network. The
training scheme is semi-supervised
where we assume that the m6A labels
are a mixed set with false
positives and the negative labels
are a clean set.
"""

import torch
import argparse
import numpy as np
import configparser
from m6a_cnn import M6ANet
from torchsummary import summary
from sklearn.metrics import (average_precision_score,
                             roc_auc_score)

verbose=False


def count_pos_neg(labels,
                  set_name=""):
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
        print(f"{set_name} has {num_pos}"
              f" positives and {num_neg} negatives")
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
    y_array_ohe = np.zeros(
        (len(y_array),
         2)
    )
    one_idx = np.where(
        y_array == 1
    )[0]
    y_array_ohe[one_idx, 0] = 1

    zero_idx = np.where(
        y_array == 0
    )[0]
    y_array_ohe[zero_idx, 1] = 1
    return y_array_ohe


class M6ADataGenerator(torch.utils.data.Dataset):
    """
    Data generator for the m6A
    model. It randomly selects
    batches from the data to
    train the m6A model.
    """

    def __init__(self,
                 features,
                 labels):
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


def m6AGenerator(train_path,
                 val_path,
                 input_size,
                 random_state=None,
                 train_sample=True,
                 train_sample_fraction=0.5,
                 val_sample=True,
                 val_sample_fraction=0.1):
    """
    This generator returns a training
    data generator as well as validation
    features and labels.
    :param train_path: str, path where
                            the training
                            data is stored.
    :param val_path: str, path where the
                          val data is stored.
    :param input_size: int, number of input channels
    :param random_state: np.random, random seed
    :param train_sample: bool, sample train data
    :param train_sample_fraction: float, what fraction
                                         to sample
    :param val_sample: bool, sample validation data
    :param val_sample_fraction: float, what fraction
                                       to sample
    :return: X_gen: training data generator,
             X_val: validation features,
             y_val: validation labels
    """
    # Load training data
    train_data = np.load(
        train_path,
        allow_pickle=True
    )
    # initialize Random state
    random_state = np.random.RandomState(
        random_state
    )

    # Load training and validation
    # features and labels. Sometimes
    # we want to train on input subsets,
    # this will achieve that.
    X_train = train_data["features"]
    X_train = X_train[:, 0:input_size, :]
    y_train = train_data["labels"]

    if train_sample:
        rand_val = np.random.choice(
            np.arange(
                len(y_train), dtype=int),
            size=(
                int(train_sample_fraction * len(y_train)),
            ),
            replace=False
        )

        X_train = X_train[rand_val, :, :]
        y_train = y_train[rand_val]

    # Load validation data
    val_data = np.load(
        val_path,
        allow_pickle=True
    )

    X_val = val_data['features']
    X_val = X_val[:, 0:input_size, :]
    y_val = val_data['labels']

    if val_sample:
        rand_val = np.random.choice(
            np.arange(
                len(y_val), dtype=int),
            size=(
                int(val_sample_fraction * len(y_val)),
            ),
            replace=False
        )

        X_val = X_val[rand_val, :, :]
        y_val = y_val[rand_val]

    print(f"Training features shape {X_train.shape},"
          f" training labels shape: {y_train.shape}")
    print(f"Validation features shape {X_val.shape}, "
          f" validation labels shape: {y_val.shape}")

    count_pos_neg(
        y_train,
        set_name="Train"
    )
    count_pos_neg(
        y_val,
        set_name="Validation"
    )

    return X_train, y_train, X_val, y_val, random_state


def _fdr2qvalue(fdr,
                num_total,
                met,
                indices):
    """
    Method from Will Fondrie's mokapot,
    remember to cite:
    https://github.com/wfondrie/mokapot
    Quickly turn a list of FDRs to q-values.
    All of the inputs are assumed to be sorted.
    :param fdr: numpy.ndarray, A vector of all
                         unique FDR values.
    :param num_total: numpy.ndarray, A vector of
                               the cumulative
                               number of PSMs
                               at each score.
    :param met: numpy.ndarray, A vector of the scores
                         for each PSM.
    :param indices: tuple of numpy.ndarray,
                        Tuple where the vector
                        at index i indicates
                        the PSMs that shared
                        the unique FDR value in `fdr`.
    :return: numpy.ndarray, A vector of q-values.
    """
    min_q = 1
    qvals = np.ones(len(fdr))
    group_fdr = np.ones(len(fdr))
    prev_idx = 0
    for idx in range(met.shape[0]):
        next_idx = prev_idx + indices[idx]
        group = slice(prev_idx, next_idx)
        prev_idx = next_idx

        fdr_group = fdr[group]
        n_group = num_total[group]
        curr_fdr = fdr_group[np.argmax(n_group)]
        if curr_fdr < min_q:
            min_q = curr_fdr

        group_fdr[group] = curr_fdr
        qvals[group] = min_q

    return qvals


def tdc(scores,
        target,
        pn_ratio=2,
        desc=True):
    """
    Method from Will Fondrie's mokapot,
    remember to cite:
    https://github.com/wfondrie/mokapot

    Estimate q-values using target decoy
    competition. Estimates q-values using
    the simple target decoy competition method.
    For set of target and decoy PSMs meeting a
    specified score threshold, the false discovery
    rate (FDR) is estimated as:
    ...math:
        FDR = \frac{2*Decoys}{Targets + Decoys}
    More formally, let the scores of target and
    decoy PSMs be indicated as
    :math:`f_1, f_2, ..., f_{m_f}` and
    :math:`d_1, d_2, ..., d_{m_d}`,
    respectively. For a score threshold
    :math:`t`, the false discovery
    rate is estimated as:
    ...math:
        E\\{FDR(t)\\} = \frac{|\\{d_i > t; i=1, ..., m_d\\}| + 1}
        {\\{|f_i > t; i=1, ..., m_f|\\}}
    The reported q-value for each PSM is the
    minimum FDR at which that PSM would be accepted.

    :param scores: numpy.ndarray of float
        A 1D array containing the score to rank by
    :param target : numpy.ndarray of bool
        A 1D array indicating if the entry is from
        a target or decoy hit. This should be boolean,
        where `True` indicates a target and `False`
        indicates a decoy. `target[i]` is the label
        for `metric[i]`; thus `target` and `metric`
        should be of equal length.
    :param pn_ratio: float, ratio of positive/negative
                            class examples
    :param desc : bool Are higher scores better?
                      `True` indicates that they are,
                      `False` indicates that they are not.

    :returns: numpy.ndarray
        A 1D array with the estimated q-value for each entry.
        The array is the same length as the `scores` and
        `target` arrays.
    """
    scores = np.array(scores)

    try:
        target = np.array(
            target,
            dtype=bool
        )
    except ValueError:
        raise ValueError(
            "'target' should be boolean."
        )

    if scores.shape[0] != target.shape[0]:
        raise ValueError(
            "'scores' and 'target' must be the same length"
        )

    # Unsigned integers can cause weird things to happen.
    # Convert all scores to floats to for safety.
    if np.issubdtype(
            scores.dtype,
            np.integer
    ):
        scores = scores.astype(np.float_)

    # Sort and estimate FDR
    if desc:
        srt_idx = np.argsort(-scores)
    else:
        srt_idx = np.argsort(scores)

    scores = scores[srt_idx]
    target = target[srt_idx]

    cum_targets = target.cumsum()

    cum_decoys = (
            (target - 1) ** 2
    ).cumsum()
    num_total = cum_targets + cum_decoys

    # Handles zeros in denominator
    fdr = np.divide(
        (1 + pn_ratio) * cum_decoys,
        (cum_targets + cum_decoys),
        out=np.ones_like(cum_targets, dtype=float),
        where=(cum_targets != 0),
    )

    # Calculate q-values
    unique_metric, indices = np.unique(
        scores,
        return_counts=True
    )

    # Some arrays need to be flipped
    # so that we can loop through from
    # worse to best score.
    fdr = np.flip(fdr)
    num_total = np.flip(num_total)
    if not desc:
        unique_metric = np.flip(unique_metric)
        indices = np.flip(indices)

    qvals = _fdr2qvalue(
        fdr,
        num_total,
        unique_metric,
        indices
    )
    qvals = np.flip(qvals)
    qvals = qvals[np.argsort(srt_idx)]

    return qvals


def compute_pos_neg_sets(x_data,
                         scores,
                         y_data,
                         score_threshold):
    """
    Compute positive and negative sets using
    the validation score threshold.
    :param x_data: np.array, features are
                             stored here.
    :param scores: np.array, CNN scores
                             for data.
    :param y_data: np.array, m6A labels.
    :param score_threshold: float, score threshold
                                   based on validation
                                   data.
    :return: new features, new labels and all labels.
    """
    pos_set_score = np.where(
        scores >= score_threshold
    )[0]

    pos_set_all = np.where(
        y_data == 1
    )[0]

    pos_set = np.intersect1d(
        pos_set_score,
        pos_set_all
    )

    neg_set = np.where(
        y_data == 0
    )[0]

    y_data_init = np.zeros(
        (len(pos_set) + len(neg_set),)
    )

    y_data_init[0:len(pos_set)] = 1

    x_data_init = np.concatenate(
        (x_data[pos_set, :, :],
         x_data[neg_set, :, :])
    )

    shuffle_idx = np.arange(
        len(y_data_init),
        dtype=int
    )
    np.random.shuffle(shuffle_idx)

    x_data_init = x_data_init[shuffle_idx, :, :]
    y_data_init = y_data_init[shuffle_idx]

    count_pos_neg(
        y_data_init,
        set_name="New training set"
    )
    return x_data_init, y_data_init


def compute_fdr_score(scores,
                      y_data,
                      fdr_threshold=0.1):
    """
    Compute FDR score threshold using the validation data.
    :param scores: np.array, array of CNN scores
    :param y_data: np.array, ground truth labels
    :param fdr_threshold: float, false discovery rate
    :return:score_thresholds and number of positives
    """
    num_pos_class = len(
        np.where(
            y_data == 1
        )[0]
    )
    num_neg_class = len(
        np.where(
            y_data == 0
        )[0]
    )
    # ratio of positives to negatives
    pn_ratio = num_pos_class / float(num_neg_class)
    if verbose:
        print(f"positive class to negative"
              f" class ratio: {pn_ratio}")

    ipd_fdr = tdc(
        scores,
        y_data,
        pn_ratio,
        desc=True
    )
    if verbose:
        print(f"ipd_fdr: min: {np.min(ipd_fdr)}, "
              f"max: {np.max(ipd_fdr)}, "
              f"mean: {np.mean(ipd_fdr)}, "
              f"median: {np.median(ipd_fdr)}, "
              f"std: {np.std(ipd_fdr)}")

    # Get positive set
    # from samples with
    # positive label
    # and FDR below the
    # threshold.
    pos_set = np.where(
        ipd_fdr <= fdr_threshold
    )[0]

    # If we found no samples,
    # relax FRD criteria
    # to get an initial
    # positive set.
    if len(pos_set) == 0:
        pos_set = np.where(
            ipd_fdr <= np.min(ipd_fdr)
        )[0]

    if torch.is_tensor(pos_set):
        pos_set = pos_set.numpy()

    if torch.is_tensor(scores):
        scores = scores.numpy()

    pos_scores = scores[pos_set]
    # Get the score threshold
    # for positive examples
    # which are below a certain
    # threshold.
    score_thresholds = np.min(pos_scores)

    # Number of positives
    # below the FDR threshold
    num_pos = len(pos_scores)

    return score_thresholds, num_pos


def main(config_file,
         train_chem):
    """
    Run data preprocess and model training.
    :param config_file: str, path to config
                            file.
    :param train_chem: str, which chemistry
                            to train.
    :return:
    """
    # read parameters from config file
    config = configparser.ConfigParser()
    config.read(config_file)

    # get parameters for the relevant chemistry
    rel_config = config[train_chem]
    input_size = int(rel_config["input_size"])
    
    input_length = int(rel_config["input_length"])

    train_data = rel_config["semi_train_data"]

    val_data = rel_config["semi_val_data"]

    device = rel_config["device"]

    best_sup_save_model = rel_config["pretrain_model_name"]

    best_save_model = rel_config["best_semi_supervised_model_name"]

    final_save_model = rel_config["final_semi_supervised_model_name"]

    max_epochs = int(rel_config["semi_supervised_train_epochs"])

    fdr_threshold = float(rel_config["fdr"])

    save_pos = rel_config["save_pos"]
    
    num_workers = int(rel_config["semi_num_workers"])
    
    batch_size = int(rel_config["semi_batch_size"])
    
    train_sample = bool(rel_config["train_sample"])
    
    train_sample_fraction = float(rel_config["train_sample_fraction"])
    
    val_sample = bool(rel_config["val_sample"])
    
    val_sample_fraction = float(rel_config["val_sample_fraction"])
    # learning rate
    semi_lr=float(rel_config["semi_lr"])
    
    min_pos_proportion = float(rel_config["min_pos_proportion"])

    # Load the supervised model
    # for transfer learning
    model = torch.load(
        best_sup_save_model,
        map_location=torch.device(device)
    )

    # Adam optimizer with learning
    # rate 1e-4
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=semi_lr
    )

    # Print model architecture summary
    summary_str = summary(
        model,
        input_size=(input_size,
                    input_length)
    )
    
    # keep track of validation set's
    # number of positives below FDR < 0.05
    # precision in supervised setting
    # and score threshold for validation
    # FDR <= 0.05
    all_num_pos = []
    val_ap = []
    val_scores = []
    
    # If the initial validation
    # set is sampled, we need 
    # to make sure that the 
    # sample has at least 100
    # m6A calls below the 
    # FDR threshold
    regenerate = True
    # to avoid infinite 
    # repetitions, introduce
    # num_repeat, which will
    # give the while loop 5 
    # chances to generate good
    # intial sample, if desired
    # sample is not generated by
    # the end of fifth iteration,
    # proceed with the given 
    # set with a warning. 
    num_repeat = 0
    while regenerate:
        # Get training data generator
        # and validation data.
        gen_outputs = m6AGenerator(
            train_data,
            val_data,
            input_size=input_size,
            random_state=None, 
            train_sample=train_sample,
            train_sample_fraction=train_sample_fraction,
            val_sample=val_sample,
            val_sample_fraction=val_sample_fraction
        )

        X_train, y_train, X_val, y_val, random_state = gen_outputs


        # Use Inter-pulse distance
        # for the central base as
        # the initial classifier
        y_score_val = X_val[:, 4, 7]

        # Convert y_val to
        # a one-hot-encoded
        # vector
        y_val_ohe = make_one_hot_encoded(
            y_val
        )

        # Compute initial score threshold
        # for the inter-pulse distance
        # classifier using the FDR <= 0.05
        # critera and the validation data
        score_threshold, num_pos = compute_fdr_score(
            y_score_val,
            np.array(y_val, dtype=bool),
            fdr_threshold=fdr_threshold
        )
        
        pos_proportion = (float(num_pos)/len(y_score_val))
        if pos_proportion >= min_pos_proportion:
            regenerate = False
        elif val_sample and pos_proportion < min_pos_proportion:
            if num_repeat < 5:
                regenerate = True
                print(f"Number of positives sampled at FDR {fdr_threshold} is {num_pos},"
                      f" needs to be >= {int(len(y_score_val)*min_pos_proportion)}, "
                      f" sampling again.")
            else:
                print(f"Number of positives sampled at FDR {fdr_threshold} is {num_pos},"
                  f" needs to be >= {int(len(y_score_val)*min_pos_proportion)} "
                  f" optimization may not converge.")
                regenerate = False
                
        else:
            print(f"Number of positives at FDR {fdr_threshold} is {num_pos},"
                  f" needs to be >= {int(len(y_score_val)*min_pos_proportion)} "
                  f" optimization may not converge.")
            regenerate = False
        num_repeat += 1
            
            
    # store the number of
    # positives identified
    # and the score theshold
    all_num_pos.append(num_pos)
    val_scores.append(score_threshold)

    # Compute the average precision
    # of the initial inter-pulse
    # distance classifier
    sklearn_ap = average_precision_score(
        y_val,
        y_score_val
    )
    
    # Compute total number of positives
    # and negatives in the validation set
    val_pos_all, val_neg_all = count_pos_neg(
        y_val,
        set_name="Validation set"
    )

    print(f"Validation IPD average precision: "
          f"{sklearn_ap}, Number of positives "
          f" at FDR of {fdr_threshold} are: {num_pos}")

    # store the initial
    # average precision
    val_ap.append(sklearn_ap)

    # Use Inter-pulse distance
    # for the central base as
    # the initial classifier
    # Get score for training
    # data
    y_score = X_train[:, 4, 7]
    if verbose:
        print(f"y_score: {y_score.shape}, "
              f"min: {np.min(y_score)}, "
              f"max: {np.max(y_score)}, "
              f"mean: {np.mean(y_score)}, "
              f"std: {np.std(y_score)}")

    # Get initial training set using initial
    # score determined using IPD classifier
    # on validation data.
    X_init, y_init = compute_pos_neg_sets(
        X_train,
        y_score,
        np.array(y_train, dtype=bool),
        score_threshold
    )

    # Convert y_init to a one-hot-encoded
    # vector
    y_init_ohe = make_one_hot_encoded(
        y_init
    )

    # keep all training data and
    # all validation data for
    # precision computations
    X_init_cpu = torch.tensor(X_train)
    X_init_cpu = X_init_cpu.float()
    X_val_cpu = torch.tensor(X_val)
    X_val_cpu = X_val_cpu.float()

    # Begin the semi-supervised
    # loop
    for i in range(max_epochs):
        # Get the training data generator
        X_gen = M6ADataGenerator(X_init,
                                 y_init_ohe)

        # Wrap it in a data loader
        X_gen = torch.utils.data.DataLoader(X_gen,
                                            pin_memory=True,
                                            num_workers=num_workers,
                                            batch_size=batch_size,
                                            shuffle=True)
        
        validation_iter = int(np.floor(len(y_init_ohe)/(batch_size)))
        
        # Train the model
        model.fit_semisupervised(X_gen,
                                 optimizer,
                                 X_valid=X_val, 
                                 y_valid=y_val_ohe,
                                 max_epochs=2,
                                 validation_iter=validation_iter,
                                 device=device,
                                 prev_aupr=sklearn_ap,
                                 best_save_model=best_save_model,
                                 final_save_model=final_save_model)
        
        sklearn_ap = model.evaluate(X_val,
                                    y_val_ohe, 
                                    device=device)
        # Get current model's AUPR
        # on the validation data
        val_ap.append(sklearn_ap)

        # Generate scores for
        # the validation data
        # using the current CNN model
        y_score_val = model.predict(
            X_val_cpu,
            device=device
        )

        # Compute new score threshold
        # using the new validation
        # set predictions
        score_threshold, num_pos = compute_fdr_score(
            y_score_val[:, 0],
            np.array(y_val, dtype=bool),
            fdr_threshold=fdr_threshold)

        # Store the number of positives in
        # the validation set and score threshold
        # for the positives below FDR <= 0.05.
        all_num_pos.append(num_pos)
        val_scores.append(score_threshold)

        print(f"Validation CNN epoch {i}"
              f" average precision: {sklearn_ap}, "
              f" Number of positives at estimated precision of "
              f"{1.0-fdr_threshold} are: {num_pos}")

        # Generate scores for
        # all the training data
        y_score = model.predict(
            X_init_cpu,
            device=device
        )

        # Compute positive and negative
        # sets using the validation set
        # determined score_threshold
        # and training set data
        X_init, y_init = compute_pos_neg_sets(
            X_train,
            y_score[:, 0],
            np.array(y_train, dtype=bool),
            score_threshold
        )

        # Convert y_init to a one-hot-encoded
        # vector
        y_init_ohe = make_one_hot_encoded(
            y_init
        )

        # Store the total positives
        # in the validation set,
        # and lists of validation
        # precision and number of
        # positives identified as
        # training progressed.
        np.savez(
            save_pos,
            total_pos=val_pos_all,
            num_pos=all_num_pos,
            val_ap=val_ap,
            val_score=val_scores
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config_file',
        type=str,
        default="config.yml",
        help="path to the config file."
    )

    parser.add_argument(
        '--train_chem',
        type=str,
        default="train_2_2_chemistry",
        choices=["train_2_2_chemistry",
                 "train_3_2_chemistry",
                 "train_revio_chemistry"],
        help="which chemistry to train."
    )

    args = parser.parse_args()

    print(f"Training a {args.train_chem} "
          f"semi-supervised CNN model.")

    main(
        args.config_file,
        args.train_chem
    )
