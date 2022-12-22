import torch
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from torchsummary import summary
from m6a_semi_supervised_ipd import M6ANet
from sklearn.metrics import (confusion_matrix,
                             accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             balanced_accuracy_score,
                             matthews_corrcoef,
                             roc_auc_score,
                             average_precision_score,
                             roc_curve,
                             precision_recall_curve,
                             auc
                             )


def count_pos_neg(labels, set_name=""):
    """
    Count the number of positive (m6A) and negative (everything else) examples in our set
    :param labels: np.array, one-hot-encoded label
    :param set_name: str, training, validation or test set.
    :return:
    """
    # First column are positive labels
    m6as = np.where(labels == 1)[0]
    # Second column are negative labels
    nulls = np.where(labels == 0)[0]
    num_pos = len(m6as)
    num_neg = len(nulls)
    print(f"{set_name} has {num_pos} positives and {num_neg} negatives")
    return num_pos, num_neg


def m6AGenerator(val_path, random_state=None):
    """
    This generator returns a training data generator as well as
    validation features and labels
    :param train_path: str, path where the training data is stored.
    :param val_path: str, path where the val data is stored.
    :param random_state: int, seed for numpy random_state.
    :param pin_memory: bool, Makes CUDA efficient by skipping one copy
                             operation, true by default.
    :param num_workers: int, number of worker threads the generator can
                             utilize, 0 by default.
    :param batch_size: int, number of examples in each batch
    :return: X_gen: training data generator, X_val: validation features, y_val: validation labels
    """
    # initialize Random state
    random_state = np.random.RandomState(random_state)


    # Load validation data
    val_data = np.load(val_path, allow_pickle=True)
    X_val = val_data['features']
    y_val = val_data['labels']
    
    # Take 50% val labels
    rand_val = np.random.choice(np.arange(len(y_val), dtype=int),
                                size=(int(0.1 * len(y_val)),), replace=False)

    X_val = X_val[rand_val, :, :]
    y_val = y_val[rand_val]
    
    print(f"Validation features shape {X_val.shape}, validation labels shape: {y_val.shape}")

    count_pos_neg(y_val, set_name="Validation")

    return X_val, y_val, random_state


def _fdr2qvalue(fdr, num_total, met, indices):
    """
    Quickly turn a list of FDRs to q-values.
    All of the inputs are assumed to be sorted.
    Parameters
    ----------
    fdr : numpy.ndarray
        A vector of all unique FDR values.
    num_total : numpy.ndarray
        A vector of the cumulative number of PSMs at each score.
    met : numpy.ndarray
        A vector of the scores for each PSM.
    indices : tuple of numpy.ndarray
        Tuple where the vector at index i indicates the PSMs that
        shared the unique FDR value in `fdr`.
    Returns
    -------
    numpy.ndarray
        A vector of q-values.
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


def tdc(scores, target, pn_ratio=2, desc=True):
    """
    Method from Will Fondrie's mokapot, remember to cite:
    https://github.com/wfondrie/mokapot/blob/master/mokapot/qvalues.py

    Estimate q-values using target decoy competition.
    Estimates q-values using the simple target decoy competition method.
    For set of target and decoy PSMs meeting a specified score threshold,
    the false discovery rate (FDR) is estimated as:
    ...math:
        FDR = \frac{2*Decoys}{Targets + Decoys}
    More formally, let the scores of target and decoy PSMs be indicated as
    :math:`f_1, f_2, ..., f_{m_f}` and :math:`d_1, d_2, ..., d_{m_d}`,
    respectively. For a score threshold :math:`t`, the false discovery
    rate is estimated as:
    ...math:
        E\\{FDR(t)\\} = \frac{|\\{d_i > t; i=1, ..., m_d\\}| + 1}
        {\\{|f_i > t; i=1, ..., m_f|\\}}
    The reported q-value for each PSM is the minimum FDR at which that
    PSM would be accepted.
    Parameters
    ----------
    scores : numpy.ndarray of float
        A 1D array containing the score to rank by
    target : numpy.ndarray of bool
        A 1D array indicating if the entry is from a target or decoy
        hit. This should be boolean, where `True` indicates a target
        and `False` indicates a decoy. `target[i]` is the label for
        `metric[i]`; thus `target` and `metric` should be of
        equal length.
    pn_ratio: float, ratio of positive/negative class examples
    desc : bool
        Are higher scores better? `True` indicates that they are,
        `False` indicates that they are not.
    Returns
    -------
    numpy.ndarray
        A 1D array with the estimated q-value for each entry. The
        array is the same length as the `scores` and `target` arrays.
    """
    scores = np.array(scores)

    try:
        target = np.array(target, dtype=bool)
    except ValueError:
        raise ValueError("'target' should be boolean.")

    if scores.shape[0] != target.shape[0]:
        raise ValueError("'scores' and 'target' must be the same length")

    # Unsigned integers can cause weird things to happen.
    # Convert all scores to floats to for safety.
    if np.issubdtype(scores.dtype, np.integer):
        scores = scores.astype(np.float_)

    # Sort and estimate FDR
    if desc:
        srt_idx = np.argsort(-scores)
    else:
        srt_idx = np.argsort(scores)

    scores = scores[srt_idx]
    target = target[srt_idx]
    cum_targets = target.cumsum()
    cum_decoys = ((target - 1) ** 2).cumsum()
    num_total = cum_targets + cum_decoys

    # Handles zeros in denominator
    fdr = np.divide(
        (1 + pn_ratio) * cum_decoys,
        (cum_targets + cum_decoys),
        out=np.ones_like(cum_targets, dtype=float),
        where=(cum_targets != 0),
    )

    # Calculate q-values
    unique_metric, indices = np.unique(scores, return_counts=True)

    # Some arrays need to be flipped so that we can loop through from
    # worse to best score.
    fdr = np.flip(fdr)
    num_total = np.flip(num_total)
    if not desc:
        unique_metric = np.flip(unique_metric)
        indices = np.flip(indices)

    qvals = _fdr2qvalue(fdr, num_total, unique_metric, indices)
    qvals = np.flip(qvals)
    qvals = qvals[np.argsort(srt_idx)]

    return qvals


def compute_fdr_scores(scores, y_data):
    """
    Compute FDR score threshold using the validation data
    :param scores: np.array, array of xgboost/CNN scores
    :param y_data: np.array, ground truth labels
    :param fdr_threshold: float, false discovery rate
    :return:
    """
    num_pos_class = len(np.where(y_data == 1)[0])
    num_neg_class = len(np.where(y_data == 0)[0])
    pn_ratio = num_pos_class / float(num_neg_class)
    print(f"pn_ratio: {pn_ratio}")
    ipd_fdr = tdc(scores, y_data, pn_ratio, desc=True)
    
    precision = 1-ipd_fdr
    
    sort_idx = np.argsort(precision)
    
    sorted_scores = scores[sort_idx]
    sorted_precision = precision[sort_idx]
    
    return sorted_scores, sorted_precision


def make_precision_score_table_from_npz(file):
    score_precision_table = np.load(file, 
                                    allow_pickle=True)
    
    cnn_scores = score_precision_table['cnn_scores']
    precisions = score_precision_table['precision']
    
    df = pd.DataFrame({"cnn_score": cnn_scores, 
                       "precision": precisions}).drop_duplicates()
    
    # make closest u8 to the precision float
    df["precision_u8"] = (df.precision* 255.0).round().astype(np.uint8)
    
    # group by equal u8 precisions and get 
    # the minimum cnn score that achieves that precision
    t=df.groupby("precision_u8").aggregate(
        {"cnn_score":"min"}).reset_index()
    
    return t[["cnn_score", "precision_u8"]]


def force_cudnn_initialization():
    """
    Force cuda to reinitialize.
    """
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev),
                               torch.zeros(s, s, s, s, device=dev))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--val_data',
        type=str,
        default="/net/noble/vol4/noble/user/anupamaj/proj/m6A-calling/data/PS00075_3.npz",
        help="path to the val npz file. Default is in the data directory"
    )

    parser.add_argument(
        '--model_load_path',
        type=str,
        default="models/m6ANet_PS00075_no_init.3.best_semi-supervised.torch",
        help="Path of the model to be stored."
    )
    parser.add_argument(
        '--save_score_precision_table',
        type=str,
        default="results/PS00075_3_precision_score_2.0.npz",
        help="Path of the model to be stored."
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help="Training on cpu or cuda. Default is cuda"
    )


    args = parser.parse_args()

    print(f"Predicting on {args.device}")

    if args.device == 'cuda':
        force_cudnn_initialization()

    
    model = torch.load(args.model_load_path, map_location=torch.device(args.device))
    # Get validation data.
    X_val, y_val, random_state = m6AGenerator(args.val_data,
                                              random_state=None)
    
    y_val_ohe = np.zeros((len(y_val), 2))
    y_val_ohe[np.where(y_val == 1)[0], 0] = 1
    y_val_ohe[np.where(y_val == 0)[0], 1] = 1
    
    X_val = torch.tensor(X_val).float()
    y_val_ohe = torch.tensor(y_val_ohe).float()
    X_val = X_val.to(args.device)
    y_val_ohe = y_val_ohe.to(args.device)
    
    preds_y = model.predict(X_val, device=args.device)
    
    sorted_scores, sorted_precision = compute_fdr_scores(preds_y[:, 0].cpu().numpy(), 
                                                         y_val_ohe[:, 0].cpu().numpy())
    
    np.savez(args.save_score_precision_table, cnn_scores=sorted_scores, 
             precision=sorted_precision)
    
    tbl = make_precision_score_table_from_npz(args.save_score_precision_table)
    tbl.to_json(f"{args.save_score_precision_table}.json", index=False, orient="split")