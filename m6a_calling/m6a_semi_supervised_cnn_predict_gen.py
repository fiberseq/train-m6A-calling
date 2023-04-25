#!/usr/bin/env python3
"""
m6a_semi_supervised_cnn_predict.py
Author: Anupama Jha <anupamaj@uw.edu>
This script runs the m6A model to
generate predictions and find score
threshold corresponding 95% precision
for the downstream fibertools pipeline.
"""
import torch
import argparse
import numpy as np
import configparser
import pandas as pd
import _pickle as pickle
from .m6a_cnn import M6ANet
from .m6a_semi_supervised_cnn import tdc, count_pos_neg, make_one_hot_encoded


def m6AGenerator(val_path, input_size, random_state=None):
    """
    This generator returns a validation
    features and labels.
    :param val_path: str, path where the
                          val data is stored.
    :param input_size: int, number of input
                            channels.
    :param random_state: np.random, random seed
    :return: X_val: validation features,
             y_val: validation labels
             random_state
    """
    # initialize Random state
    random_state = np.random.RandomState(random_state)

    # Load validation data
    val_data = np.load(val_path, allow_pickle=True)

    # Load validation features and
    # labels.
    X_val = val_data["features"]
    X_val = X_val[:, 0:input_size, :]
    y_val = val_data["labels"]

    print(
        f"Validation features shape {X_val.shape}, "
        f" validation labels shape: {y_val.shape}"
    )

    count_pos_neg(y_val, set_name="Validation")

    return X_val, y_val, random_state


def compute_fdr_scores(scores, y_data):
    """
    Compute FDR score threshold using
    the validation data
    :param scores: np.array, array of
                             CNN scores
    :param y_data: np.array, ground truth
                             labels
    :return: sorted scores, precision
    """
    num_pos_class = len(np.where(y_data == 1)[0])
    num_neg_class = len(np.where(y_data == 0)[0])

    pn_ratio = num_pos_class / float(num_neg_class)
    print(f"pn_ratio: {pn_ratio}")

    ipd_fdr = tdc(scores, y_data, pn_ratio, desc=True)

    precision = 1 - ipd_fdr

    sort_idx = np.argsort(precision)

    sorted_scores = scores[sort_idx]
    sorted_precision = precision[sort_idx]

    return sorted_scores, sorted_precision


def make_ap_table(cnn_scores, precisions):
    """
    Convert precisions to u8 for downstream
     processing
    :param cnn_scores: np.array, list of sorted
                                 cnn scores.
    :param precisions: np.array, and list of
                                 corresponding
                                 sorted precision.

    :return: pandas data frames with
             sorted scores and precision
    """
    df = pd.DataFrame(
        {"cnn_score": cnn_scores, "precision": precisions}
    ).drop_duplicates()

    # make closest u8 to the precision float
    df["precision_u8"] = (df.precision * 255.0).round().astype(np.uint8)

    # group by equal u8 precisions and get
    # the minimum cnn score that achieves that precision
    t = (
        df.groupby("precision_u8")
            .aggregate({"cnn_score": "min", "precision": "min"})
            .reset_index()
    )

    t_json = t[["cnn_score", "precision_u8"]]
    t_table = t[["cnn_score", "precision", "precision_u8"]]

    return t_json, t_table


def run(config_file, train_chem):
    """
    Run the m6A model to generate predictions
    and find score threshold corresponding
    95% precision for the downstream fibertools
    pipeline.
    :param config_file: str, path to the config
                            file with all details
    :param train_chem: str, section header from
                            the config file with
                            all resources for a
                            Fiber-seq chemistry
                            ML training. 
    :return: None.
    """
    # read parameters from config file
    config = configparser.ConfigParser()
    config.read(config_file)

    # get parameters for the relevant chemistry
    rel_config = config[train_chem]
    input_size = int(rel_config["input_size"])
    # path to validation data
    val_data = rel_config["semi_val_data_pr"]
    # run inference on cpu or cuda
    device = rel_config["device"]
    # path to model to use
    best_save_model = rel_config["best_semi_supervised_model_name"]
    # path to save tsv table with sorted scores,
    # precision and u8 precision score
    score_ap_table = rel_config["score_ap_table"]
    # path to save tsv table with sorted scores,
    # precision.
    score_ap_json = rel_config["score_ap_json"]

    # Load the model for
    # inference
    model = M6ANet()
    with open(best_save_model, "rb") as fp:
        model.load_state_dict(pickle.load(fp))
        
    model = model.to(device)

    # Get validation data.
    X_val, y_val, random_state = m6AGenerator(val_data, input_size, random_state=None)
    # convert to one hot encoded
    y_val_ohe = make_one_hot_encoded(y_val)

    # convert data to tensors
    X_val = torch.tensor(X_val).float()
    y_val_ohe = torch.tensor(y_val_ohe).float()
    X_val = X_val.to(device)
    y_val_ohe = y_val_ohe.to(device)

    # generate predictions
    preds_y = model.predict(X_val, device=device)

    # compute cnn scores and
    # corresponding precision
    sorted_scores, sorted_precision = compute_fdr_scores(
        preds_y[:, 0].cpu().numpy(), y_val_ohe[:, 0].cpu().numpy()
    )
    # get table and json to be stored
    # with sorted scores and precisions
    # without duplicates
    tbl_json, tbl_tsv = make_ap_table(sorted_scores, sorted_precision)
    # store json to be used by fibertools
    tbl_json.to_json(score_ap_json, index=False, orient="split")
    # save table for
    # paper
    tbl_tsv.to_csv(score_ap_table, sep="\t")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, default="paper_v1/config.yml", help="path to the config file."
    )

    parser.add_argument(
        "--train_chem",
        type=str,
        default="train_2_2_chemistry",
        help="Which chemistry to validate. The name should match the section header in the config file.",
    )

    args = parser.parse_args()

    print(f"Validating a {args.train_chem} " f"semi-supervised CNN model.")

    run(args.config_file, args.train_chem)


if __name__ == "__main__":
    main()
