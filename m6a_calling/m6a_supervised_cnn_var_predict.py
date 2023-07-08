#!/usr/bin/env python3
"""
m6a_supervised_cnn.py
Author: Anupama Jha <anupamaj@uw.edu>
This module predicts whether an
adenine is methylated or not. The
model is trained with Fiber-seq
HiFi read sequence, inter-pulse
distance and pulse width signal
from pacbio. The model is a
convolution neural network.
"""

import torch
import argparse
import numpy as np
import configparser
import _pickle as pickle
from .m6a_cnn import M6ANet
import matplotlib.pyplot as plt
from torchsummary import summary
from sklearn.metrics import average_precision_score, precision_recall_curve
from .m6a_semi_supervised_cnn import tdc, count_pos_neg, make_one_hot_encoded



def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    

def m6AGenerator(test_path, 
                 input_size, 
                 input_window, 
                 random_state=None):
    """
    This generator returns a validation
    features and labels.
    :param test_path: str, path where the
                          val data is stored.
    :param input_size: int, number of input
                            channels.
    :param input_window: int, size of input 
                              window on each side.
    :param random_state: np.random, random seed
    :return: X_val: validation features,
             y_val: validation labels
             random_state
    """
    # initialize Random state
    random_state = np.random.RandomState(random_state)

    # Load validation data
    test_data = np.load(test_path, allow_pickle=True)

    # Load validation features and
    # labels.
    X_val = test_data["features"]
    
    mid = int(np.floor(X_val.shape[2]/2.0))
    start = mid-input_window
    end = (mid+input_window+1)

    X_val = test_data["features"]
    X_val = X_val[:, 0:input_size, start:end]
    y_val = test_data["labels"]

    print(
        f"Validation features shape {X_val.shape}, "
        f" validation labels shape: {y_val.shape}"
    )

    count_pos_neg(y_val, set_name="Validation")
    
    # One-hot-encode val labels
    y_val_ohe = make_one_hot_encoded(y_val)

    return X_val, y_val_ohe, random_state



def run_model_var(config, 
                  train_chem, 
                  X_val, 
                  y_val_ohe):
    # path to model to use
    # get parameters for the relevant chemistry
    rel_config = config[train_chem]
    best_save_model = f"{rel_config['best_supervised_model_name']}"
    
    # model architecture
    rel_config = config[train_chem]
    device = rel_config["device"]
    input_size = int(rel_config["input_size"])
    sec_last_layer_size = int(rel_config["sec_last_layer_size"])
    last_layer_size = int(rel_config["last_layer_size"])
    out_channels_1 = int(rel_config["out_channels_1"])
    out_channels_2 = int(rel_config["out_channels_2"])
    out_channel_3 = int(rel_config["out_channel_3"])
    kernel_size_1 = int(rel_config["kernel_size_1"])
    kernel_size_2 = int(rel_config["kernel_size_2"])
    kernel_size_3 = int(rel_config["kernel_size_3"])
    output_shape = int(rel_config["output_shape"])
    

    # Move the model to appropriate
    # device
    model = M6ANet(input_size=input_size, 
                   sec_last_layer_size=sec_last_layer_size,
                   last_layer_size=last_layer_size,
                   output_shape=output_shape, 
                   out_channels_1=out_channels_1,
                   out_channels_2=out_channels_2,
                   out_channel_3=out_channel_3,
                   kernel_size_1=kernel_size_1,
                   kernel_size_2=kernel_size_2,
                   kernel_size_3=kernel_size_3).to(device)
    
    with open(best_save_model, "rb") as fp:
        model.load_state_dict(pickle.load(fp))
        
    model = model.to(device)
    
    # convert data to tensors
    X_val = torch.tensor(X_val).float()
    y_val_ohe = torch.tensor(y_val_ohe).float()
    X_val = X_val.to(device)
    y_val_ohe = y_val_ohe.to(device)

    # generate predictions
    preds_y = model.predict(X_val, device=device)
    return preds_y


def run(config_file, 
        train_chem_var1, 
        train_chem_var2, 
        train_chem_var3, 
        train_chem_var4):
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
    rel_config_var4 = config[train_chem_var4]
    input_size_var4 = int(rel_config_var4["input_size"])
    input_window_var4 = int(rel_config_var4["input_window"])
    # path to validation data
    test_data_var4 = rel_config_var4["sup_test_data"]
    # Get validation data.
    X_val_var4, y_val_ohe_var4, random_state = m6AGenerator(test_data_var4, 
                                              input_size_var4, 
                                              input_window_var4, 
                                              random_state=None)
    
    # Central IPD score classifier
    pred_y_var0 = X_val_var4[:, 4, 2]
    precision_var0, recall_var0, thresholds_var0 = precision_recall_curve(
        y_val_ohe_var4[:, 0], pred_y_var0
    )
    sklearn_ap_var0 = average_precision_score(
        y_val_ohe_var4[:, 0], pred_y_var0
    )  
    print(f"Central IDP, AUPR: {sklearn_ap_var0})")
    
    
    # CNN (5 bp + IPD)
    pred_y_var4 = run_model_var(config, 
                                train_chem=train_chem_var4, 
                                X_val=X_val_var4, 
                                y_val_ohe=y_val_ohe_var4)
    precision_var4, recall_var4, thresholds_var4 = precision_recall_curve(
        y_val_ohe_var4[:, 0], pred_y_var4.cpu().numpy()[:, 0]
    )
    sklearn_ap_var4 = average_precision_score(
        y_val_ohe_var4[:, 0], pred_y_var4.cpu().numpy()[:, 0]
    )
    print(f"CNN(5bp+IPD, AUPR: {sklearn_ap_var4})")
    
    
    # CNN (5 bp + IPD + PW)
    # get parameters for the relevant chemistry
    rel_config_var3 = config[train_chem_var3]
    input_size_var3 = int(rel_config_var3["input_size"])
    input_window_var3 = int(rel_config_var3["input_window"])
    # path to validation data
    test_data_var3 = rel_config_var3["sup_test_data"]
    # Get validation data.
    X_val_var3, y_val_ohe_var3, random_state = m6AGenerator(test_data_var3, 
                                              input_size_var3, 
                                              input_window_var3, 
                                              random_state=None)
    
    pred_y_var3 = run_model_var(config,
                                train_chem=train_chem_var3, 
                                X_val=X_val_var3, 
                                y_val_ohe=y_val_ohe_var3)
    precision_var3, recall_var3, thresholds_var3 = precision_recall_curve(
        y_val_ohe_var3[:, 0], pred_y_var3.cpu().numpy()[:, 0]
    )
    sklearn_ap_var3 = average_precision_score(
        y_val_ohe_var3[:, 0], pred_y_var3.cpu().numpy()[:, 0]
    )
    print(f"CNN(5bp+IPD+PW, AUPR: {sklearn_ap_var3})")

    
    
    # CNN (15 bp + IPD)
    # get parameters for the relevant chemistry
    rel_config_var2 = config[train_chem_var2]
    input_size_var2 = int(rel_config_var2["input_size"])
    input_window_var2 = int(rel_config_var2["input_window"])
    # path to validation data
    test_data_var2 = rel_config_var2["sup_test_data"]
    # Get validation data.
    X_val_var2, y_val_ohe_var2, random_state = m6AGenerator(test_data_var2, 
                                              input_size_var2, 
                                              input_window_var2, 
                                              random_state=None)

    
    pred_y_var2 = run_model_var(config,
                                train_chem=train_chem_var2, 
                                X_val=X_val_var2, 
                                y_val_ohe=y_val_ohe_var2)
    precision_var2, recall_var2, thresholds_var2 = precision_recall_curve(
        y_val_ohe_var2[:, 0], pred_y_var2.cpu().numpy()[:, 0]
    )
    sklearn_ap_var2 = average_precision_score(
        y_val_ohe_var2[:, 0], pred_y_var2.cpu().numpy()[:, 0]
    )
    print(f"CNN(15bp+IPD, AUPR: {sklearn_ap_var2})")
    
    
    
    
    # CNN (15 bp + IPD + PW) 
    # get parameters for the relevant chemistry
    rel_config_var1 = config[train_chem_var1]
    input_size_var1 = int(rel_config_var1["input_size"])
    input_window_var1 = int(rel_config_var1["input_window"])
    # path to validation data
    test_data_var1 = rel_config_var1["sup_test_data"]
    # Get validation data.
    X_val_var1, y_val_ohe_var1, random_state = m6AGenerator(test_data_var1, 
                                              input_size_var1, 
                                              input_window_var1, 
                                              random_state=None)
    
    pred_y_var1 = run_model_var(config,
                                train_chem=train_chem_var1, 
                                X_val=X_val_var1, 
                                y_val_ohe=y_val_ohe_var1)
    precision_var1, recall_var1, thresholds_var1 = precision_recall_curve(
        y_val_ohe_var1[:, 0], pred_y_var1.cpu().numpy()[:, 0]
    )
    sklearn_ap_var1 = average_precision_score(
        y_val_ohe_var1[:, 0], pred_y_var1.cpu().numpy()[:, 0]
    )
    print(f"CNN(15bp+IPD+PW, AUPR: {sklearn_ap_var1})")
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 6))
    ax.plot(recall_var0, precision_var0, c="#949fcf", label=f"Central IDP value, AUPR: {sklearn_ap_var0:0.3f})")
    ax.plot(recall_var4, precision_var4, c="#ff9696", label=f"CNN(5bp+IPD, AUPR: {sklearn_ap_var4:0.3f})")
    ax.plot(recall_var3, precision_var3, c="#d3af37", label=f"CNN(5bp+IPD+PW, AUPR: {sklearn_ap_var3:0.3f})")
    ax.plot(recall_var2, precision_var2, c="#536040", label=f"CNN(15bp+IPD, AUPR: {sklearn_ap_var2:0.3f})")
    ax.plot(recall_var1, precision_var1, c="#A020F0", label=f"CNN(15bp+IPD+PW, AUPR: {sklearn_ap_var1:0.3f})")
      
    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)

    ax.set_ylim([0.0, 1.0])
    ax.set_xlim([0.0, 1.0])
    
    simpleaxis(ax)
    ax.grid(True)

    plt.legend(bbox_to_anchor=(1.2, 1.0), fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{rel_config_var4['pr_fig']}", transparent=True)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 6))
    ax.plot(recall_var0, precision_var0, c="#949fcf", label=f"Central IDP value, AUPR: {sklearn_ap_var0:0.3f})")
    ax.plot(recall_var4, precision_var4, c="#ff9696", label=f"CNN(5bp+IPD, AUPR: {sklearn_ap_var4:0.3f})")
    ax.plot(recall_var3, precision_var3, c="#d3af37", label=f"CNN(5bp+IPD+PW, AUPR: {sklearn_ap_var3:0.3f})")
    ax.plot(recall_var2, precision_var2, c="#536040", label=f"CNN(15bp+IPD, AUPR: {sklearn_ap_var2:0.3f})")
    ax.plot(recall_var1, precision_var1, c="#A020F0", label=f"CNN(15bp+IPD+PW, AUPR: {sklearn_ap_var1:0.3f})")
      

    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)

    ax.set_ylim([0.5, 1.0])
    ax.set_xlim([0.0, 1.0])
    
    simpleaxis(ax)
    ax.grid(True)

    plt.legend(bbox_to_anchor=(1.2, 1.0), fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{rel_config_var4['pr_fig_subset']}", transparent=True)
    


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file", type=str, default="paper_v1/config_variants.yml", help="path to the config file."
    )

    parser.add_argument(
        "--train_chem_var1",
        type=str,
        default="train_2_2_chem_var1",
        help="Which chemistry to validate. The name should match the section header in the config file.",
    )
    
    parser.add_argument(
        "--train_chem_var2",
        type=str,
        default="train_2_2_chem_var2",
        help="Which chemistry to validate. The name should match the section header in the config file.",
    )
    
    parser.add_argument(
        "--train_chem_var3",
        type=str,
        default="train_2_2_chem_var3",
        help="Which chemistry to validate. The name should match the section header in the config file.",
    )
    
    parser.add_argument(
        "--train_chem_var4",
        type=str,
        default="train_2_2_chem_var4",
        help="Which chemistry to validate. The name should match the section header in the config file.",
    )

    args = parser.parse_args()

    run(args.config_file, 
        args.train_chem_var1, 
        args.train_chem_var2, 
        args.train_chem_var3, 
        args.train_chem_var4)


if __name__ == "__main__":
    main()
