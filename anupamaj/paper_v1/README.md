# Fibertools: training supervised and semi-supervised CNN models for m6A detection from Fiber-seq HiFi reads. 

## Installation

Install pre-requisites using the following command: 

`pip install -r requirements.txt`

This code has been tested with Python 3.8.5. 

## Introduction
This directory code for training supervised and semi-supervised convolutional neural network for m6A detection from Fiber-seq reads. Fiber-seq is a long reads technology where the adenines in accessible 
chromatin regions are methylated. Subsequently these long reads are sequenced with PacBio sequencing and
methylated A are recognised using the nucleotide sequence and two other signals: inter-pulse distance and pulse width, the pulses for methlylated A's are longer than unmethylated bases. In this work, we train supervised and semi-supervised CNNs on two different PacBio long read chemistries and a new sequencer: 
1. 2.2 chemistry
2. 3.2 chemistry
3. Revio sequencer

Previously, the methylated Adenines (m6A) were identified using a machine learning based pipleine called IPDSummary from PacBio, but this pipeline has three significant short comings. First, it needs sub-read level information, which is no longer available with current sequencers like Revio. Second, IPDsummary is extremely slow, with run times of more than a day for a SMRT flowcell. Third, the IPDSummary code is closed-source. In this work, we first train supervised CNN models which beat IPDSummary and XGBoost models. The positive labels for the supervised CNN are derived from a Gaussian mixture model on top of IPDSummary m6A calls. The negative labels are derived from a nucleosome occupancy HMM model.

Lack of ground truth is a significant shortcoming of the supervised approach of m6A calling. Since the m6A labels are derived from IPDSummary, they can contain false positives. Therefore, we design a semi-supervised task, in line with target-decoy design for PSM identification in proteomics. We assume that our m6A set is a mixed population with false positives. 

## Data download

Download training and validation data for all three chemistries from Zenodo[TODO: add link] and place the downloaded data into the folder named `data` in the current directory. Ensure that the `train_data` and `val_data` variables in the `config.yml` file are pointing to the correct folder.

## Supervised CNN model

You can run all three versions of the CNN model with the following commands: 

`python m6a_supervised_cnn.py --train_chem train_2_2_chemistry`

`python m6a_supervised_cnn.py --train_chem train_3_2_chemistry`

`python m6a_supervised_cnn.py --train_chem train_revio_chemistry`

All required resources are defined in the `config.yml` file. See configuration section for more details on the resources. 

## Semi-supervised CNN model

To run the semi-supervised CNN model, run the following commands: 

`python m6a_semi_supervised_cnn.py --train_chem train_2_2_chemistry`

`python m6a_semi_supervised_cnn.py --train_chem train_3_2_chemistry`

`python m6a_semi_supervised_cnn.py --train_chem train_revio_chemistry`

## Inference on semi-supervised model

To generate precision at different CNN scores for all chemistries, run the following commands:

`python m6a_semi_supervised_cnn_predict.py --train_chem train_2_2_chemistry`

`python m6a_semi_supervised_cnn_predict.py --train_chem train_3_2_chemistry`

`python m6a_semi_supervised_cnn_predict.py --train_chem train_revio_chemistry`



## Configuration file for new chemistries
The configuration file is divided into three sections, one section for each new chemistry. For training a model for a new chemistry, simply add the required resources in the `config.yml` file under a new header and add the new header to the choices of train_chem in `m6a_supervised_cnn.py` and `m6a_semi_supervised_cnn.py`. Then run the following commands: 

`python m6a_supervised_cnn.py --train_chem new_chemistry`

`python m6a_supervised_cnn.py --train_chem new_chemistry`

Here is an example resources section: 

```
[train_2_2_chemistry]
fdr=0.05
device=cuda
input_size=6
supervised_train_epochs=30
semi_supervised_train_epochs=15
train_data=data/PS00109_2_train.npz
val_data=data/PS00109_3_val.npz
best_supervised_model_name=models/m6A_2_2_supervised_cnn.best.torch
final_supervised_model_name=models/m6A_2_2_supervised_cnn.final.torch
best_semi_supervised_model_name=models/m6A_2_2_semi_supervised_cnn.best.torch
final_semi_supervised_model_name=models/m6A_2_2_semi_supervised_cnn.final.torch
save_pos=results/semi_train_2_2_chemistry.npz
score_ap_json=results/semi_2_2_score_precision.json
score_ap_table=results/semi_2_2_score_precision.tsv
```

And here are the explanations for each variable:

```
fdr: acceptable false discovery rate for training the semi-supervised model
device: training device for models, cpu or cuda
input_size: number of input channels, currently 4 nucleotides, IPD and PW. 
supervised_train_epochs: number of training epochs for the supervised approach
semi_supervised_train_epochs: number of training epochs for the semi-supervised approach
train_data: path to training data
val_data: path to validation data
best_supervised_model_name: path and name for best supervised model
final_supervised_model_name: path and name for final supervised model
best_semi_supervised_model_name: path and name for best semi-supervised model
final_semi_supervised_model_name: path and name for final semi-supervised model
save_pos: path to save validation AUPR and # positives identified in validation set for semi-supervised approach
score_ap_json: json for cnn score to u8 precision conversion
score_ap_table: table with cnn scores, precision and u8 precision
```




